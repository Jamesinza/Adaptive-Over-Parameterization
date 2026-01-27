"""adaptive_overparam_experiment_v24.py"""

import os
import math
import random
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from contextlib import contextmanager
from sklearn.decomposition import PCA
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import tensorflow as tf
import keras

# -----------------------------
# Config / seeds
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Dataset helpers
# -----------------------------
def make_dataset(data, n, seq_len):
    X = np.empty([n - seq_len, seq_len, 1], dtype=np.float32)
    Y = np.empty([n - seq_len], dtype=np.float32)
    for i in range(n - seq_len):
        X[i] = data[i:i + seq_len]
        Y[i] = data[i + seq_len, -1]
    return X, Y

def compute_batch_size(dataset_length: int) -> int:
    base_unit = 16_384
    base_batch = 32
    scale = math.ceil(max(1, dataset_length) / base_unit)
    return int(base_batch * scale)

def numpy_batch_iter(X, Y, batch_size):
    n = len(X)
    for i in range(0, n, batch_size):
        yield X[i:i+batch_size], Y[i:i+batch_size]

# -----------------------------
# Load data
# -----------------------------
raw_df = pd.read_csv('datasets/EURUSD_M1_245.csv')
df = raw_df.tail(10_000).copy()
df = df[df['High'] != df['Low']]
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)
data = df[['Close']].values
split = 1000
train_data = data[:-split]
val_data = data[-split:]
batch_size = compute_batch_size(len(train_data))
CONFIG = {
    'seq_len': 8,
    'train_size': len(train_data),
    'val_size': len(val_data),
    'batch_size': batch_size,
    'epochs_per_phase': 10,
    'cycles': 64,
    'regrow_finetune_steps': 5,
    'mask_thresh_min': 0.1,
    'mask_thresh_multi': 0.05,
    'prune_fraction': 0.25,
    'regrow_fraction': 0.5,
    'connection_rewire_fraction': 0.25,
    'd_model': 8,
    'hidden_units': 32,
    'hidden_layers': 2,
    'num_heads': 8,
    'learning_rate': 1e-3,
    'max_fisher_samples_per_batch': 16,
    'log_dir': './logs/adaptive_overparam_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    'cycle_patience': 10,
    'cycle_min_delta': 1e-10,
    'cycle_monitor': 'val_loss_final',
    'save_best_cycle_model': True,
}
MASK_THRESH_MIN = CONFIG['mask_thresh_min']
os.makedirs(CONFIG['log_dir'], exist_ok=True)
X_train, Y_train = make_dataset(train_data, CONFIG['train_size'], CONFIG['seq_len'])
X_val, Y_val = make_dataset(val_data, CONFIG['val_size'], CONFIG['seq_len'])

# Create built-in normalization layer for the model
norm = keras.layers.Normalization()
norm.adapt(X_train)

# Use NumPy tuples instead of tf.data pipelines to avoid exhaustion issues while prototyping
train_ds = (X_train, Y_train)
val_ds = (X_val, Y_val)
print('Train samples:', len(X_train), 'Val samples:', len(X_val), 'Batch size:', CONFIG['batch_size'])

# -----------------------------
# Positional Encoding
# -----------------------------
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(pe, dtype=tf.float32)

# -----------------------------
# Masked Dense (serializable)
# -----------------------------
@keras.utils.register_keras_serializable(package="Custom")
class MaskedDense(keras.layers.Layer):
    def __init__(self, units, activation=None, name=None):
        super().__init__(name=name)
        self.units = int(units)
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        w_init = tf.random_normal_initializer(stddev=0.02)
        self.kernel = self.add_weight(
            name='kernel',
            shape=(last_dim, self.units),
            initializer=w_init,
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        # neuron-level mask
        self.mask = self.add_weight(
            name='mask',
            shape=(self.units,),
            initializer='ones',
            trainable=False
        )
        # connection-level mask (same shape as kernel)
        self.conn_mask = self.add_weight(
            name='conn_mask',
            shape=(last_dim, self.units),
            initializer='ones',
            trainable=False
        )
        self.conn_age = self.add_weight(
            name='conn_age',
            shape=(last_dim, self.units),
            initializer='zeros',
            trainable=False
        )        

    def call(self, inputs, return_pre_activation=False):
        masked_kernel = self.kernel * self.conn_mask * tf.reshape(self.mask, (1, -1))
        masked_bias = self.bias * self.mask
        pre = tf.matmul(inputs, masked_kernel) + masked_bias
        out = self.activation(pre) if self.activation is not None else pre
        return (pre, out) if return_pre_activation else out

    # neuron-level operations
    def prune_units(self, indices):
        m = self.mask.numpy()
        m[indices] = 0.0
        self.mask.assign(m)

    def regrow_units(self, indices):
        """
        Reactivate the given neuron indices:
         - restore mask bit
         - reinit kernel column and bias
         - reset conn_age for all input connections to this output neuron to 0
        """
        if len(indices) == 0:
            return

        # update mask
        m = self.mask.numpy()
        m[indices] = 1.0
        self.mask.assign(m)

        # reinit kernel columns and biases
        k = self.kernel.numpy()
        b = self.bias.numpy()
        for idx in indices:
            k[:, idx] = np.random.normal(scale=0.02, size=(k.shape[0],))
            b[idx] = 0.0

        # assign kernel and bias once
        self.kernel.assign(k)
        self.bias.assign(b)

        # reset ages for all inputs to these regrown output neurons
        conn_age = self.conn_age.numpy()
        conn_age[:, indices] = 0.0
        self.conn_age.assign(conn_age)

    # connection-level operations
    def prune_connections(self, idx_pairs):
        if len(idx_pairs) == 0:
            return
        m = self.conn_mask.numpy()
        for (i, j) in idx_pairs:
            m[i, j] = 0.0
        self.conn_mask.assign(m)

    def regrow_connections(self, idx_pairs):
        """
        Bring specific input->output connections back to life;
        initialize weights and reset their ages.
        
        idx_pairs: iterable of (i, j) pairs
        """
        if len(idx_pairs) == 0:
            return
        m = self.conn_mask.numpy()
        k = self.kernel.numpy()
        conn_age = self.conn_age.numpy()
        for (i, j) in idx_pairs:
            m[i, j] = 1.0
            k[i, j] = np.random.normal(scale=0.02)
            conn_age[i, j] = 0.0
        self.conn_mask.assign(m)
        self.kernel.assign(k)
        self.conn_age.assign(conn_age)

    def get_config(self):
        return {'units': self.units, 'activation': keras.activations.serialize(self.activation)}

# -----------------------------
# Transformer block
# -----------------------------
@keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.att = keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model // self.num_heads)
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
        self.ffn_dense1 = MaskedDense(self.ff_dim, activation='gelu', name='ffn_dense1')
        self.ffn_dense2 = MaskedDense(self.d_model, activation=None, name='ffn_dense2')

    def call(self, x, training=False):
        attn_out = self.att(x, x, x)
        x = self.norm1(x + attn_out)
        seq_len = tf.shape(x)[1]
        x_flat = tf.reshape(x, (-1, tf.shape(x)[-1]))
        pre, h = self.ffn_dense1(x_flat, return_pre_activation=True)

        h2 = self.ffn_dense2(h)
        h2 = tf.reshape(h2, (-1, seq_len, self.d_model))
        x = self.norm2(x + h2)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model,
                       'num_heads': self.num_heads,
                       'ff_dim': self.ff_dim,
                       # 'drop_rate': self.drop_rate
                      })
        return config

# -----------------------------
# Model builder
# -----------------------------
def build_transformer_model(seq_len, d_model, num_layers, num_heads, ff_dim):
    inp = keras.layers.Input(shape=(seq_len, 1))
    x = norm(inp)
    x = keras.layers.Dense(d_model)(x)
    pe = positional_encoding(seq_len, d_model)
    x = x + pe
    for i in range(num_layers):
        x = TransformerBlock(d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, name=f'transformer_block_{i}')(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    out = keras.layers.Dense(1, name='head')(x)
    model = keras.Model(inp, out)
    return model

model = build_transformer_model(CONFIG['seq_len'], d_model=CONFIG['d_model'], num_layers=CONFIG['hidden_layers'],
                                num_heads=CONFIG['num_heads'], ff_dim=CONFIG['hidden_units'])
optim = keras.optimizers.AdamW(CONFIG['learning_rate'])
model.compile(optim, loss='huber', metrics=['mae'])

# -----------------------------
# Utilities: robust discovery of MaskedDense sublayers (recursive)
# -----------------------------
def get_masked_dense_layers(model):
    visited = set()
    found = []

    def walk(layer):
        if id(layer) in visited:
            return
        visited.add(id(layer))
        try:
            if isinstance(layer, MaskedDense):
                found.append(layer)
        except Exception:
            pass
        for attr in ("layers", "submodules", "_layers", "_saved_model_inputs_spec"):
            children = getattr(layer, attr, None)
            if not children:
                continue
            try:
                for child in children:
                    if child is layer:
                        continue
                    walk(child)
            except Exception:
                pass

    if hasattr(model, "layers") and getattr(model, "layers"):
        for top in model.layers:
            walk(top)
    # also guard-walk the model object itself to catch other nesting
    walk(model)

    out = []
    seen = set()
    for l in found:
        if id(l) in seen:
            continue
        out.append(l)
        seen.add(id(l))
    return out

# -----------------------------
# Capture monkeypatch helper for subclassed MaskedDense
# -----------------------------
@contextmanager
def capture_maskeddense_outputs(masked_layers, record_tensors=False):
    """
    Monkeypatch MaskedDense.call on the provided instances and record outputs.
    Records keyed by unique_key = f"{layer.name}_{hex(id(layer))}".

    Safe against late-binding closure issues: orig method is explicitly bound into the wrapper factory.
    If record_tensors=True the wrapper will store raw tensors (useful for gradient/jacobian computation).
    """
    records = {}
    originals = {}

    for layer in masked_layers:
        key = f"{layer.name}_{hex(id(layer))}"
        records[key] = []
        originals[key] = layer.call  # bound method

        def make_wrapper(orig_method, key_ref, record_tensors_ref):
            # orig_method and key_ref are captured as *defaults* to avoid late-binding
            def wrapper(inputs, *args, **kwargs):
                # Call the original bound method exactly
                res = orig_method(inputs, *args, **kwargs)
                # Decide what to record: if (pre,out) then record out, else record res
                try:
                    if isinstance(res, (tuple, list)) and len(res) == 2:
                        out_tensor = res[1]
                    else:
                        out_tensor = res
                except Exception:
                    out_tensor = res
                # Store tensor or numpy according to flag
                if record_tensors_ref:
                    records[key_ref].append(out_tensor)
                else:
                    try:
                        records[key_ref].append(out_tensor.numpy())
                    except Exception:
                        # fallback: store object if numpy extraction fails
                        records[key_ref].append(out_tensor)
                return res
            return wrapper

        # bind explicit original to avoid capturing loop variable `layer` etc.
        orig = originals[key]
        layer.call = make_wrapper(orig, key, record_tensors)

    try:
        yield records
    finally:
        # restore originals (safe even if some layers were removed)
        for layer in masked_layers:
            key = f"{layer.name}_{hex(id(layer))}"
            if key in originals:
                layer.call = originals[key]


# -----------------------------
# Capture activations (NumPy-aware)
# -----------------------------
def capture_activations(model, dataset, masked_layers=None, batch_size=None):
    """
    Capture activations for MaskedDense layers.
    dataset may be a tf.data.Dataset or a tuple (X_np, Y_np).
    Returns dict keyed by unique instance key: { 'ffn_dense1_0xabc': np.array([...]) , ... }
    """
    if masked_layers is None:
        masked_layers = get_masked_dense_layers(model)
    if len(masked_layers) == 0:
        return {}
    if batch_size is None:
        batch_size = CONFIG['batch_size']

    submodels = {}
    acts = {}
    for l in masked_layers:
        key = f"{l.name}_{hex(id(l))}"
        acts[key] = []
        try:
            submodels[key] = keras.Model(model.input, l.output)
        except Exception:
            submodels[key] = None

    need_monkey = [l for l in masked_layers if submodels.get(f"{l.name}_{hex(id(l))}") is None]

    # choose iterator
    if isinstance(dataset, tuple) and isinstance(dataset[0], np.ndarray):
        batch_iter = numpy_batch_iter(dataset[0], dataset[1], batch_size)
    else:
        batch_iter = iter(dataset)

    seen_any = False
    for x, y in batch_iter:
        seen_any = True
        x_tf = tf.convert_to_tensor(x)

        # functional submodels
        for l in masked_layers:
            key = f"{l.name}_{hex(id(l))}"
            sm = submodels.get(key)
            if sm is None:
                continue
            try:
                a = sm(x_tf, training=False).numpy()
            except Exception:
                continue
            if a.ndim == 3:
                a = a.reshape(a.shape[0] * a.shape[1], a.shape[2])
            acts[key].append(a)

        # monkeypatch path
        if need_monkey:
            # ---- robust eager toggle helper ----
            def _get_run_eagerly_flag():
                return bool(getattr(tf.config, "functions_run_eagerly", lambda: False)())

            prev = _get_run_eagerly_flag()
            tf.config.run_functions_eagerly(True)
            try:
                with capture_maskeddense_outputs(need_monkey) as records:
                    _ = model(x_tf, training=False)
                    for l in need_monkey:
                        key = f"{l.name}_{hex(id(l))}"
                        outs = records.get(key, [])
                        if not outs:
                            continue
                        arrs = []
                        for o in outs:
                            if isinstance(o, np.ndarray):
                                o_np = o
                            else:
                                try:
                                    o_np = o.numpy()
                                except Exception:
                                    continue
                            if o_np.ndim == 3:
                                o_np = o_np.reshape(o_np.shape[0] * o_np.shape[1], o_np.shape[2])
                            arrs.append(o_np)
                        if arrs:
                            acts[key].append(np.concatenate(arrs, axis=0))
            finally:
                tf.config.run_functions_eagerly(bool(prev))

    if not seen_any:
        print('capture_activations: WARNING - dataset yielded zero batches (empty or exhausted).')
        final = {}
        for l in masked_layers:
            key = f"{l.name}_{hex(id(l))}"
            final[key] = np.zeros((0, l.units), dtype=np.float32)
        return final

    final = {}
    for l in masked_layers:
        key = f"{l.name}_{hex(id(l))}"
        if len(acts[key]) == 0:
            final[key] = np.zeros((0, l.units), dtype=np.float32)
        else:
            final[key] = np.concatenate(acts[key], axis=0)
    return final

# -----------------------------
# Activation entropy
# -----------------------------
def compute_activation_entropy(activations, eps=1e-9):
    if activations.size == 0:
        return np.array([])
        
    n_samples, units = activations.shape
    ent = np.zeros((units,), dtype=np.float32)
    
    for u in range(units):
        vals = activations[:, u]
        vmin, vmax = vals.min(), vals.max()
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            continue
        if vmax - vmin < 1e-6:
            ent[u] = 0.0
            continue
        if vmax== vmin:
            vmax += 1e-6

        try:
            bins = np.histogram_bin_edges(vals, bins='auto', range=(vmin, vmax))
        except Exception:
            bins = np.linspace(vmin, vmax, 10)
            
        hist, _ = np.histogram(vals, bins=bins)
        p = hist / (hist.sum() + eps)
        p = p[p > 0]
        ent[u] = -np.sum(p * np.log(p))
    return ent

# -----------------------------
# Improved Fisher: per-sample gradients wrt layer outputs (NumPy-aware)
# -----------------------------
def approx_fisher_per_sample(model, dataset, loss_fn, max_samples_per_batch=8, batch_size=None):
    """
    Improved per-batch Fisher:
      - Prebuilds per-layer @tf.function per-example functions (reduces retracing).
      - Uses tf.vectorized_map over a small per-batch prefix (n_samples) for speed.
      - Falls back to monkeypatch + tape.jacobian for subclassed layers.
    Notes:
      - Keep max_samples_per_batch small (<=8) to bound memory.
      - This implementation tries hard to avoid retracing by moving tf.function creation
        outside the per-batch loop.
    """
    md_layers = get_masked_dense_layers(model)
    if batch_size is None:
        batch_size = CONFIG['batch_size']

    # Build functional submodels where possible
    submodels = {}
    for l in md_layers:
        key = f"{l.name}_{hex(id(l))}"
        try:
            submodels[key] = keras.Model(model.input, l.output)
        except Exception:
            submodels[key] = None

    # Prepare vmapped per-layer functions (only for layers with functional submodels).
    # Each per_example_fn accepts (x_i, y_i) where x_i shape = (seq_len, features), y_i = scalar
    vmapped_fns = {}  # key -> callable that accepts (x_small, y_small) and returns (n_samples, units)
    seq_len = CONFIG['seq_len']
    feature_dim = 1  # your model uses shape (seq_len, 1)

    for l in md_layers:
        key = f"{l.name}_{hex(id(l))}"
        subm = submodels.get(key)
        if subm is None:
            continue

        # Create a tf.function-wrapped per-example function once.
        # Input signature uses fixed per-example shapes to limit retracing.
        @tf.function(input_signature=[
            tf.TensorSpec(shape=(seq_len, feature_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        ])
        def per_example_fn(x_i, y_i, _subm=subm):
            # note: _subm captured to avoid closure issues
            x_b = tf.expand_dims(x_i, 0)   # (1, seq_len, feat)
            y_b = tf.expand_dims(y_i, 0)
            with tf.GradientTape() as tape:
                out_b = _subm(x_b, training=False)    # (1, ..., units)
                tape.watch(out_b)
                pred_b = model(x_b, training=False)   # (1, 1) or (1,)
                loss_b = tf.reshape(loss_fn(y_b, pred_b), [])  # scalar
            g = tape.gradient(loss_b, out_b)
            if g is None:
                # return zeros of unit size
                # determine units from out_b's static shape if possible, otherwise dynamic
                units = tf.shape(out_b)[-1]
                return tf.zeros((units,), dtype=tf.float32)
            g0 = g[0]  # remove leading batch dim -> (..., units)
            # collapse temporal dims (if any) into per-unit magnitude
            if tf.rank(g0) == 1:
                unit_mag = tf.abs(g0)
            else:
                sq = tf.square(g0)
                reduce_axes = tf.range(tf.rank(g0) - 1)  # all axes except last
                reduced = tf.sqrt(tf.reduce_sum(sq, axis=reduce_axes))
                unit_mag = reduced
            return tf.cast(unit_mag, tf.float32)  # (units,)

        # Wrap into a vmapped caller function (not decorated) that we can call with tensors.
        def make_vmapped(fn):
            def call_vmapped(x_small, y_small):
                # ensure we're in graph mode for vectorized_map; vectorized_map will enable tf.function if needed
                try:
                    return tf.vectorized_map(lambda elems: fn(elems[0], elems[1]), (x_small, y_small))
                except Exception as e:
                    # fallback to python loop if vectorized_map fails for some reason
                    lst = []
                    for i in range(tf.shape(x_small)[0]):
                        lst.append(fn(x_small[i], y_small[i]))
                    return tf.stack(lst, axis=0)
            return call_vmapped

        vmapped_fns[key] = make_vmapped(per_example_fn)

    # initialize accumulators
    fisher = {f"{l.name}_{hex(id(l))}": np.zeros((l.units,), dtype=np.float32) for l in md_layers}
    count = 0

    # choose iterator
    if isinstance(dataset, tuple) and isinstance(dataset[0], np.ndarray):
        batch_iter = numpy_batch_iter(dataset[0], dataset[1], batch_size)
    else:
        batch_iter = iter(dataset)

    if max_samples_per_batch > 8:
        print("approx_fisher_per_sample: Warning — large max_samples_per_batch may be slow or memory heavy. Recommend <=8.")

    for x_batch, y_batch in batch_iter:
        x_batch_tf = tf.convert_to_tensor(x_batch, dtype=tf.float32)
        y_batch_tf = tf.convert_to_tensor(y_batch, dtype=tf.float32)
        B = x_batch_tf.shape[0]
        n_samples = int(min(int(B), int(max_samples_per_batch)))
        if n_samples == 0:
            continue
        x_small = x_batch_tf[:n_samples]
        y_small = y_batch_tf[:n_samples]

        # Partition layers: those with vmapped_fns and those that need monkeypatch/jacobian
        have_vmapped = []
        need_jacobian = []
        for l in md_layers:
            key = f"{l.name}_{hex(id(l))}"
            if key in vmapped_fns:
                have_vmapped.append((l, key))
            else:
                need_jacobian.append((l, key))

        # --- Run vmapped per-layer functions (fast path) ---
        if have_vmapped:
            # Ensure graph-mode for vectorized_map (turn off eager if it was on)
            prev_run_eager = bool(getattr(tf.config, "functions_run_eagerly", lambda: False)())
            if prev_run_eager:
                tf.config.run_functions_eagerly(False)
            try:
                for l, key in have_vmapped:
                    try:
                        per_example_units = vmapped_fns[key](x_small, y_small)  # (n_samples, units)
                    except Exception:
                        # fallback: try python loop inside TF (slower)
                        per_list = []
                        for i in range(n_samples):
                            per_list.append(vmapped_fns[key](x_small[i:i+1, ...], y_small[i:i+1, ...])[0])
                        per_example_units = tf.stack(per_list, axis=0)
                    # accumulate squared gradients per unit
                    batch_sum = tf.reduce_sum(tf.square(per_example_units), axis=0)  # (units,)
                    try:
                        fisher[key] += batch_sum.numpy()
                    except Exception:
                        fisher[key] += np.array(batch_sum)
                count += n_samples * len(have_vmapped)
            finally:
                # restore previous eager mode
                tf.config.run_functions_eagerly(bool(prev_run_eager))

        # --- Monkeypatch + jacobian fallback for layers that couldn't build submodels ---
        if need_jacobian:
            # Capture tensors by monkeypatching; ensure eager to capture actual tensors
            prev = bool(getattr(tf.config, "functions_run_eagerly", lambda: False)())
            tf.config.run_functions_eagerly(True)
            try:
                with capture_maskeddense_outputs([l for (l, _) in need_jacobian], record_tensors=True) as records:
                    _ = model(x_small, training=False)
                    outs = {}
                    for l, key in need_jacobian:
                        rec = records.get(key, [])
                        if not rec:
                            continue
                        outs[key] = rec[0]
            finally:
                tf.config.run_functions_eagerly(bool(prev))

            if outs:
                # compute loss vector and jacobian once
                with tf.GradientTape() as tape:
                    for out_t in outs.values():
                        tape.watch(out_t)
                    preds = model(x_small, training=False)
                    loss_vec = tf.reshape(loss_fn(y_small, preds), [-1])  # (n_samples,)

                for key, out_t in outs.items():
                    try:
                        jac = tape.jacobian(loss_vec, out_t)  # may be large: (n_samples, n_samples, ... )
                    except Exception:
                        continue
                    if jac is None:
                        continue
                    # flatten trailing dims into a single dimension
                    try:
                        jac_flat = tf.reshape(jac, (n_samples, n_samples, -1))
                    except Exception:
                        # robust reshape if static dims are None
                        total_trailing = tf.reduce_prod(tf.shape(out_t)[1:]) if tf.rank(out_t) > 1 else 1
                        jac_flat = tf.reshape(jac, (n_samples, n_samples, total_trailing))
                    # gather diagonal elements
                    idx = tf.stack([tf.range(n_samples), tf.range(n_samples)], axis=1)
                    per_sample_flat = tf.gather_nd(jac_flat, idx)  # (n_samples, P_flat)
                    # reconstruct per-sample per-unit and collapse non-last dims
                    out_dyn = tf.shape(out_t)
                    if tf.size(out_dyn) > 1:
                        # last dimension is units; reshape accordingly
                        units = tf.shape(out_t)[-1]
                        per_sample = tf.reshape(per_sample_flat, tf.concat([[n_samples], [-1, units]][:2], axis=0))
                        # if there were time dims, collapse them: sum squares across time dims
                        if tf.rank(per_sample) > 2:
                            sq = tf.square(per_sample)
                            reduce_axes = list(range(1, tf.rank(per_sample) - 1))
                            unit_sq = tf.reduce_sum(sq, axis=reduce_axes)
                        else:
                            unit_sq = tf.square(per_sample)
                    else:
                        unit_sq = tf.square(per_sample_flat)
                    batch_sum = tf.reduce_sum(unit_sq, axis=0)
                    try:
                        fisher[key] += batch_sum.numpy()
                    except Exception:
                        fisher[key] += np.array(batch_sum)
                    count += n_samples

    # Normalize
    if count == 0:
        return fisher
    for name in fisher:
        fisher[name] /= (count + 1e-9)
    return fisher

# ---------------------------------------------------------------------------
# Configuration dataclass (extended for layer-wise control + quality metrics)
# ---------------------------------------------------------------------------
@dataclass
class EvoConfig:
    base_prune: float = CONFIG['prune_fraction']
    regrow_fraction: float = CONFIG['regrow_fraction']
    connection_rewire_fraction: float = CONFIG['connection_rewire_fraction']
    regrow_loss_tol: float = 0.01
    regrow_finetune_steps: int = CONFIG['regrow_finetune_steps']
    finetune_batch_size: Optional[int] = CONFIG['batch_size']//2
    prune_search: bool = False
    mask_thresh_multi: float = CONFIG['mask_thresh_multi']
    mask_thresh_min: float = CONFIG['mask_thresh_min']
    log_dir: Optional[str] = CONFIG['log_dir']
    debug: bool = False
    max_prune_frac: float = 0.5
    min_prune_frac: float = 0.01
    compress_policy: str = 'try_longer_retrain'  # 'rollback' | 'accept_if_compression_win' | 'try_longer_retrain'
    compression_tradeoff: float = 0.15
    long_retrain_epochs: int = 10

    # New layer-wise options
    layerwise_prune: bool = True                 # perform sequential layer-wise testing & pruning
    layer_prune_rel_tol: float = 0.005           # per-layer allowed relative loss increase when accepting a layer prune
    layer_regrow_min_rel_loss: float = 0.002     # regrow only layers that individually caused at least this rel-loss

    # New quality-metric toggles and weights
    use_grad_importance: bool = True             # compute activation-gradient importance
    use_weight_grad: bool = True                 # compute kernel * grad(kernel) importance for connections
    grad_unit_weight: float = 0.6
    grad_weight_weight: float = 0.3
    fisher_weight: float = 0.1
    max_grad_samples: int = 64                   # cap number of val batches to use for gradient-based scoring


# ---------------------------------------------------------------------------
# Pareto logger
# ---------------------------------------------------------------------------
class ParetoLogger:
    def __init__(self):
        self.points: List[Tuple[int, float]] = []

    def add(self, alive_count: int, val_loss: float):
        self.points.append((int(alive_count), float(val_loss)))

    def last(self):
        return self.points[-1] if self.points else None


# ---------------------------------------------------------------------------
# EvoCompressor (modular, layer-aware pruning/regrowth)
# ---------------------------------------------------------------------------
class EvoCompressor:
    def __init__(
        self,
        model: keras.Model,
        val_ds,
        config: EvoConfig = EvoConfig(),
        train_ds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.model = model
        self.val_ds = val_ds
        self.train_ds = train_ds
        self.config = config

        # records
        self.history: List[Dict[str, Any]] = []
        self.pareto_logger = ParetoLogger()
        self.conn_age_log = defaultdict(list)

        # store last layerwise decisions for regrowth logic / debugging
        self.last_layer_decisions: Dict[str, Dict[str, Any]] = {}

        # internal
        self._setup_logger()

    # ---------------------- logging -----------------------------------------
    def _setup_logger(self):
        name = 'EvoCompressor'
        self.logger = logging.getLogger(name)
        # avoid duplicated handlers if user re-instantiates multiple compressors
        if not self.logger.handlers:
            level = logging.DEBUG if self.config.debug else logging.INFO
            self.logger.setLevel(level)
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            ch.setLevel(level)
            self.logger.addHandler(ch)

            if self.config.log_dir:
                os.makedirs(self.config.log_dir, exist_ok=True)
                fh = logging.FileHandler(os.path.join(self.config.log_dir, 'evo_compressor.log'))
                fh.setFormatter(fmt)
                fh.setLevel(level)
                self.logger.addHandler(fh)

        # convenience
        self.log = self.logger

    # ---------------------- high-level API ---------------------------------
    def evaluate_and_compress(self, train_loss: Optional[float] = None, val_loss: Optional[float] = None, do_regrow: bool = True) -> Tuple[Dict[str, Any], float]:
        """Main entrypoint: two-phase compress (prune -> optional regrow) returning (summary, mask_thresh_min)

        This method mirrors the original logic but routes steps to modular methods. The key difference:
        - if config.layerwise_prune is True, pruning decisions are taken sequentially per-layer (safer, but more evals)
        - we maintain neuron-only and connection-only scenarios (no 'both' simultaneous scenario).
        """
        self.log.info('Starting evaluate_and_compress cycle')

        md_layers = get_masked_dense_layers(self.model)

        # compute utilities
        utilities = self.compute_utilities(md_layers)

        # decide dynamic prune fraction
        prune_fraction = self._compute_dynamic_prune_fraction(train_loss, val_loss)
        self.log.info(f'Computed prune_fraction={prune_fraction:.4f}')

        # baseline val loss
        baseline_val_loss = self._obtain_baseline_loss(val_loss)
        self.log.info(f'Baseline validation loss: {baseline_val_loss:.9f}')

        # compute prune candidates (no mutation yet)
        stored_prune_info, total_alive_pre, mask_thresh_min = self.compute_prune_candidates(md_layers, utilities, prune_fraction)
        self.log.info(f'Total alive neurons pre-prune: {total_alive_pre}')

        # decide scenarios and evaluate - now layer-aware
        best_scenario, pruned_val_loss, baseline_weights = self._search_or_apply_scenarios(stored_prune_info, total_alive_pre)
        self.log.info(f'Chosen pruning scenario: {best_scenario} => pruned_val_loss={pruned_val_loss:.9f}')

        regrow_triggered = False
        val_loss_after_regrow = pruned_val_loss

        # regrow decision (global), but we will prefer layer-targeted regrowth
        if do_regrow and (pruned_val_loss > baseline_val_loss * (1.0 + self.config.regrow_loss_tol)):
            self.log.info('Pruned model degraded beyond tolerance; evaluating finetune before regrow')
            val_after_ft = self._finetune_short()
            self.log.info(f'Val loss after short finetune: {val_after_ft:.9f}')

            if val_after_ft <= baseline_val_loss * (1.0 + self.config.regrow_loss_tol):
                self.log.info('Short finetune recovered performance; skipping regrowth')
                val_loss_after_regrow = val_after_ft
            else:
                # select layers to target for regrowth based on per-layer deltas recorded during layerwise prune
                self.log.info('Short finetune did NOT recover performance; performing selective regrowth')
                regrow_triggered = True

                # find target layers that individually caused the largest relative loss increases
                target_layers = []
                for lname, info in stored_prune_info.items():
                    rel = info.get('rel_delta', 0.0)
                    if rel >= self.config.layer_regrow_min_rel_loss:
                        target_layers.append(lname)

                if not target_layers:
                    # fallback: if no per-layer deltas available, regrow everything
                    self.log.debug('No per-layer delta exceeded threshold; regrowing across all layers')
                    target_layers = list(stored_prune_info.keys())

                self.perform_regrowth(stored_prune_info, utilities, target_layers=target_layers)

                # evaluate after regrowth
                val_metrics_after = self._evaluate_val()
                val_loss_after_regrow = float(val_metrics_after[0])
                self.log.info(f'Val loss after regrowth: {val_loss_after_regrow:.9f}')

        # finalize stats and compute final loss
        df_rows = self.finalize_stats()
        val_metrics_final = self._evaluate_val()
        val_loss_final = float(val_metrics_final[0])
        self.log.info(f'Final validation loss: {val_loss_final:.9f}')

        # run post-compression decision logic
        baseline_weights_for_decision = baseline_weights
        action, reason = self._post_compression_decision(
            baseline_val_loss=baseline_val_loss,
            val_loss_final=val_loss_final,
            baseline_alive=total_alive_pre,
            final_alive=int(sum(r['alive_neurons'] for r in df_rows)),
            baseline_weights=baseline_weights_for_decision,
            policy=self.config.compress_policy,
            rel_tol=0.01,
            compression_tradeoff=self.config.compression_tradeoff,
            long_retrain_epochs=self.config.long_retrain_epochs,
            train_ds=self.train_ds,
            finetune_bs=self.config.finetune_batch_size
        )
        self.log.info(f'Post-compression decision: action={action}; reason={reason}')

        # recompute df_rows if rollback
        if action.startswith('rolled_back'):
            df_rows = self.finalize_stats()  # recompute to reflect rollback
            val_metrics_final = self._evaluate_val()
            val_loss_final = float(val_metrics_final[0])

        # pareto / history
        total_alive_now = int(sum(r['alive_neurons'] for r in df_rows))
        self.pareto_logger.add(total_alive_now, val_loss_final)

        summary = {
            'alive_total': total_alive_now,
            'prune_fraction_used': float(prune_fraction),
            'baseline_val_loss': baseline_val_loss,
            'pruned_val_loss': float(pruned_val_loss),
            'regrow_triggered': bool(regrow_triggered),
            'val_loss_after_regrow': float(val_loss_after_regrow),
            'val_loss_final': float(val_loss_final),
            'action': action,
            'reason': reason,
        }
        self.history.append(summary)

        # persist snapshot CSV
        if self.config.log_dir:
            df = pd.DataFrame(df_rows)
            fname = os.path.join(self.config.log_dir, f'compressor_snapshot_{len(self.history)}.csv')
            df.to_csv(fname, index=False)
            self.log.debug(f'Wrote snapshot CSV: {fname}')

        # print age monitor
        for lname, stats in self.conn_age_log.items():
            last = stats[-1]
            self.log.info(f"[AGE MONITOR] {lname}: mean={last['mean']:.2f}, std={last['std']:.2f}, max={last['max']:.0f}, alive={last['alive_count']}")

        return summary, mask_thresh_min

    # ---------------------- utility computation -----------------------------
    def compute_utilities(self, md_layers) -> Dict[str, np.ndarray]:
        """Compute per-layer and per-unit utilities that reflect QUALITY, not just raw activity.

        New approach:
        - Keep fisher (approx second-order importance) as a stable signal
        - Add gradient-based unit importance: mean(abs(activation * dL/dactivation)) across validation samples
        - Add weight-gradient importance for connections: mean(abs(kernel * dkernel)) across samples
        - Combine normalized signals using config weights to produce a single importance score per unit

        Returns a dict mapping unique layer name -> per-unit utility array (higher == more important)
        """
        self.log.debug('Computing quality-aware utilities (grad-based + fisher)')

        # 1) activation stats (no gradients)
        acts = capture_activations(self.model, self.val_ds, masked_layers=md_layers)
        entropies = {name: compute_activation_entropy(a) for name, a in acts.items()}

        # 2) fisher (approx second-order)
        fisher = approx_fisher_per_sample(
            self.model, self.val_ds, keras.losses.Huber(),
            max_samples_per_batch=CONFIG['max_fisher_samples_per_batch']
        )

        utilities: Dict[str, np.ndarray] = {}

        # 3) gradient-based importance (per-unit and per-connection)
        # We'll compute gradients over a limited number of batches for efficiency
        use_grad = getattr(self.config, 'use_grad_importance', True)
        use_wgrad = getattr(self.config, 'use_weight_grad', True)
        max_batches = max(1, int(self.config.max_grad_samples))

        # Prepare a small generator over validation data (numpy tuple or tf.data)
        if isinstance(self.val_ds, tuple) and isinstance(self.val_ds[0], np.ndarray):
            X_val, y_val = self.val_ds
            val_iter = numpy_batch_iter(X_val, y_val, CONFIG['batch_size'])
        else:
            # assume tf.data
            val_iter = iter(self.val_ds)

        # cached grads and counts per layer
        grad_unit_accum = {f"{l.name}_{hex(id(l))}": None for l in md_layers}
        grad_unit_count = {k: 0 for k in grad_unit_accum}
        grad_weight_accum = {k: None for k in grad_unit_accum}

        # We'll construct per-layer submodels to expose activation tensors and compute gradients
        layer_submodels = {}
        try:
            for layer in md_layers:
                unique_name = f"{layer.name}_{hex(id(layer))}"
                # model outputs: (layer_activation, full_model_output)
                layer_submodels[unique_name] = keras.Model(self.model.inputs, [layer.output, self.model.output])
        except Exception as e:
            self.log.warning(f'Failed to build layer submodels for gradient importance: {e}')
            use_grad = False
            use_wgrad = False

        # iterate over batches (bounded by max_batches)
        batches_done = 0
        loss_fn = keras.losses.get(keras.losses.Huber())
        for batch in val_iter:
            if batches_done >= max_batches:
                break
            batches_done += 1

            if isinstance(self.val_ds, tuple) and isinstance(self.val_ds[0], np.ndarray):
                xb, yb = batch
            else:
                xb, yb = batch

            xb_tf = tf.convert_to_tensor(xb)
            yb_tf = tf.convert_to_tensor(yb)

            if use_grad and layer_submodels:
                for layer in md_layers:
                    unique_name = f"{layer.name}_{hex(id(layer))}"
                    subm = layer_submodels.get(unique_name)
                    if subm is None:
                        continue
                    with tf.GradientTape() as tape:
                        # forward pass: capture activation and preds
                        act, preds = subm(xb_tf, training=False)
                        tape.watch(act)
                        loss = loss_fn(yb_tf, preds)
                    # gradient wrt activation (shape: batch x units)
                    grad_act = tape.gradient(loss, act)
                    if grad_act is None:
                        continue
                    act_np = act.numpy()
                    grad_np = grad_act.numpy()
                    # importance per-unit: mean(abs(act * grad)) across batch
                    per_unit_imp = np.mean(np.abs(act_np * grad_np), axis=0)

                    if grad_unit_accum[unique_name] is None:
                        grad_unit_accum[unique_name] = per_unit_imp
                    else:
                        grad_unit_accum[unique_name] += per_unit_imp
                    grad_unit_count[unique_name] += 1

                    # weight gradient importance: gradient wrt kernel
                    if use_wgrad:
                        try:
                            # compute gradient w.r.t. kernel (connection importance)
                            kernel = layer.kernel
                            with tf.GradientTape() as tape2:
                                act2, preds2 = subm(xb_tf, training=False)
                                loss2 = loss_fn(yb_tf, preds2)
                            grad_kernel = tape2.gradient(loss2, kernel)
                            if grad_kernel is not None:
                                grad_k_np = grad_kernel.numpy()
                                w_imp = np.abs(kernel.numpy() * grad_k_np)
                                if grad_weight_accum[unique_name] is None:
                                    grad_weight_accum[unique_name] = w_imp
                                else:
                                    grad_weight_accum[unique_name] += w_imp
                        except Exception as e:
                            # kernel gradient may fail for some custom layers; ignore per-layer
                            self.log.debug(f'weight-grad failed for {unique_name}: {e}')

        # finalize accumulators (average)
        for layer in md_layers:
            unique_name = f"{layer.name}_{hex(id(layer))}"

            if grad_unit_accum.get(unique_name) is not None and grad_unit_count.get(unique_name, 0) > 0:
                gunit = grad_unit_accum[unique_name] / float(max(1, grad_unit_count[unique_name]))
            else:
                gunit = np.zeros((layer.units,), dtype=np.float32)

            if grad_weight_accum.get(unique_name) is not None:
                gweight = grad_weight_accum[unique_name] / float(max(1, grad_unit_count[unique_name]))
            else:
                gweight = np.zeros_like(layer.kernel.numpy())

            # normalize signals
            # fisher per-unit exists? fisher returns per-unit or per-layer? we expect per-unit
            f = fisher.get(unique_name, np.zeros_like(gunit))

            # normalize to zero-mean unit-std to combine
            def norm(x):
                x = np.asarray(x, dtype=np.float32)
                if x.size == 0:
                    return x
                x = x - np.median(x)
                s = np.std(x) + 1e-9
                return (x / s)

            gunit_n = norm(gunit)
            f_n = norm(f)

            # combined per-unit importance score (higher = more important)
            alpha = float(self.config.grad_unit_weight)
            beta = float(self.config.fisher_weight)
            unit_score = alpha * gunit_n + beta * f_n

            utilities[unique_name] = unit_score
            # store weight importance as side-channel in stored structures via attributes on layer if needed
            # but we return only per-unit utilities for pruning neurons. Connection pruning will use gweight.
            # temporarily attach to layer for later use
            setattr(layer, '_last_grad_weight_importance', gweight)
            setattr(layer, '_last_grad_unit_importance', gunit)
            setattr(layer, '_last_fisher', f)
            setattr(layer, '_last_unit_score', unit_score)

            self.log.debug(f'Computed quality utility for {unique_name} (units={unit_score.size})')

        return utilities

    # ---------------------- candidate generation ---------------------------
    def compute_prune_candidates(self, md_layers, utilities: Dict[str, np.ndarray], prune_fraction: float):
        self.log.debug('Computing prune candidates for each masked-dense layer (quality-aware)')
        stored_prune_info = {}
        total_alive_pre = 0

        for unique_name, score in utilities.items():
            layer = next((l for l in md_layers if f"{l.name}_{hex(id(l))}" == unique_name), None)
            if layer is None:
                self.log.warning(f'No matching layer object for {unique_name} — skipping')
                continue

            mask_thresh_min = self.config.mask_thresh_min * (1.0 + self.config.mask_thresh_multi)
            mask_threshold = min(mask_thresh_min, 0.5)
            self.log.debug(f'Using mask threshold {mask_threshold:.4f} for {unique_name}')

            mask = layer.mask.numpy()
            conn_mask = layer.conn_mask.numpy()
            alive_idx = np.where(mask > mask_threshold)[0]
            total_alive_pre += len(alive_idx)

            prune_idx = np.zeros((0,), dtype=int)
            n_prune = 0
            if len(alive_idx) > 0:
                # reverse-sort by quality: low-score = low quality -> prune
                scores_alive = score[alive_idx]
                # lower scores mean lower importance (since we normalized around median)
                n_prune = max(1, int(len(alive_idx) * prune_fraction))
                prune_pos = np.argsort(scores_alive)[:n_prune]
                prune_idx = alive_idx[prune_pos]

            # connection pruning candidates using grad-weight importance if available
            W = layer.kernel.numpy()
            conn_age = layer.conn_age.numpy()
            # take per-connection importance from attached _last_grad_weight_importance if set
            gweight = getattr(layer, '_last_grad_weight_importance', None)
            if gweight is None or gweight.size == 0:
                # fallback: abs weight with age penalty as before
                absW = np.abs(W) * (conn_mask > mask_threshold)
                max_age = np.max(conn_age) if np.max(conn_age) > 0 else 1.0
                age_penalty = np.tanh(conn_age / (max_age + 1e-6))
                score_matrix = absW * (1.0 - 0.5 * age_penalty)
            else:
                # when grad-weight importance available, lower importance means prune
                # normalize and combine with age penalty
                gw = np.abs(gweight) * (conn_mask > mask_threshold)
                max_age = np.max(conn_age) if np.max(conn_age) > 0 else 1.0
                age_penalty = np.tanh(conn_age / (max_age + 1e-6))
                score_matrix = gw * (1.0 - 0.5 * age_penalty)

            flat_score = score_matrix.flatten()
            n_rewire = int(flat_score.size * self.config.connection_rewire_fraction)

            low_pairs = np.zeros((0, 2), dtype=int)
            if n_rewire > 0:
                nonzero_count = np.count_nonzero(flat_score)
                k_prune = min(n_rewire, nonzero_count)
                if k_prune > 0:
                    low_idx = np.argpartition(flat_score, k_prune)[:k_prune]
                    low_pairs = np.array(np.unravel_index(low_idx, conn_mask.shape)).T

            stored_prune_info[unique_name] = {
                'prune_idx': prune_idx,
                'n_prune': int(n_prune),
                'conn_low_pairs': low_pairs,
                'n_rewire': int(n_rewire),
                # placeholders for layerwise bookkeeping
                'val_loss_if_pruned': None,
                'rel_delta': 0.0,
                'val_loss_neurons': None,
                'val_loss_connections': None,
                'chosen_mode': None,
            }
            self.log.debug(f'Layer {unique_name}: n_prune={n_prune}, n_rewire={int(n_rewire)}')

        # Snapshot baseline weights
        baseline_weights = self.model.get_weights()
        return stored_prune_info, int(total_alive_pre), mask_thresh_min

    # ---------------------- scenario evaluation ----------------------------
    def _apply_prune_scenario(self, stored_prune_info, scenario: str):
        """Apply pruning for the provided stored_prune_info. Scenario must be 'neurons' or 'connections'."""
        md_layers = get_masked_dense_layers(self.model)
        for unique_name, info in stored_prune_info.items():
            layer = next((l for l in md_layers if f"{l.name}_{hex(id(l))}" == unique_name), None)
            if layer is None:
                continue
            if scenario == 'neurons':
                if info.get('prune_idx', None) is not None and getattr(info['prune_idx'], 'size', 0) > 0:
                    self.log.debug(f'Applying neuron prune to {unique_name}: n={len(info["prune_idx"])}')
                    layer.prune_units(info['prune_idx'])
            elif scenario == 'connections':
                if info.get('conn_low_pairs', None) is not None and getattr(info['conn_low_pairs'], 'size', 0) > 0:
                    self.log.debug(f'Applying connection prune to {unique_name}: n_pairs={len(info["conn_low_pairs"])})')
                    layer.prune_connections(info['conn_low_pairs'])
            else:
                self.log.warning(f'Unknown prune scenario requested: {scenario} (supported: neurons, connections)')

    def _evaluate_val(self):
        if isinstance(self.val_ds, tuple) and isinstance(self.val_ds[0], np.ndarray):
            return self.model.evaluate(self.val_ds[0], self.val_ds[1], verbose=0)
        else:
            return self.model.evaluate(self.val_ds, verbose=0)

    def _apply_prune_layerwise(self, stored_prune_info, baseline_weights, rel_tol):
        """Sequentially test and accept/reject pruning per-layer.

        Strategy:
        - Start from baseline weights
        - Iterate layers ordered by potential compression (descending n_prune)
        - For each layer: test neuron-only and connection-only pruning separately
        - Choose the mode with lowest val loss; accept if within rel_tol of current baseline
        - Record per-layer val_loss and rel_delta and chosen_mode
        """
        self.log.info('[LAYERWISE] Starting sequential layer-wise prune evaluation (neurons vs connections)')
        accepted = []
        rejected = []
        # sort layers by n_prune descending - prefer big wins first
        items = sorted(stored_prune_info.items(), key=lambda kv: kv[1].get('n_prune', 0), reverse=True)

        current_baseline = baseline_weights
        self.model.set_weights(current_baseline)
        baseline_val = float(self._evaluate_val()[0])
        self.log.debug(f'[LAYERWISE] initial baseline_val={baseline_val:.9f}')

        for unique_name, info in items:
            # skip no-op layers
            if info.get('n_prune', 0) <= 0 and info.get('n_rewire', 0) <= 0:
                self.log.debug(f'[LAYERWISE] skipping {unique_name} (no prune candidates)')
                rejected.append(unique_name)
                info['val_loss_if_pruned'] = baseline_val
                info['rel_delta'] = 0.0
                info['chosen_mode'] = 'none'
                continue

            # reset to current baseline (cumulative accepted prunes)
            self.model.set_weights(current_baseline)

            # Test neuron-only pruning
            self._apply_prune_scenario({unique_name: info}, 'neurons')
            metrics_neu = self._evaluate_val()
            v_neu = float(metrics_neu[0])
            self.log.debug(f'[LAYERWISE] {unique_name} neuron-only val_loss={v_neu:.9f}')

            # revert to baseline before testing connections
            self.model.set_weights(current_baseline)

            # Test connection-only pruning
            self._apply_prune_scenario({unique_name: info}, 'connections')
            metrics_conn = self._evaluate_val()
            v_conn = float(metrics_conn[0])
            self.log.debug(f'[LAYERWISE] {unique_name} connection-only val_loss={v_conn:.9f}')

            # decide which mode is better (lower val loss)
            if v_neu <= v_conn:
                chosen_mode = 'neurons'
                chosen_val = v_neu
            else:
                chosen_mode = 'connections'
                chosen_val = v_conn

            delta = (chosen_val - baseline_val) / (abs(baseline_val) + 1e-12)

            info['val_loss_neurons'] = v_neu
            info['val_loss_connections'] = v_conn
            info['val_loss_if_pruned'] = chosen_val
            info['rel_delta'] = delta
            info['chosen_mode'] = chosen_mode

            # Apply chosen mode for real and accept/reject based on rel_tol
            self.model.set_weights(current_baseline)
            self._apply_prune_scenario({unique_name: info}, chosen_mode)

            if chosen_val <= baseline_val * (1.0 + rel_tol):
                # accept this layer's prune and update baseline to include it
                accepted.append(unique_name)
                current_baseline = self.model.get_weights()
                baseline_val = chosen_val
                self.log.info(f'[LAYERWISE] ACCEPTED {unique_name} ({chosen_mode}): rel_delta={delta:.9f} -> new baseline_val={baseline_val:.9f}')
            else:
                # reject this layer's prune; revert
                rejected.append(unique_name)
                self.model.set_weights(current_baseline)
                self.log.info(f'[LAYERWISE] REJECTED {unique_name} ({chosen_mode}): rel_delta={delta:.9f}')

        # persist decisions for later regrowth logic / debugging
        self.last_layer_decisions = {k: v for k, v in stored_prune_info.items()}
        return accepted, rejected, current_baseline, baseline_val

    def _search_or_apply_scenarios(self, stored_prune_info, total_alive_pre):
        baseline_weights = self.model.get_weights()

        # if user asked for layerwise pruning, run the sequential layerwise acceptance pass
        if self.config.layerwise_prune:
            accepted, rejected, new_baseline, new_val = self._apply_prune_layerwise(stored_prune_info, baseline_weights, self.config.layer_prune_rel_tol)
            # ensure model is left in the pruned-accepted state
            self.model.set_weights(new_baseline)
            pruned_val_loss = float(new_val)
            best_scenario = 'layerwise'
            # baseline_weights for decision should be the original (pre-prune) snapshot
            self.log.info(f'[LAYERWISE] accepted_layers={len(accepted)} rejected_layers={len(rejected)}')
            return best_scenario, pruned_val_loss, baseline_weights

        # fallback to old behavior (network-level scenarios) but only neurons and connections
        if self.config.prune_search:
            scenarios = ['neurons', 'connections']
            self.log.info(f'[PRUNE-SEARCH] evaluating {scenarios}')
        else:
            scenarios = ['neurons', 'connections']

        scenario_results = {}
        for scen in scenarios:
            self.model.set_weights(baseline_weights)
            self._apply_prune_scenario(stored_prune_info, scen)
            metrics = self._evaluate_val()
            scenario_val_loss = float(metrics[0])
            scenario_results[scen] = {'val_loss': scenario_val_loss}
            self.log.info(f'[PRUNE-SEARCH] scenario={scen}, val_loss={scenario_val_loss:.9f}')

        best_scenario = min(scenario_results.keys(), key=lambda k: scenario_results[k]['val_loss'])
        pruned_val_loss = float(scenario_results[best_scenario]['val_loss'])

        # ensure the model is left pruned according to best scenario
        self.model.set_weights(baseline_weights)
        self._apply_prune_scenario(stored_prune_info, best_scenario)

        return best_scenario, pruned_val_loss, baseline_weights

    # ---------------------- short finetune ---------------------------------
    def _finetune_short(self) -> float:
        if self.train_ds is None or self.config.regrow_finetune_steps <= 0:
            self.log.debug('No training data or finetune steps disabled; skipping finetune')
            metrics = self._evaluate_val()
            return float(metrics[0])

        batch_size = self.config.finetune_batch_size or CONFIG['batch_size']
        ft_iter = numpy_batch_iter(self.train_ds[0], self.train_ds[1], batch_size)
        steps_done = 0
        for xb, yb in ft_iter:
            self.model.train_on_batch(xb, yb)
            steps_done += 1
            if steps_done >= self.config.regrow_finetune_steps:
                break
        metrics_after = self._evaluate_val()
        return float(metrics_after[0])

    # ---------------------- regrowth --------------------------------------
    def perform_regrowth(self, stored_prune_info, utilities: Dict[str, np.ndarray], target_layers: Optional[List[str]] = None):
        md_layers = get_masked_dense_layers(self.model)
        self.log.debug('Performing regrowth across layers (selective if targets provided)')

        # if target_layers is None -> act on all layers
        if target_layers is None:
            target_layers = list(stored_prune_info.keys())

        for unique_name, info in stored_prune_info.items():
            if unique_name not in target_layers:
                continue

            layer = next((l for l in md_layers if f"{l.name}_{hex(id(l))}" == unique_name), None)
            if layer is None:
                continue

            # neuron regrowth
            n_prune = info['n_prune']
            if n_prune <= 0 or self.config.regrow_fraction <= 0:
                n_regrow = 0
            else:
                n_regrow = int(np.random.binomial(n_prune, self.config.regrow_fraction))

            if n_regrow > 0:
                dead_indices = np.where(layer.mask.numpy() == 0)[0]
                if len(dead_indices) >= n_regrow:
                    regrow_idx = np.random.choice(dead_indices, size=n_regrow, replace=False)
                else:
                    available = info['prune_idx'] if info['prune_idx'].size else dead_indices
                    regrow_idx = np.random.choice(available, size=min(n_regrow, len(available)), replace=False)
                layer.regrow_units(regrow_idx)
                self.log.info(f'[REGROW] {layer.name}: regrew {len(regrow_idx)} neurons')

            # connection regrowth
            regrew_conn_count = 0
            n_rewire = info['n_rewire']
            if n_rewire > 0:
                W = layer.kernel.numpy()
                out_util = utilities.get(unique_name, np.zeros((layer.units,)))
                if out_util.size == layer.units:
                    out_u = out_util.copy().astype(np.float32)
                    out_u = (out_u - out_u.min()) / (out_u.max() - out_u.min() + 1e-9)
                else:
                    out_u = np.ones((layer.units,), dtype=np.float32)
                in_u = np.mean(np.abs(W), axis=1) if W.size else np.zeros((W.shape[0],))
                if in_u.size:
                    in_u = (in_u - in_u.min()) / (in_u.max() - in_u.min() + 1e-9)
                else:
                    in_u = np.zeros_like(in_u)
                dead_pairs = np.argwhere(layer.conn_mask.numpy() < 0.5)
                if dead_pairs.size > 0:
                    alpha_out = 0.6
                    beta_in = 0.4
                    di = dead_pairs[:, 0]; dj = dead_pairs[:, 1]
                    desirability = (alpha_out * out_u[dj]) + (beta_in * in_u[di])
                    desirability += np.random.rand(len(desirability)) * 1e-6
                    k = min(n_rewire, len(dead_pairs))
                    if k > 0:
                        choose_idx = np.argpartition(-desirability, k - 1)[:k]
                        regrow_pairs = dead_pairs[choose_idx]
                        layer.regrow_connections(regrow_pairs)
                        regrew_conn_count = len(regrow_pairs)
                        self.log.info(f'[REGROW] {layer.name}: regrew {regrew_conn_count} connections')

    # ---------------------- finalize --------------------------------------
    def finalize_stats(self):
        md_layers_final = get_masked_dense_layers(self.model)
        df_rows = []
        total_alive = 0
        utilities = {}  # optional; could be passed through if desired
        for layer in md_layers_final:
            mask = layer.mask.numpy()
            conn_mask = layer.conn_mask.numpy()
            conn_age = layer.conn_age.numpy()
            alive_ages = conn_age[conn_mask > 0.5]
            if alive_ages.size > 0:
                self.conn_age_log[layer.name].append({
                    'mean': float(np.mean(alive_ages)),
                    'std': float(np.std(alive_ages)),
                    'max': float(np.max(alive_ages)),
                    'alive_count': int(np.sum(conn_mask > 0.5))
                })
            total_alive += int(mask.sum())
            conn_alive = int(conn_mask.sum())
            conn_density = conn_alive / float(np.prod(conn_mask.shape)) if conn_mask.size else 0.0
            alive_neurons = int(mask.sum())
            alive_weights = layer.kernel.numpy()[conn_mask > 0.5]
            avg_conn_mag = float(np.mean(np.abs(alive_weights))) if alive_weights.size else 0.0
            unique_name = f"{layer.name}_{hex(id(layer))}"
            df_rows.append({
                'layer': unique_name,
                'alive_neurons': alive_neurons,
                'alive_connections': conn_alive,
                'conn_density': conn_density,
                'utility_mean': float(utilities.get(unique_name, np.zeros((layer.units,))).mean()) if unique_name in utilities else 0.0,
                'avg_conn_mag': avg_conn_mag
            })
        return df_rows

    # ---------------------- helpers ---------------------------------------
    def _compute_dynamic_prune_fraction(self, train_loss, val_loss):
        prune_fraction = self.config.base_prune
        if train_loss is not None and val_loss is not None:
            gap = float(val_loss - train_loss)
            rel = gap / (abs(val_loss) + 1e-9)
            prune_fraction = float(self.config.base_prune * (1.0 + rel))
            prune_fraction = np.clip(prune_fraction, self.config.min_prune_frac, self.config.max_prune_frac)
        return float(prune_fraction)

    def _obtain_baseline_loss(self, val_loss):
        if val_loss is not None:
            return float(val_loss)
        else:
            metrics = self._evaluate_val()
            return float(metrics[0])

    # ---------------------- post compression decision ----------------------
    def _post_compression_decision(self,
                                  baseline_val_loss,
                                  val_loss_final,
                                  baseline_alive,
                                  final_alive,
                                  baseline_weights,
                                  policy='rollback',
                                  rel_tol=0.01,
                                  compression_tradeoff=0.15,
                                  long_retrain_epochs=10,
                                  train_ds=None,
                                  finetune_bs=None):
        # (Use the original logic but send logs to self.log and return standardized output)
        baseline_val_loss = float(baseline_val_loss)
        val_loss_final = float(val_loss_final)
        rel_increase = (val_loss_final - baseline_val_loss) / (abs(baseline_val_loss) + 1e-12)
        compression_gain = (baseline_alive - final_alive) / (baseline_alive + 1e-12)
        if val_loss_final <= baseline_val_loss * (1.0 + 1e-12):
            return 'kept', f'improved or equal val loss ({val_loss_final:.6g} <= {baseline_val_loss:.6g})'
        if policy == 'rollback':
            if rel_increase <= rel_tol:
                return 'kept', f'loss increase {rel_increase:.4f} <= tol {rel_tol}; keep pruned'
            else:
                self.model.set_weights(baseline_weights)
                return 'rolled_back', f'loss increase {rel_increase:.4f} > tol {rel_tol}; restored baseline'
        if policy == 'accept_if_compression_win':
            if rel_increase <= rel_tol:
                return 'kept', f'loss increase {rel_increase:.4f} within tol {rel_tol}'
            if compression_gain >= compression_tradeoff and rel_increase <= 0.05:
                return 'kept', f'accepted pruned model: compression_gain={compression_gain:.3f} rel_loss={rel_increase:.3f}'
            self.model.set_weights(baseline_weights)
            return 'rolled_back', f'loss increase {rel_increase:.4f} unacceptable; compression_gain={compression_gain:.3f} too small'
        if policy == 'try_longer_retrain':
            use_train = train_ds if train_ds is not None else getattr(self, 'train_ds', None)
            if use_train is None:
                self.model.set_weights(baseline_weights)
                return 'rolled_back', 'no training data available for longer retrain; rolled back'
            epochs = int(long_retrain_epochs)
            batch_size = finetune_bs or getattr(self.config, 'finetune_batch_size', None) or CONFIG['batch_size']
            pruned_snapshot = self.model.get_weights()
            tmp_es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
            self.model.fit(use_train[0], use_train[1], batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[tmp_es])
            metrics_after = self._evaluate_val()
            val_after = float(metrics_after[0])
            rel_after = (val_after - baseline_val_loss) / (abs(baseline_val_loss) + 1e-12)
            if val_after <= baseline_val_loss * (1.0 + rel_tol):
                return 'retrained_and_kept', f'longer retrain recovered val loss: {val_after:.6g}'
            else:
                self.model.set_weights(baseline_weights)
                return 'retrained_and_rolled_back', f'retrain failed (rel_after={rel_after:.4f}); restored baseline'
        return 'kept', 'fallback: keeping pruned model by default'


# -----------------------------
# Training + cycles
# -----------------------------
def run_experiment(model, train_ds, val_ds, config, mask_thresh_min):
    """
    Runs training + compression cycles with outer (per-cycle) early stopping.

    Config (dict) optional keys:
      - 'cycle_patience' : int (default 3)   -> number of cycles with no improvement to stop
      - 'cycle_min_delta': float (default 1e-4) -> minimum absolute improvement to qualify
      - 'cycle_monitor'  : str (default 'val_loss_final') -> metric to monitor:
            'val_loss_final' (compressor summary),
            'pruned_val_loss' (compressor pruned loss),
            'val_loss' (last training validation loss from history[-1]).
      - 'save_best_cycle_model' : bool (default True) -> persist best cycle model to log_dir
    """
    log_dir = config.get('log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)

    # cycle-level early stopping config
    cycle_patience = int(config.get('cycle_patience', 3))
    cycle_min_delta = float(config.get('cycle_min_delta', 1e-4))
    cycle_monitor = config.get('cycle_monitor', 'val_loss_final')
    save_best_cycle = bool(config.get('save_best_cycle_model', True))

    # callbacks for warmup
    chkpt_path = os.path.join(log_dir, 'best_base_model.keras')
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=log_dir),
        keras.callbacks.ModelCheckpoint(chkpt_path, save_best_only=True, monitor='val_loss'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=3, cooldown=10),
    ]
    print('\nWarmup training...\n')
    history = model.fit(
        train_ds[0], train_ds[1],
        batch_size=config['batch_size'],
        validation_data=(val_ds[0], val_ds[1]),
        epochs=5,
        callbacks=callbacks,
        verbose=2
    )

    # outer-cycle early-stop bookkeeping
    best_cycle_metric = np.inf
    best_cycle_idx = -1
    best_cycle_weights = None
    best_cycle_model_path = os.path.join(log_dir, 'best_cycle_overall.keras')
    cycles_no_improve = 0

    total_cycles = int(config.get('cycles', 10))
    for cycle in range(total_cycles):
        model.summary()
        # create compressor with fresh config per-cycle if desired
        cfg = EvoConfig(mask_thresh_min=mask_thresh_min, log_dir=log_dir, debug=True, prune_search=True)
        compressor = EvoCompressor(model, val_ds, config=cfg, train_ds=train_ds)

        print(f'\n--- Cycle {cycle + 1}/{total_cycles} ---\n')
        chkpt_path = os.path.join(log_dir, f'best_cycle_{cycle+1}_model.keras')
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=log_dir),
            keras.callbacks.ModelCheckpoint(chkpt_path, save_best_only=True, monitor='val_loss'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.99, patience=3, cooldown=0, min_lr=5e-6),
        ]

        history = model.fit(
            train_ds[0], train_ds[1],
            batch_size=config['batch_size'],
            validation_data=(val_ds[0], val_ds[1]),
            epochs=config['epochs_per_phase'],
            callbacks=callbacks,
            verbose=2
        )

        # run compression for this cycle
        train_loss = float(history.history['loss'][-1])
        val_loss = float(history.history['val_loss'][-1])
        summary, mask_thresh_min = compressor.evaluate_and_compress(train_loss=train_loss, val_loss=val_loss)
        print('\nCompressor summary:', summary,'\n')

        # choose metric to monitor for cycle-level early stopping
        if cycle_monitor == 'val_loss_final':
            current_metric = float(summary.get('val_loss_final', np.inf))
        elif cycle_monitor == 'pruned_val_loss':
            current_metric = float(summary.get('pruned_val_loss', np.inf))
        elif cycle_monitor == 'val_loss':
            # fallback to last validation loss from training
            current_metric = val_loss
        else:
            # unrecognized monitor -> default to val_loss_final
            current_metric = float(summary.get('val_loss_final', np.inf))

        # improvement check (absolute improvement by min_delta)
        improved = (best_cycle_metric - current_metric) > cycle_min_delta
        if improved:
            best_cycle_metric = current_metric
            best_cycle_idx = cycle
            # keep an in-memory copy of weights (fast)
            best_cycle_weights = [w.copy() for w in model.get_weights()]
            cycles_no_improve = 0
            # also persist full model if requested (restores optimizer state if reloaded)
            if save_best_cycle and log_dir:
                try:
                    model.save(best_cycle_model_path, overwrite=True, include_optimizer=True)
                    print(f'[CYCLE-ES] Saved best-cycle model (cycle={cycle+1}) -> {best_cycle_model_path}')
                except Exception as e:
                    print(f'[CYCLE-ES] Warning: failed to save best-cycle model to disk: {e}')
            print(f'[CYCLE-ES] New best cycle {cycle+1}: {cycle_monitor}={current_metric:.9g}')
        else:
            cycles_no_improve += 1
            print(f'[CYCLE-ES] No improvement on cycle {cycle+1} ({cycle_monitor}={current_metric:.9g}); '
                  f'no_improve_count={cycles_no_improve}/{cycle_patience}')

        # check patience
        if cycles_no_improve >= cycle_patience:
            print(f'[CYCLE-ES] Early stopping across cycles after {cycle+1} cycles. '
                  f'Restoring best cycle: #{best_cycle_idx+1} (metric={best_cycle_metric:.9g})')
            # restore best snapshot (prefer in-memory; fallback to saved model file)
            if best_cycle_weights is not None:
                model.set_weights(best_cycle_weights)
                print('[CYCLE-ES] Restored best weights from memory snapshot.')
            else:
                # try to load saved model from disk
                if os.path.exists(best_cycle_model_path):
                    model = keras.models.load_model(best_cycle_model_path)
                    print(f'[CYCLE-ES] Restored best model from disk: {best_cycle_model_path}')
                else:
                    print('[CYCLE-ES] WARNING: no best snapshot available to restore.')
            break

        # continue to next cycle
        # --- optionally ---
        # We can also reload the best-cycle model before next cycle to avoid drift however,
        # every cycle that builds on top of this snap-shot could end up producing the same
        # results with each iteration unless we introduce a slight change like different seeds.
        
        # model.set_weights(best_cycle_weights)  # uncomment if conservative approach desired

        # Write cycle-specific snapshot for reproducibility (already saved by ModelCheckpoint)
    else:
        # loop completed without early stopping - ensure final best is restored
        if best_cycle_weights is not None:
            print(f'[CYCLE-ES] Completed all cycles. Restoring best cycle #{best_cycle_idx+1}.')
            model.set_weights(best_cycle_weights)

    # final save
    final_path = os.path.join(log_dir, 'final_model.keras')
    try:
        model.save(final_path, overwrite=True, include_optimizer=True)
        print(f'[CYCLE-ES] Final model saved to {final_path}')
    except Exception as e:
        print(f'[CYCLE-ES] Warning: final save failed: {e}')

    return model, compressor


if __name__ == '__main__':
    trained_model, compressor = run_experiment(model, train_ds, val_ds, CONFIG, MASK_THRESH_MIN)
    print('\nFinal alive neuron counts per masked layer:')
    md_layers = get_masked_dense_layers(trained_model)
    for l in md_layers:
        print(f"{l.name}_{hex(id(l))}", int(l.mask.numpy().sum()), '/', l.units)
    print('Logs and snapshots saved to', CONFIG['log_dir'])
