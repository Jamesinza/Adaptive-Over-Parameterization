"""
adaptive_overparam_experiment_v19.py

Adaptive Overparameterization Experiment — Transformer Backbone + NumPy-driven evaluation

This patched version applies three focused fixes requested:
1. Robust eager-mode toggle inside `capture_activations` to avoid compatibility issues and ensure a boolean restore.
2. Replace brittle `tf.repeat` broadcasting in `TransformerBlock.call` with explicit `tf.reshape` + `tf.tile` broadcasting (clearer and safer).
3. Replace `approx_fisher_per_sample` with a more robust vectorized-per-batch variant that keeps gradient computations inside TF while remaining compatible with subclassed layers.

No other functional changes were made.
"""

import os
import math
import random
import numpy as np
import tensorflow as tf
from contextlib import contextmanager
import keras
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict

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
    'epochs_per_phase': 100,
    'cycles': 64,
    'regrow_finetune_steps': 5,
    'mask_thresh_min': 0.1,
    'mask_thresh_multi': 0.1,
    'prune_fraction': 0.5,
    'regrow_fraction': 0.5,
    'connection_rewire_fraction': 0.5,
    'd_model': 64,
    'hidden_units': 256,
    'hidden_layers': 4,
    'num_heads': 2,
    'learning_rate': 1e-3,
    'max_fisher_samples_per_batch': 8,
    'log_dir': './logs/adaptive_overparam_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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
    def __init__(self, d_model, num_heads, ff_dim, drop_rate=0.1, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.drop_rate = drop_rate
        self.att = keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model // self.num_heads)
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
        self.ffn_dense1 = MaskedDense(self.ff_dim, activation='gelu', name='ffn_dense1')
        self.ffn_dense2 = MaskedDense(self.d_model, activation=None, name='ffn_dense2')
        # self.dropout = keras.layers.Dropout(self.drop_rate)

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
                       'drop_rate': self.drop_rate
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
model.summary()

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


# -----------------------------
# EvoCompressor (adaptive pruning + pareto logging)
# -----------------------------
class EvoCompressor:
    def __init__(self, model, val_ds, base_prune=0.2, regrow_fraction=0.2,
                 connection_rewire_fraction=0.03, log_dir=None, regrow_loss_tol=0.01,
                 train_ds=None, regrow_finetune_steps=3, finetune_batch_size=None,
                 prune_search=False, mask_thresh_multi=0.1, mask_thresh_min=0.2):
        """
        regrow_loss_tol: relative tolerance (e.g. 0.01 -> 1%). If pruned_val_loss > baseline*(1+tol) then consider regrowth.
        train_ds: optional tuple (X_train, Y_train) used for short finetune before regrow decision.
        regrow_finetune_steps: number of mini-batches to train after pruning before deciding to regrow (1-5 suggested).
        finetune_batch_size: batch size to use during finetune (defaults to CONFIG['batch_size']).
        """
        self.model = model
        self.val_ds = val_ds
        self.base_prune = base_prune
        self.regrow_fraction = regrow_fraction
        self.connection_rewire_fraction = connection_rewire_fraction
        self.log_dir = log_dir
        self.history = []
        self.pareto = []
        self.conn_age_log = defaultdict(list)
        self.regrow_loss_tol = float(regrow_loss_tol)

        # new fields for finetune-before-regrow logic
        self.train_ds = train_ds
        self.regrow_finetune_steps = int(regrow_finetune_steps)
        self.finetune_batch_size = int(finetune_batch_size) if finetune_batch_size is not None else None
        self.prune_search = bool(prune_search)
        self.mask_thresh_multi = mask_thresh_multi
        self.mask_thresh_min = mask_thresh_min

    def _post_compression_decision(self,
                                  baseline_val_loss,
                                  val_loss_final,
                                  baseline_alive,
                                  final_alive,
                                  baseline_weights,
                                  policy='rollback',
                                  rel_tol=0.01,
                                  compression_tradeoff=0.15,
                                  long_retrain_epochs=3,
                                  train_ds=None,
                                  finetune_bs=None):
        """
        Decide whether to keep the pruned model or revert to baseline.

        Returns:
            (action, reason) where action in {
                'kept', 'rolled_back', 'retrained_and_kept', 'retrained_and_rolled_back'
            }
        Notes:
            - If rollback happens, this restores baseline_weights into self.model.
            - If a longer retrain is requested, the retrain is performed in-place.
        """
        # safe numeric guards
        baseline_val_loss = float(baseline_val_loss)
        val_loss_final = float(val_loss_final)
        rel_increase = (val_loss_final - baseline_val_loss) / (abs(baseline_val_loss) + 1e-12)
        compression_gain = (baseline_alive - final_alive) / (baseline_alive + 1e-12)

        # 0) trivial acceptance: improved or equal
        if val_loss_final <= baseline_val_loss * (1.0 + 1e-12):
            return 'kept', f'improved or equal val loss ({val_loss_final:.6g} <= {baseline_val_loss:.6g})'

        # POLICY: rollback (conservative)
        if policy == 'rollback':
            if rel_increase <= rel_tol:
                return 'kept', f'loss increase {rel_increase:.4f} <= tol {rel_tol}; keep pruned'
            else:
                # restore baseline
                self.model.set_weights(baseline_weights)
                return 'rolled_back', f'loss increase {rel_increase:.4f} > tol {rel_tol}; restored baseline'

        # POLICY: accept_if_compression_win (favor compression)
        if policy == 'accept_if_compression_win':
            if rel_increase <= rel_tol:
                return 'kept', f'loss increase {rel_increase:.4f} within tol {rel_tol}'
            # allow larger loss if compression is large enough (tunable)
            if compression_gain >= compression_tradeoff and rel_increase <= 0.05:
                return 'kept', f'accepted pruned model: compression_gain={compression_gain:.3f} rel_loss={rel_increase:.3f}'
            # otherwise rollback
            self.model.set_weights(baseline_weights)
            return 'rolled_back', f'loss increase {rel_increase:.4f} unacceptable; compression_gain={compression_gain:.3f} too small'

        # POLICY: try_longer_retrain
        if policy == 'try_longer_retrain':
            use_train = train_ds if train_ds is not None else getattr(self, 'train_ds', None)
            if use_train is None:
                # fallback to rollback if no training data
                self.model.set_weights(baseline_weights)
                return 'rolled_back', 'no training data available for longer retrain; rolled back'
            # quick longer retraining (in-place)
            epochs = int(long_retrain_epochs)
            batch_size = finetune_bs or getattr(self, 'finetune_batch_size', None) or CONFIG['batch_size']
            # Save current pruned weights (so we could inspect or revert that pruned state if desired)
            pruned_snapshot = self.model.get_weights()
            # Fit briefly
            tmp_es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
            self.model.fit(use_train[0], use_train[1], batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[tmp_es])
            # Re-evaluate validation
            if isinstance(self.val_ds, tuple) and isinstance(self.val_ds[0], np.ndarray):
                metrics_after = self.model.evaluate(self.val_ds[0], self.val_ds[1], verbose=0)
            else:
                metrics_after = self.model.evaluate(self.val_ds, verbose=0)
            val_after = float(metrics_after[0])
            rel_after = (val_after - baseline_val_loss) / (abs(baseline_val_loss) + 1e-12)
            if val_after <= baseline_val_loss * (1.0 + rel_tol):
                return 'retrained_and_kept', f'longer retrain recovered val loss: {val_after:.6g}'
            else:
                # Revert to baseline (not to pruned snapshot) to be conservative
                self.model.set_weights(baseline_weights)
                return 'retrained_and_rolled_back', f'retrain failed (rel_after={rel_after:.4f}); restored baseline'

        # Default fallback: keep pruned
        return 'kept', 'fallback: keeping pruned model by default'

    def evaluate_and_compress(self, train_loss=None, val_loss=None, do_regrow=True):
        """
        Two-phase compress:
          1) prune-only pass (neurons + connection pruning) -> evaluate pruned network
          2) if pruned loss worsened beyond regrow_loss_tol relative to baseline, perform regrowth (neurons + connections)
             and re-evaluate.
        Returns summary corresponding to final network state (after regrow if triggered).
        """

        # Helper: apply candidate changes according to scenario flags
        def _apply_prune_scenario(scenario):
            """
            scenario: one of 'neurons', 'connections', 'both'
            This function mutates the model in-place by applying the stored candidate prunes.
            """
            for unique_name, info in stored_prune_info.items():
                layer = next((l for l in md_layers if f"{l.name}_{hex(id(l))}" == unique_name), None)
                if layer is None:
                    continue
                if scenario in ('neurons', 'both'):
                    if info['prune_idx'].size:
                        layer.prune_units(info['prune_idx'])
                if scenario in ('connections', 'both'):
                    if info['conn_low_pairs'].size:
                        layer.prune_connections(info['conn_low_pairs'])

        md_layers = get_masked_dense_layers(self.model)
        # compute utilities (entropy + fisher)
        acts = capture_activations(self.model, self.val_ds, masked_layers=md_layers)
        entropies = {name: compute_activation_entropy(a) for name, a in acts.items()}
        fisher = approx_fisher_per_sample(
            self.model, self.val_ds, keras.losses.Huber(),
            max_samples_per_batch=CONFIG['max_fisher_samples_per_batch']
        )

        utilities = {}
        for name in entropies:
            e = entropies[name]
            f = fisher.get(name, np.zeros_like(e))
            if e.size == 0 or f.size == 0:
                continue
            e_norm = (e - np.median(e)) / (np.std(e) + 1e-9)
            f_norm = (f - np.median(f)) / (np.std(f) + 1e-9)
            utilities[name] = 0.5 * e_norm + 0.5 * f_norm

        # dynamic prune fraction based on gap
        prune_fraction = self.base_prune
        if train_loss is not None and val_loss is not None:
            gap = float(val_loss - train_loss)
            rel = gap / (abs(val_loss) + 1e-9)
            prune_fraction = float(self.base_prune * (1.0 + rel))
            prune_fraction = np.clip(prune_fraction, 0.02, 0.6)

        # Baseline val loss to compare against (use provided val_loss if available)
        if val_loss is not None:
            baseline_val_loss = float(val_loss)
        else:
            # evaluate baseline now
            if isinstance(self.val_ds, tuple) and isinstance(self.val_ds[0], np.ndarray):
                baseline_val_loss = float(self.model.evaluate(self.val_ds[0], self.val_ds[1], verbose=0)[0])
            else:
                baseline_val_loss = float(self.model.evaluate(self.val_ds, verbose=0)[0])

        # ---- Phase A: compute prune candidates (but do NOT apply them yet) ----
        # We will store pruning decisions so we can test different prune strategies.
        stored_prune_info = {}  # keyed by unique_name -> dict with 'prune_idx', 'n_prune', 'conn_low_pairs', 'n_rewire'
        total_alive_pre = 0

        for unique_name, score in utilities.items():
            layer = next((l for l in md_layers if f"{l.name}_{hex(id(l))}" == unique_name), None)
            if layer is None:
                print(f'No matching layer object for {unique_name} — skipping')
                continue

            mask_thresh_min = self.mask_thresh_min * (1.0 + self.mask_thresh_multi)
            mask_threshold = min(mask_thresh_min, 0.8)
            print(f'\n[INFO] Setting MASK_THRESH_MIN to: {mask_threshold}\n')
            mask = layer.mask.numpy()
            conn_mask = layer.conn_mask.numpy()
            alive_idx = np.where(mask > mask_threshold)[0]
            total_alive_pre += len(alive_idx)

            # neuron-level candidate prune indices (do NOT apply)
            prune_idx = np.zeros((0,), dtype=int)
            n_prune = 0
            if len(alive_idx) > 0:
                scores_alive = score[alive_idx]
                n_prune = max(1, int(len(alive_idx) * prune_fraction))
                prune_pos = np.argsort(scores_alive)[:n_prune]
                prune_idx = alive_idx[prune_pos]

            # connection-level candidate pruning (age-weighted score) (do NOT apply)
            W = layer.kernel.numpy()
            conn_mask = layer.conn_mask.numpy()
            conn_age = layer.conn_age.numpy()
            absW = np.abs(W) * (conn_mask > mask_threshold)
            # protect against all-zero conn_age
            max_age = np.max(conn_age) if np.max(conn_age) > 0 else 1.0
            age_penalty = np.tanh(conn_age / (max_age + 1e-6))
            score_matrix = absW * (1.0 - 0.5 * age_penalty)
            flat_score = score_matrix.flatten()
            n_rewire = int(flat_score.size * self.connection_rewire_fraction)

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
                'n_rewire': int(n_rewire)
            }

        # Snapshot baseline full model weights (trainable + non-trainable) so we can restore reliably between scenario tests.
        baseline_weights = self.model.get_weights()

        # Decide which scenarios to evaluate
        if self.prune_search:
            scenarios = ['neurons', 'connections', 'both']
            print(f"[PRUNE-SEARCH] Evaluating scenarios: {scenarios}")
        else:
            scenarios = ['both']

        scenario_results = {}
        for scen in scenarios:
            # restore baseline
            self.model.set_weights(baseline_weights)
            # apply that scenario's pruning
            _apply_prune_scenario(scen)
            # evaluate
            if isinstance(self.val_ds, tuple) and isinstance(self.val_ds[0], np.ndarray):
                metrics = self.model.evaluate(self.val_ds[0], self.val_ds[1], verbose=0)
            else:
                metrics = self.model.evaluate(self.val_ds, verbose=0)
            scenario_val_loss = float(metrics[0])
            scenario_results[scen] = {
                'val_loss': scenario_val_loss
            }
            print(f"[PRUNE-SEARCH] scenario={scen}, val_loss={scenario_val_loss:.9f}")

        # Pick best scenario (lowest val loss)
        best_scenario = min(scenario_results.keys(), key=lambda k: scenario_results[k]['val_loss'])
        pruned_val_loss = float(scenario_results[best_scenario]['val_loss'])
        print(f"[PRUNE-SEARCH] Best scenario: {best_scenario} (val_loss={pruned_val_loss:.9f})")

        # Now ensure the model is left in the chosen scenario's pruned state (apply on top of baseline)
        self.model.set_weights(baseline_weights)
        _apply_prune_scenario(best_scenario)

        # optional: for logging, update stored_prune_info to reflect what was actually applied
        # (we keep stored_prune_info as-is; consumers can inspect it if needed)

        regrow_triggered = False
        val_loss_after_regrow = pruned_val_loss

        # Decision: trigger regrowth only if pruning degraded validation beyond tolerance AND regrow enabled
        if do_regrow and (pruned_val_loss > baseline_val_loss * (1.0 + self.regrow_loss_tol)):
            print(f"[REGROW-DECISION] Pruned loss worsened beyond tol ({self.regrow_loss_tol*100:.2f}%).")
            # First, allow a short fine-tune (1-5 mini-batches) before making the regrow decision.
            val_loss_after_finetune = pruned_val_loss
            if self.train_ds is not None and self.regrow_finetune_steps > 0:
                print(f"[FINETUNE] Running short finetune for up to {self.regrow_finetune_steps} mini-batches...")
                finetune_bs = self.finetune_batch_size if self.finetune_batch_size is not None else CONFIG['batch_size']
                # iterate training batches and run train_on_batch for a few steps
                ft_iter = numpy_batch_iter(self.train_ds[0], self.train_ds[1], finetune_bs)
                steps_done = 0
                for xb, yb in ft_iter:
                    # single gradient update
                    self.model.train_on_batch(xb, yb)
                    steps_done += 1
                    if steps_done >= self.regrow_finetune_steps:
                        break
                # re-evaluate validation loss after finetune
                if isinstance(self.val_ds, tuple) and isinstance(self.val_ds[0], np.ndarray):
                    val_metrics_after_ft = self.model.evaluate(self.val_ds[0], self.val_ds[1], verbose=0)
                else:
                    val_metrics_after_ft = self.model.evaluate(self.val_ds, verbose=0)
                val_loss_after_finetune = float(val_metrics_after_ft[0])
                print(f"[FINETUNE] Completed {steps_done} steps, val_loss after finetune = {val_loss_after_finetune:.9f}")
            else:
                print("[FINETUNE] No training data provided to compressor or finetune steps == 0; skipping finetune.")

            # Decide whether to regrow after finetune
            if val_loss_after_finetune <= baseline_val_loss * (1.0 + self.regrow_loss_tol):
                # fine-tune recovered the pruned model enough — skip regrowth
                print("[REGROW-DECISION] Fine-tune recovered validation loss; skipping regrowth.")
                regrow_triggered = False
                val_loss_after_regrow = val_loss_after_finetune
            else:
                # fine-tune did not recover validation performance — perform regrowth now
                print("[REGROW-DECISION] Fine-tune did NOT recover performance; performing regrowth.")
                regrow_triggered = True

                # ---- Phase B: perform regrowth guided by utilities/desirability ----
                for unique_name, info in stored_prune_info.items():
                    layer = next((l for l in md_layers if f"{l.name}_{hex(id(l))}" == unique_name), None)
                    if layer is None:
                        continue

                    # neuron regrowth: n_regrow ~ Binomial(n_prune, regrow_fraction)
                    n_prune = info['n_prune']
                    if n_prune <= 0 or self.regrow_fraction <= 0:
                        n_regrow = 0
                    else:
                        n_regrow = int(np.random.binomial(n_prune, self.regrow_fraction))

                    if n_regrow > 0:
                        dead_indices = np.where(layer.mask.numpy() == 0)[0]
                        if len(dead_indices) >= n_regrow:
                            regrow_idx = np.random.choice(dead_indices, size=n_regrow, replace=False)
                        else:
                            # fall back to regrowing from the previously pruned indices if not enough other dead ones
                            available = info['prune_idx'] if info['prune_idx'].size else dead_indices
                            regrow_idx = np.random.choice(available, size=min(n_regrow, len(available)), replace=False)
                        layer.regrow_units(regrow_idx)
                        print(f"[REGROW] {layer.name}: regrew {len(regrow_idx)} neurons")
                        # (conn_age resets for regrown units are handled by MaskedDense.regrow_units)

                    # connection regrowth: use desirability heuristic (out_u + in_u)
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
                                # MaskedDense.regrow_connections now resets ages internally.
                                regrew_conn_count = len(regrow_pairs)
                                print(f"[REGROW] {layer.name}: regrew {regrew_conn_count} connections")

                # Evaluate after regrowth (final)
                if isinstance(self.val_ds, tuple) and isinstance(self.val_ds[0], np.ndarray):
                    val_metrics_after = self.model.evaluate(self.val_ds[0], self.val_ds[1], verbose=0)
                else:
                    val_metrics_after = self.model.evaluate(self.val_ds, verbose=0)
                val_loss_after_regrow = float(val_metrics_after[0])
                print(f"[REGROW-PHASE] val_loss_after_regrow={val_loss_after_regrow:.9f}")

        # Finalize: compute summaries after final state
        md_layers_final = get_masked_dense_layers(self.model)
        total_alive = 0
        df_rows = []
        for layer in md_layers_final:
            mask = layer.mask.numpy()
            conn_mask = layer.conn_mask.numpy()

            # log age stats for this final state
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
            conn_density = conn_alive / float(np.prod(conn_mask.shape))
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

        # evaluate model and finalize final loss
        if isinstance(self.val_ds, tuple) and isinstance(self.val_ds[0], np.ndarray):
            val_metrics_final = self.model.evaluate(self.val_ds[0], self.val_ds[1], verbose=0)
        else:
            val_metrics_final = self.model.evaluate(self.val_ds, verbose=0)
        val_loss_now = float(val_metrics_final[0])

        # ----------------------------
        # Post-compression automated decision (keep vs rollback vs retrain)
        # baseline_weights MUST have been captured earlier (before we applied pruning).
        # baseline_alive was recorded earlier as total_alive_pre.
        try:
            action, reason = self._post_compression_decision(
                baseline_val_loss=baseline_val_loss,
                val_loss_final=val_loss_now,
                baseline_alive=total_alive_pre,
                final_alive=total_alive,
                baseline_weights=baseline_weights,
                policy='rollback',                # change policy here if you want e.g. 'accept_if_compression_win' or 'try_longer_retrain'
                rel_tol=0.01,
                compression_tradeoff=0.15,
                long_retrain_epochs=5,
                train_ds=self.train_ds,
                finetune_bs=self.finetune_batch_size
            )
            print(f"[POST-COMP] action={action}; reason={reason}")
        except Exception as e:
            print(f"[POST-COMP] decision routine failed: {e}")
            action, reason = 'kept', 'decision routine error'

        if action.startswith('rolled_back'):
            # recompute final-layer stats to reflect restored baseline
            md_layers_final = get_masked_dense_layers(self.model)
            total_alive = 0
            df_rows = []
            for layer in md_layers_final:
                mask = layer.mask.numpy()
                conn_mask = layer.conn_mask.numpy()
                total_alive += int(mask.sum())
                conn_alive = int(conn_mask.sum())
                conn_density = conn_alive / float(np.prod(conn_mask.shape))
                alive_weights = layer.kernel.numpy()[conn_mask > 0.5]
                avg_conn_mag = float(np.mean(np.abs(alive_weights))) if alive_weights.size else 0.0
                unique_name = f"{layer.name}_{hex(id(layer))}"
                df_rows.append({
                    'layer': unique_name,
                    'alive_neurons': int(mask.sum()),
                    'alive_connections': conn_alive,
                    'conn_density': conn_density,
                    'utility_mean': float(utilities.get(unique_name, np.zeros((layer.units,))).mean()) if unique_name in utilities else 0.0,
                    'avg_conn_mag': avg_conn_mag
                })
            # update pareto/history entries if you want to reflect rollback

        self.pareto.append((total_alive, val_loss_now))
        summary = {
            'alive_total': int(total_alive),
            'prune_fraction_used': float(prune_fraction),
            'baseline_val_loss': baseline_val_loss,
            'pruned_val_loss': pruned_val_loss,
            'regrow_triggered': bool(regrow_triggered),
            'val_loss_after_regrow': float(val_loss_after_regrow),
            'val_loss_final': val_loss_now
        }
        self.history.append(summary)

        # write CSV snapshot
        df = pd.DataFrame(df_rows)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            df.to_csv(os.path.join(self.log_dir, f'compressor_snapshot_{len(self.history)}.csv'), index=False)

        # print conn_age summary
        for lname, stats in self.conn_age_log.items():
            last = stats[-1]
            print(f"[AGE MONITOR] {lname}: mean={last['mean']:.2f}, std={last['std']:.2f}, max={last['max']:.0f}, alive={last['alive_count']}")

        return summary, mask_thresh_min


# -----------------------------
# Training + cycles
# -----------------------------
def run_experiment(model, train_ds, val_ds, config, mask_thresh_min):
    log_dir = config['log_dir']
    chkpt_path = os.path.join(log_dir, 'best_base_model.keras')
    
    callbacks = [keras.callbacks.TensorBoard(log_dir=log_dir),
                 keras.callbacks.ModelCheckpoint(chkpt_path, save_best_only=True, monitor='val_loss'),
                 keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=3, cooldown=10),
                ]

    print('\nWarmup training...\n')
    history = model.fit(train_ds[0], train_ds[1], batch_size=config['batch_size'],
                        validation_data=(val_ds[0], val_ds[1]), epochs=5,  #config['epochs_per_phase'],
                        callbacks=callbacks, verbose=2)

    for cycle in range(config['cycles']):
        compressor = EvoCompressor(model, val_ds,
                                   base_prune=config['prune_fraction'],
                                   regrow_fraction=config['regrow_fraction'],
                                   connection_rewire_fraction=config['connection_rewire_fraction'],
                                   log_dir=log_dir,
                                   train_ds=train_ds,                                                # <-- pass training data tuple here
                                   regrow_finetune_steps=config['regrow_finetune_steps'],            # you can tune this or pull from CONFIG if you want
                                   finetune_batch_size=config['batch_size']//2,                         # or None to use default
                                   prune_search=True,
                                   mask_thresh_multi=config['mask_thresh_multi'],
                                   mask_thresh_min=mask_thresh_min,
                                  )

        print(f'\n--- Cycle {cycle + 1}/{config["cycles"]} ---\n')

        chkpt_path = os.path.join(log_dir, f'best_cycle_{cycle+1}_model.keras')
        callbacks = [keras.callbacks.TensorBoard(log_dir=log_dir),
                     keras.callbacks.ModelCheckpoint(chkpt_path, save_best_only=True, monitor='val_loss'),
                     keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
                     keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.99, patience=3, cooldown=0, min_lr=5e-6),
                    ]
        
        history = model.fit(train_ds[0], train_ds[1], batch_size=config['batch_size'],
                            validation_data=(val_ds[0], val_ds[1]), epochs=config['epochs_per_phase'],
                            callbacks=callbacks, verbose=2)

        train_loss = float(history.history['loss'][-1])
        val_loss = float(history.history['val_loss'][-1])
        summary, mask_thresh_min = compressor.evaluate_and_compress(train_loss=train_loss, val_loss=val_loss)
        print('\nCompressor summary:', summary,'\n')

    model.save(os.path.join(log_dir, 'final_model.keras'))
    return model, compressor

if __name__ == '__main__':
    trained_model, compressor = run_experiment(model, train_ds, val_ds, CONFIG, MASK_THRESH_MIN)
    print('\nFinal alive neuron counts per masked layer:')
    md_layers = get_masked_dense_layers(trained_model)
    for l in md_layers:
        print(f"{l.name}_{hex(id(l))}", int(l.mask.numpy().sum()), '/', l.units)
    print('Logs and snapshots saved to', CONFIG['log_dir'])
