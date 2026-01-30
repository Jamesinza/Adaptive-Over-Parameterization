"""
Adaptive Overparameterization Experiment — Transformer Backbone + NumPy-driven evaluation

This version applies the following changes:
- Replaces tf.data pipelines with plain NumPy arrays for training/validation to avoid iterator exhaustion.
- capture_activations and approx_fisher_per_sample accept either a tf.data.Dataset or a (X_np, Y_np) tuple.
- Robust, unique-keyed monkeypatch capture for MaskedDense (context manager).
- Utilities preserved: get_masked_dense_layers, compute_activation_entropy, visualization, EvoCompressor.

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
        Y[i] = data.flatten()[i + seq_len]
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
df = raw_df.tail(16_384).copy()
df = df[df['High'] != df['Low']]
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)
data = df[['Close']].values

split = 1024
train_data = data[:-split]
val_data = data[-split:]

batch_size = compute_batch_size(len(train_data))

CONFIG = {
    'seq_len': 8,
    'train_size': len(train_data),
    'val_size': len(val_data),
    'batch_size': batch_size,
    'epochs_per_phase': 10,
    'cycles': 12,
    'prune_fraction': 0.3,
    'regrow_fraction': 0.3,
    'connection_rewire_fraction': 0.1,
    'd_model': 32,
    'hidden_units': 128,
    'hidden_layers': 2,
    'num_heads': 2,
    'learning_rate': 1e-3,
    'max_fisher_samples_per_batch': 8,
    'log_dir': './logs/adaptive_overparam_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
}

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
        m = self.mask.numpy()
        m[indices] = 1.0
        self.mask.assign(m)
        k = self.kernel.numpy()
        b = self.bias.numpy()
        for idx in indices:
            k[:, idx] = np.random.normal(scale=0.02, size=(k.shape[0],))
            b[idx] = 0.0
        self.kernel.assign(k)
        self.bias.assign(b)

    # connection-level operations
    def prune_connections(self, idx_pairs):
        if len(idx_pairs) == 0:
            return
        m = self.conn_mask.numpy()
        for (i, j) in idx_pairs:
            m[i, j] = 0.0
        self.conn_mask.assign(m)

    def regrow_connections(self, idx_pairs):
        if len(idx_pairs) == 0:
            return
        m = self.conn_mask.numpy()
        k = self.kernel.numpy()
        for (i, j) in idx_pairs:
            m[i, j] = 1.0
            k[i, j] = np.random.normal(scale=0.02)
        self.conn_mask.assign(m)
        self.kernel.assign(k)

    def get_config(self):
        return {'units': self.units, 'activation': keras.activations.serialize(self.activation)}


# -----------------------------
# Router (Concrete)
# -----------------------------
@keras.utils.register_keras_serializable(package="Custom")
class Router(keras.layers.Layer):
    def __init__(self, units, temp=0.5, name=None):
        super().__init__(name=name)
        self.units = int(units)
        self.temp = tf.Variable(initial_value=float(temp), trainable=False, dtype=tf.float32)

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        self.proj = self.add_weight(name='proj', shape=(last_dim, self.units), initializer=tf.random_normal_initializer(stddev=0.02), trainable=True)
        self.logit_bias = self.add_weight(name='logit_bias', shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs, training=True):
        logits = tf.matmul(inputs, self.proj) + self.logit_bias
        if training:
            uniform = tf.random.uniform(tf.shape(logits), minval=1e-6, maxval=1.0 - 1e-6)
            g = tf.math.log(uniform) - tf.math.log(1.0 - uniform)
            noisy = (logits + g) / (self.temp + 1e-8)
            gate = tf.sigmoid(noisy)
        else:
            gate = tf.sigmoid(logits / (self.temp + 1e-8))
        return gate

    def set_temp(self, t):
        self.temp.assign(float(t))

    def get_config(self):
         return {'units': int(self.units), 'temp': float(self.temp.numpy() if hasattr(self.temp, 'numpy') else self.temp)}

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
        self.dropout = keras.layers.Dropout(self.drop_rate)
        self.router = Router(units=self.ff_dim, temp=0.5, name='router')

    def call(self, x, training=False):
        attn_out = self.att(x, x, x)
        x = self.norm1(x + attn_out)
        pooled = tf.reduce_mean(x, axis=1)  # (batch, d_model)
        gates = self.router(pooled, training=training)  # (batch, ff_dim)
        seq_len = tf.shape(x)[1]
        x_flat = tf.reshape(x, (-1, tf.shape(x)[-1]))
        pre, h = self.ffn_dense1(x_flat, return_pre_activation=True)
        gates_exp = tf.repeat(gates, repeats=seq_len, axis=0)
        h = h * gates_exp
        h2 = self.ffn_dense2(h)
        # h2 = tf.reshape(h2, (-1, seq_len, tf.shape(x)[2]))
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
def capture_maskeddense_outputs(masked_layers):
    """
    Monkeypatch MaskedDense.call on the provided instances and record outputs.
    Records are keyed by unique_key = f"{layer.name}_{hex(id(layer))}".
    The wrapper calls the original bound method exactly and returns its result unchanged.
    Caller should enable eager mode (tf.config.run_functions_eagerly(True)) when needed.
    """
    records = {}
    originals = {}

    for layer in masked_layers:
        key = f"{layer.name}_{hex(id(layer))}"
        records[key] = []
        originals[key] = layer.call  # bound method

        def make_wrapper(orig_method, layer, key):
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
                # Try converting to numpy if eager; otherwise store tensor object
                try:
                    records[key].append(out_tensor.numpy())
                except Exception:
                    records[key].append(out_tensor)
                return res
            return wrapper

        layer.call = make_wrapper(originals[key], layer, key)

    try:
        yield records
    finally:
        # restore
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
            prev = tf.config.functions_run_eagerly()
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
                tf.config.run_functions_eagerly(prev)

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
        if vmax - vmin < 1e-6:
            ent[u] = 0.0
            continue
        bins = np.histogram_bin_edges(vals, bins='auto')
        hist, _ = np.histogram(vals, bins=bins)
        p = hist / (hist.sum() + eps)
        p = p[p > 0]
        ent[u] = -np.sum(p * np.log(p))
    return ent

# -----------------------------
# Improved Fisher: per-sample gradients wrt layer outputs (NumPy-aware)
# -----------------------------
def approx_fisher_per_sample(model, dataset, loss_fn, max_samples_per_batch=8, batch_size=None):
    md_layers = get_masked_dense_layers(model)
    if batch_size is None:
        batch_size = CONFIG['batch_size']

    submodels = {}
    for l in md_layers:
        key = f"{l.name}_{hex(id(l))}"
        try:
            submodels[key] = keras.Model(model.input, l.output)
        except Exception:
            submodels[key] = None

    fisher = {f"{l.name}_{hex(id(l))}": np.zeros((l.units,), dtype=np.float32) for l in md_layers}
    count = 0

    # choose iterator
    if isinstance(dataset, tuple) and isinstance(dataset[0], np.ndarray):
        batch_iter = numpy_batch_iter(dataset[0], dataset[1], batch_size)
    else:
        batch_iter = iter(dataset)

    for x_batch, y_batch in batch_iter:
        x_batch_tf = tf.convert_to_tensor(x_batch)
        y_batch_tf = tf.convert_to_tensor(y_batch)
        batch_size_real = x_batch.shape[0]
        n_samples = int(min(batch_size_real, max_samples_per_batch))
        for i in range(n_samples):
            x = x_batch_tf[i:i + 1]
            y = y_batch_tf[i:i + 1]
            with tf.GradientTape() as tape:
                preds = model(x, training=False)
                outs = {}
                for l in md_layers:
                    key = f"{l.name}_{hex(id(l))}"
                    subm = submodels.get(key, None)
                    if subm is None:
                        continue
                    out = subm(x)
                    tape.watch(out)
                    outs[key] = out
                loss = loss_fn(y, preds)
            for key, out in outs.items():
                grad = tape.gradient(loss, out)
                if grad is None:
                    continue
                grad_np = grad.numpy()
                if grad_np.ndim == 3:
                    unit_grad = np.sqrt(np.sum(grad_np[0] ** 2, axis=0))
                elif grad_np.ndim == 2:
                    unit_grad = np.abs(grad_np[0])
                else:
                    unit_grad = np.abs(grad_np.ravel())
                fisher[key] += unit_grad ** 2
            count += 1

    if count == 0:
        return fisher
    for name in fisher:
        fisher[name] /= (count + 1e-9)
    return fisher

# -----------------------------
# Visualization helpers
# -----------------------------
def visualize_specialists(model, dataset, log_dir, max_points=2000):
    md_layers = get_masked_dense_layers(model)
    acts = capture_activations(model, dataset, masked_layers=md_layers)
    for name, a in acts.items():
        if a.size == 0:
            continue
        if a.shape[0] > max_points:
            idx = np.random.choice(a.shape[0], max_points, replace=False)
            a_sub = a[idx]
        else:
            a_sub = a
        try:
            pca = PCA(n_components=2)
            comp = pca.fit_transform(a_sub.T)
        except Exception:
            continue
        plt.figure(figsize=(6, 6))
        plt.scatter(comp[:, 0], comp[:, 1], s=6)
        plt.title(f'PCA neuron embedding - {name}')
        plt.xlabel('PC1'); plt.ylabel('PC2')
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, f'pca_neurons_{name}.png'))
        plt.close()

# -----------------------------
# EvoCompressor (adaptive pruning + pareto logging)
# -----------------------------
class EvoCompressor:
    def __init__(self, model, val_ds, base_prune=0.2, regrow_fraction=0.2,
                 connection_rewire_fraction=0.03, log_dir=None, regrow_loss_tol=0.01):
        """
        regrow_loss_tol: relative tolerance (e.g. 0.01 -> 1%). If pruned_val_loss > baseline*(1+tol) then trigger regrowth.
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

    def evaluate_and_compress(self, train_loss=None, val_loss=None, do_regrow=True):
        """
        Two-phase compress:
          1) prune-only pass (neurons + connection pruning) -> evaluate pruned network
          2) if pruned loss worsened beyond regrow_loss_tol relative to baseline, perform regrowth (neurons + connections)
             and re-evaluate.
        Returns summary corresponding to final network state (after regrow if triggered).
        """
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

        # ---- Phase A: prune-only pass ----
        # We will store pruning decisions so we can regrow later if necessary.
        stored_prune_info = {}  # keyed by unique_name -> dict with 'prune_idx', 'n_prune', 'conn_low_pairs', 'n_rewire'
        total_alive_pre = 0

        # First, increment age for current alive connections (before pruning)
        for layer in md_layers:
            conn_mask = layer.conn_mask.numpy()
            conn_age = layer.conn_age.numpy()
            conn_age[conn_mask > 0.5] += 1.0
            layer.conn_age.assign(conn_age)

        # Now perform prune-only operations across layers
        for unique_name, score in utilities.items():
            layer = next((l for l in md_layers if f"{l.name}_{hex(id(l))}" == unique_name), None)
            if layer is None:
                print(f'No matching layer object for {unique_name} — skipping')
                continue

            mask = layer.mask.numpy()
            conn_mask = layer.conn_mask.numpy()
            print(f'\n[PRUNE-PHASE] Layer {unique_name}: alive neurons={int(mask.sum())}, alive_conns={int(conn_mask.sum())}')

            alive_idx = np.where(mask > 0.5)[0]
            total_alive_pre += len(alive_idx)

            # ---- neuron-level prune/regrow ----
            prune_idx = np.zeros((0,), dtype=int)
            n_prune = 0
            if len(alive_idx) > 0:
                scores_alive = score[alive_idx]
                n_prune = max(1, int(len(alive_idx) * prune_fraction))
                prune_pos = np.argsort(scores_alive)[:n_prune]
                prune_idx = alive_idx[prune_pos]
                layer.prune_units(prune_idx)
                print(f"[PRUNE-PHASE] {layer.name}: pruned {len(prune_idx)} neurons")

            #  ---- connection-level pruning (age-weighted score) ----
            W = layer.kernel.numpy()
            conn_mask = layer.conn_mask.numpy()
            conn_age = layer.conn_age.numpy()
            absW = np.abs(W) * (conn_mask > 0.5)
            age_penalty = np.tanh(conn_age / (np.max(conn_age) + 1e-6))
            score_matrix = absW * (1.0 - 0.5 * age_penalty)
            flat_score = score_matrix.flatten()
            n_rewire = int(flat_score.size * self.connection_rewire_fraction)

            low_pairs = np.zeros((0, 2), dtype=int)
            if n_rewire > 0:
                # prune the weakest (by score) alive connections
                nonzero_count = np.count_nonzero(flat_score)
                k_prune = min(n_rewire, nonzero_count)
                if k_prune > 0:
                    low_idx = np.argpartition(flat_score, k_prune)[:k_prune]
                    low_pairs = np.array(np.unravel_index(low_idx, conn_mask.shape)).T
                    layer.prune_connections(low_pairs)
                    print(f"[PRUNE-PHASE] {layer.name}: pruned {len(low_pairs)} connections (weak+old)")

            stored_prune_info[unique_name] = {
                'prune_idx': prune_idx,
                'n_prune': int(n_prune),
                'conn_low_pairs': low_pairs,
                'n_rewire': int(n_rewire)
            }

        # Evaluate pruned network
        if isinstance(self.val_ds, tuple) and isinstance(self.val_ds[0], np.ndarray):
            val_metrics_pruned = self.model.evaluate(self.val_ds[0], self.val_ds[1], verbose=0)
        else:
            val_metrics_pruned = self.model.evaluate(self.val_ds, verbose=0)
        pruned_val_loss = float(val_metrics_pruned[0])
        print(f"\n[PRUNE-PHASE] baseline_val_loss={baseline_val_loss:.9f}, pruned_val_loss={pruned_val_loss:.9f}")

        regrow_triggered = False
        val_loss_after_regrow = pruned_val_loss

        # Decision: trigger regrowth only if pruning degraded validation beyond tolerance AND regrow enabled
        if do_regrow and (pruned_val_loss > baseline_val_loss * (1.0 + self.regrow_loss_tol)):
            print(f"[REGROW-DECISION] Pruned loss worsened beyond tol ({self.regrow_loss_tol*100:.2f}%), triggering regrowth.")
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
                            # reset ages for regrown connections
                            conn_age = layer.conn_age.numpy()
                            for (i, j) in regrow_pairs:
                                conn_age[i, j] = 0.0
                            layer.conn_age.assign(conn_age)
                            regrew_conn_count = len(regrow_pairs)
                            print(f"[REGROW] {layer.name}: regrew {regrew_conn_count} connections")

            # Evaluate after regrowth
            if isinstance(self.val_ds, tuple) and isinstance(self.val_ds[0], np.ndarray):
                val_metrics_after = self.model.evaluate(self.val_ds[0], self.val_ds[1], verbose=0)
            else:
                val_metrics_after = self.model.evaluate(self.val_ds, verbose=0)
            val_loss_after_regrow = float(val_metrics_after[0])
            print(f"[REGROW-PHASE] val_loss_after_regrow={val_loss_after_regrow:.6f}")

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

        return summary

    def plot_conn_age_history(self):
        if not self.conn_age_log:
            print("No connection age history to plot.")
            return
        os.makedirs(self.log_dir, exist_ok=True)
        for lname, logs in self.conn_age_log.items():
            steps = np.arange(len(logs))
            means = [x['mean'] for x in logs]
            stds = [x['std'] for x in logs]
            plt.figure(figsize=(7, 4))
            plt.fill_between(steps, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.3)
            plt.plot(steps, means, label=f'{lname} mean±std')
            plt.xlabel('Compression cycle')
            plt.ylabel('Connection age')
            plt.title(f'Connection age evolution - {lname}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f'conn_age_{lname}.png'))
            plt.close()

    def visualize_conn_age_heatmaps(self):
        md_layers = get_masked_dense_layers(self.model)
        os.makedirs(self.log_dir, exist_ok=True)
        for layer in md_layers:
            conn_age = layer.conn_age.numpy()
            conn_mask = layer.conn_mask.numpy()
            conn_age_masked = conn_age * conn_mask
            plt.figure(figsize=(6, 5))
            plt.imshow(conn_age_masked, aspect='auto', cmap='viridis')
            plt.colorbar(label='Connection Age')
            plt.title(f'{layer.name} Connection Ages')
            plt.xlabel('Output Neuron')
            plt.ylabel('Input Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f'conn_age_heatmap_{layer.name}.png'))
            plt.close()


# -----------------------------
# Training + cycles
# -----------------------------
def run_experiment(model, train_ds, val_ds, config):
    log_dir = config['log_dir']
    chkpt_path = os.path.join(log_dir, 'best_base_model.keras')
    
    callbacks = [keras.callbacks.TensorBoard(log_dir=log_dir),
                 keras.callbacks.ModelCheckpoint(chkpt_path, save_best_only=True, monitor='val_loss'),
                 keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, cooldown=10),
                ]

    print('\nWarmup training...\n')
    history = model.fit(train_ds[0], train_ds[1], batch_size=config['batch_size'],
                        validation_data=(val_ds[0], val_ds[1]), epochs=config['epochs_per_phase'],
                        callbacks=callbacks, verbose=2)

    for cycle in range(config['cycles']):
        compressor = EvoCompressor(model, val_ds, base_prune=config['prune_fraction'],
                                   regrow_fraction=config['regrow_fraction'],
                                   connection_rewire_fraction=config['connection_rewire_fraction'],
                                   log_dir=log_dir
                                  )
        print(f'\n--- Cycle {cycle + 1}/{config["cycles"]} ---\n')

        chkpt_path = os.path.join(log_dir, f'best_cycle_{cycle+1}_model.keras')
        callbacks = [keras.callbacks.TensorBoard(log_dir=log_dir),
                     keras.callbacks.ModelCheckpoint(chkpt_path, save_best_only=True, monitor='val_loss'),
                     keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                     keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, cooldown=2),
                    ]
        
        history = model.fit(train_ds[0], train_ds[1], batch_size=config['batch_size'],
                            validation_data=(val_ds[0], val_ds[1]), epochs=config['epochs_per_phase'],
                            callbacks=callbacks, verbose=2)

        train_loss = float(history.history['loss'][-1])
        val_loss = float(history.history['val_loss'][-1])
        summary = compressor.evaluate_and_compress(train_loss=train_loss, val_loss=val_loss)
        print('\nCompressor summary:', summary,'\n')
        visualize_specialists(model, val_ds, log_dir)        

    model.save(os.path.join(log_dir, 'final_model.keras'))
    return model, compressor

if __name__ == '__main__':
    trained_model, compressor = run_experiment(model, train_ds, val_ds, CONFIG)
    print('\nFinal alive neuron counts per masked layer:')
    md_layers = get_masked_dense_layers(trained_model)
    for l in md_layers:
        print(f"{l.name}_{hex(id(l))}", int(l.mask.numpy().sum()), '/', l.units)
    print('Logs and snapshots saved to', CONFIG['log_dir'])
