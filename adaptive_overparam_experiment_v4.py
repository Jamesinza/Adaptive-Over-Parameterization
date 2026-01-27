"""
Adaptive Overparameterization Experiment — Transformer Backbone + Enhanced Tools (FIXED)

Fixes applied in this version:
- Router: removed unused variable that caused "Gradients do not exist for variables [...]" warnings; logits now include a bias term so it's part of the graph.
- MaskedDense discovery: previously we only scanned model.layers (top-level). Transformer blocks kept MaskedDense as nested submodules; that caused no masked layers to be found and resulted in zero alive counts. Now we search model.submodules recursively.
- capture_activations and approx_fisher_per_sample now build submodels from the actual MaskedDense layer outputs (using layer.output) and operate on the discovered masked layers list.
- Minor safety: reduced default aggressive pruning and added guards.

Notes:
- You'll still see occasional 'Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence' logs when iterating datasets many times — that's informational, not fatal. If it bothers you, use dataset.repeat() when streaming.
- Per-sample Fisher remains expensive; keep CONFIG['max_fisher_samples_per_batch'] small for local runs.
"""

import os
import math
import random
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------------
# Config
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Dataset
# -----------------------------

def make_dataset(data ,n, seq_len):
    X = np.empty([n-seq_len, seq_len, 1], dtype=np.float32)
    Y = np.empty([n-seq_len], dtype=np.float32)
    for i in range(n-seq_len):
        X[i] = data[i:i+seq_len]
        Y[i] = data.flatten()[i+seq_len]
    return X, Y

def compute_batch_size(dataset_length: int) -> int:
    base_unit = 24_576
    base_batch = 32
    return int(base_batch * math.ceil(max(1, dataset_length) / base_unit))    

raw_df = pd.read_csv('datasets/EURUSD_M1_245.csv')
df = raw_df.tail(10_000).copy()
df = df[df['High'] != df['Low']]
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)
data = df[['Close']].values

split      = 1000
train_data = data[:-split]
val_data   = data[-split:]
batch_size = compute_batch_size(len(train_data))

CONFIG = {
    'seq_len': 8,
    'train_size': len(train_data),
    'val_size': len(val_data),
    'batch_size': batch_size,
    'epochs_per_phase': 5,
    'cycles': 2,
    'prune_fraction': 0.50,
    'regrow_fraction': 0.50,
    'hidden_units': 128,
    'hidden_layers': 2,
    'num_heads': 2,
    'ffn_mult': 4,
    'learning_rate': 1e-3,
    'max_fisher_samples_per_batch': 4,
    'log_dir': './logs/adaptive_overparam_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
}

os.makedirs(CONFIG['log_dir'], exist_ok=True)

X_train, Y_train = make_dataset(train_data, CONFIG['train_size'], CONFIG['seq_len'])
X_val, Y_val = make_dataset(val_data, CONFIG['val_size'], CONFIG['seq_len'])

train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# Positional Encoding
# -----------------------------

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(pe, dtype=tf.float32)

# -----------------------------
# Masked Dense (serializable)
# -----------------------------
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
        self.mask = self.add_weight(
            name='mask',
            shape=(self.units,),
            initializer=keras.initializers.Constant(1.0),
            trainable=False
        )

    def call(self, inputs, return_pre_activation=False):
        masked_kernel = self.kernel * tf.reshape(self.mask, (1, -1))
        masked_bias = self.bias * self.mask
        pre = tf.matmul(inputs, masked_kernel) + masked_bias
        if self.activation is not None:
            out = self.activation(pre)
        else:
            out = pre
        if return_pre_activation:
            return pre, out
        return out

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

    def get_config(self):
        return {'units': self.units, 'activation': keras.activations.serialize(self.activation)}

# -----------------------------
# Router (Concrete) - FIXED: no unused variables
# -----------------------------
class Router(keras.layers.Layer):
    def __init__(self, units, temp=0.5, name=None):
        super().__init__(name=name)
        self.units = int(units)
        self.temp = tf.Variable(initial_value=float(temp), trainable=False, dtype=tf.float32)

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        # projection that produces logits from pooled context
        self.proj = self.add_weight(name='proj', shape=(last_dim, self.units), initializer=tf.random_normal_initializer(stddev=0.02), trainable=True)
        # small bias for logits so it's part of the trainable graph and receives gradients
        self.logit_bias = self.add_weight(name='logit_bias', shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs, training=True):
        # inputs: (batch, dim) pooled context
        logits = tf.matmul(inputs, self.proj) + self.logit_bias  # (batch, units)
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

# -----------------------------
# Transformer block
# -----------------------------
class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, name=None):
        super().__init__(name=name)
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
        self.ffn_dense1 = MaskedDense(ff_dim, activation='gelu', name='ffn_dense1')
        self.ffn_dense2 = MaskedDense(d_model, activation=None, name='ffn_dense2')
        self.dropout = keras.layers.Dropout(0.1)
        self.router = Router(units=ff_dim, temp=0.5, name='router')

    def call(self, x, training=False):
        attn_out = self.att(x, x, x)
        x = self.norm1(x + attn_out)
        pooled = tf.reduce_mean(x, axis=1)  # (batch, d_model)
        gates = self.router(pooled, training=training)  # (batch, ff_dim)
        # process FFN per time-step
        seq_len = tf.shape(x)[1]
        x_flat = tf.reshape(x, (-1, tf.shape(x)[-1]))
        pre, h = self.ffn_dense1(x_flat, return_pre_activation=True)
        gates_exp = tf.repeat(gates, repeats=seq_len, axis=0)
        h = h * gates_exp
        h2 = self.ffn_dense2(h)
        h2 = tf.reshape(h2, (-1, seq_len, tf.shape(x)[2]))
        x = self.norm2(x + h2)
        return x

# -----------------------------
# Model builder
# -----------------------------

def build_transformer_model(seq_len, d_model, num_layers, num_heads, ff_dim):
    inp = keras.layers.Input(shape=(seq_len, 1))
    x = keras.layers.TimeDistributed(keras.layers.Dense(d_model))(inp)
    pe = positional_encoding(seq_len, d_model)
    x = x + pe
    for i in range(num_layers):
        x = TransformerBlock(d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, name=f'transformer_block_{i}')(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    out = keras.layers.Dense(1, name='head')(x)
    model = keras.Model(inp, out)
    return model

D_MODEL = 64
model = build_transformer_model(CONFIG['seq_len'], d_model=D_MODEL, num_layers=CONFIG['hidden_layers'], num_heads=CONFIG['num_heads'], ff_dim=CONFIG['hidden_units'])
optim = keras.optimizers.AdamW(CONFIG['learning_rate'])
model.compile(optim, loss='mse', metrics=['mae'])
model.summary()

# -----------------------------
# Utilities: robust discovery of MaskedDense sublayers (recursive)
# -----------------------------

def get_masked_dense_layers(model):
    """Robust discovery of MaskedDense instances inside a Keras model.

    Reasons for this implementation:
    - Different TF/Keras versions expose nested layers via different attributes
      (e.g. `layers`, `submodules`, `_layers`). A Functional model may list
      nested/custom layers in `model.layers`, but custom submodules may be
      reachable only via other attributes on some versions.
    - We walk several common attributes defensively and deduplicate results.

    Returns a list of MaskedDense layer objects in discovery order.
    """
    visited = set()
    found = []

    def walk(layer):
        # dedupe
        if id(layer) in visited:
            return
        visited.add(id(layer))
        # collect if it's a MaskedDense
        try:
            if isinstance(layer, MaskedDense):
                found.append(layer)
        except Exception:
            pass
        # walk common container attributes in a defensive way
        for attr in ("layers", "submodules", "_layers"):
            children = getattr(layer, attr, None)
            if not children:
                continue
            try:
                for child in children:
                    # some containers include the parent itself — skip
                    if child is layer:
                        continue
                    walk(child)
            except Exception:
                # ignore errors when iterating unknown container types
                pass

    # Prefer starting from model.layers if present (Functional exposes top-level graph there)
    if hasattr(model, "layers") and getattr(model, "layers"):
        for top in model.layers:
            walk(top)
    else:
        walk(model)

    # preserve insertion order but dedupe (some traversal paths can hit the same layer twice)
    out = []
    seen = set()
    for l in found:
        if id(l) in seen:
            continue
        out.append(l)
        seen.add(id(l))
    return out

# -----------------------------
# Activation capture and entropy
# -----------------------------

def capture_activations(model, dataset, masked_layers=None):
    if masked_layers is None:
        masked_layers = get_masked_dense_layers(model)
    submodels = {}
    acts = {}
    for l in masked_layers:
        # only build submodel if the layer has an output tensor connected in the functional graph
        try:
            out_tensor = l.output
            submodels[l.name] = keras.Model(model.input, out_tensor)
            acts[l.name] = []
        except Exception:
            # skip layers that don't expose an output tensor
            continue

    for x, y in dataset:
        for name, sm in submodels.items():
            a = sm(x, training=False).numpy()
            if a.ndim == 3:
                a = a.reshape(a.shape[0]*a.shape[1], a.shape[2])
            acts[name].append(a)
    for name in list(acts.keys()):
        acts[name] = np.concatenate(acts[name], axis=0) if len(acts[name]) > 0 else np.zeros((0, masked_layers[0].units))
    return acts


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
# Improved Fisher: per-sample gradients wrt layer outputs (expensive)
# -----------------------------

def approx_fisher_per_sample(model, dataset, loss_fn, max_samples_per_batch=8):
    md_layers = get_masked_dense_layers(model)
    # build mapping name -> submodel if possible
    submodels = {}
    for l in md_layers:
        try:
            submodels[l.name] = keras.Model(model.input, l.output)
        except Exception:
            submodels[l.name] = None

    fisher = {l.name: np.zeros((l.units,), dtype=np.float32) for l in md_layers}
    count = 0
    for x_batch, y_batch in dataset:
        batch_size = tf.shape(x_batch)[0]
        n_samples = int(min(batch_size.numpy(), max_samples_per_batch))
        for i in range(n_samples):
            x = x_batch[i:i+1]
            y = y_batch[i:i+1]
            with tf.GradientTape() as tape:
                preds = model(x, training=False)
                loss = loss_fn(y, preds)
            # grad wrt each masked layer's output
            for l in md_layers:
                subm = submodels.get(l.name, None)
                if subm is None:
                    continue
                out = subm(x)
                grad = tape.gradient(loss, out)
                if grad is None:
                    continue
                grad_np = grad.numpy()
                if grad_np.ndim == 3:
                    unit_grad = np.sqrt(np.sum(grad_np[0]**2, axis=0))
                elif grad_np.ndim == 2:
                    unit_grad = np.abs(grad_np[0])
                else:
                    unit_grad = np.abs(grad_np.ravel())
                fisher[l.name] += unit_grad**2
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
        plt.figure(figsize=(6,6))
        plt.scatter(comp[:,0], comp[:,1], s=6)
        plt.title(f'PCA neuron embedding - {name}')
        plt.xlabel('PC1'); plt.ylabel('PC2')
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, f'pca_neurons_{name}.png'))
        plt.close()

# -----------------------------
# EvoCompressor (adaptive pruning + pareto logging)
# -----------------------------
class EvoCompressor:
    def __init__(self, model, val_ds, base_prune=0.2, regrow_fraction=0.2, log_dir=None):
        self.model = model
        self.val_ds = val_ds
        self.base_prune = base_prune
        self.regrow_fraction = regrow_fraction
        self.log_dir = log_dir
        self.history = []
        self.pareto = []

    def evaluate_and_compress(self, train_loss=None, val_loss=None):
        print('\nNow running evaluate_and_compress method...\n')
        md_layers = get_masked_dense_layers(self.model)

        print(f'\nMaskedDense list:\n{md_layers}\n')
        
        acts = capture_activations(self.model, self.val_ds, masked_layers=md_layers)

        print(f'\nActs items dictionery:\n{acts.items()}\n')
        
        entropies = {name: compute_activation_entropy(a) for name, a in acts.items()}
        
        print(f'\nEntropies dictionery:\n{entropies}\n')
        
        fisher = approx_fisher_per_sample(self.model, self.val_ds, keras.losses.Huber(), max_samples_per_batch=CONFIG['max_fisher_samples_per_batch'])
        utilities = {}
        for name in entropies:
            e = entropies[name]
            f = fisher.get(name, np.zeros_like(e))
            if e.size == 0 or f.size == 0:
                continue
            e_norm = (e - np.median(e)) / (np.std(e) + 1e-9)
            f_norm = (f - np.median(f)) / (np.std(f) + 1e-9)
            score = 0.5 * e_norm + 0.5 * f_norm
            utilities[name] = score

        print(f'\nUtilities dictionery:\n{utilities}\n')
        
        prune_fraction = self.base_prune
        if train_loss is not None and val_loss is not None:
            gap = float(val_loss - train_loss)
            rel = gap / (abs(val_loss) + 1e-9)
            prune_fraction = float(self.base_prune * (1.0 + rel))
            prune_fraction = np.clip(prune_fraction, 0.02, 0.6)

        total_alive = 0
        for name, score in utilities.items():
            layer = None
            print('\nFinding layer object by name...')
            for l in md_layers:
                if l.name == name:
                    layer = l
                    print(f'\nFound layer object {name} for pruning\n')
                    break
                else:
                    print(f'\n{l.name} is not what we are looking for..\n')
            if layer is None:
                print(f'\nNo prunable layer found...\n')
                continue
            mask = layer.mask.numpy()
            tf.print(mask, output_stream=sys.stdout)
            alive_idx = np.where(mask > 0.5)[0]
            total_alive += len(alive_idx)
            if len(alive_idx) == 0:
                continue
            scores_alive = score[alive_idx]
            n_prune = max(1, int(len(alive_idx) * prune_fraction))
            prune_pos = np.argsort(scores_alive)[:n_prune]
            prune_idx = alive_idx[prune_pos]
            layer.prune_units(prune_idx)
            n_regrow = int(n_prune * self.regrow_fraction)
            if n_regrow > 0:
                regrow_idx = np.random.choice(prune_idx, size=n_regrow, replace=False)
                layer.regrow_units(regrow_idx)

        val_metrics = self.model.evaluate(self.val_ds, verbose=0)
        val_loss_now = float(val_metrics[0])
        self.pareto.append((total_alive, val_loss_now))

        summary = {
            'alive_total': int(total_alive),
            'prune_fraction_used': float(prune_fraction),
            'val_loss': val_loss_now
        }
        self.history.append(summary)
        df_rows = []
        for name, layer in [(n, next((l for l in md_layers if l.name == n), None)) for n in utilities.keys()]:
            if layer is None:
                continue
            df_rows.append({'layer': name, 'alive': int(layer.mask.numpy().sum()), 'utility_mean': float(utilities[name].mean())})
        df = pd.DataFrame(df_rows)
        df.to_csv(os.path.join(self.log_dir, f'compressor_snapshot_{len(self.history)}.csv'), index=False)
        return summary

# -----------------------------
# Training + cycles
# -----------------------------

def run_experiment(model, train_ds, val_ds, config):
    log_dir = config['log_dir']
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_dir)
    chkpt_path = os.path.join(log_dir, 'best_model.keras')
    chkpt = keras.callbacks.ModelCheckpoint(chkpt_path, save_best_only=True, monitor='val_loss')
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    rl = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, cooldown=0)

    # compressor = EvoCompressor(model, val_ds, base_prune=config['prune_fraction'], regrow_fraction=config['regrow_fraction'], log_dir=log_dir)

    print('Warmup training...')
    history = model.fit(train_ds, validation_data=val_ds, epochs=config['epochs_per_phase'], callbacks=[tb_cb, chkpt, es, rl], verbose=2)

    compressor = EvoCompressor(model, val_ds, base_prune=config['prune_fraction'], regrow_fraction=config['regrow_fraction'], log_dir=log_dir)

    for cycle in range(config['cycles']):
        print(f'--- Cycle {cycle+1}/{config["cycles"]} ---')
        train_loss = float(history.history['loss'][-1])
        val_loss = float(history.history['val_loss'][-1])
        summary = compressor.evaluate_and_compress(train_loss=train_loss, val_loss=val_loss)
        print('Compressor summary:', summary)
        visualize_specialists(model, val_ds, log_dir)
        history = model.fit(train_ds, validation_data=val_ds, epochs=config['epochs_per_phase'], callbacks=[tb_cb, chkpt, es, rl], verbose=2)

    model.save(os.path.join(log_dir, 'final_model.keras'))
    return model, compressor

if __name__ == '__main__':
    trained_model, compressor = run_experiment(model, train_ds, val_ds, CONFIG)
    print('Final alive neuron counts per masked layer:')
    md_layers = get_masked_dense_layers(trained_model)
    for l in md_layers:
        print(l.name, int(l.mask.numpy().sum()), '/', l.units)
    print('Logs and snapshots saved to', CONFIG['log_dir'])
