"""
Adaptive Overparameterization Experiment
- Builds an intentionally overparameterized Keras model for time-series forecasting (next-step prediction)
- Tracks per-neuron "utility" using activation statistics and gradient-based Fisher approximation
- Applies evolutionary compression cycles: train -> evaluate -> prune -> (optional) regrow -> retrain
- Logs metrics to CSV + TensorBoard for inspection

Usage:
    python adaptive_overparam_experiment_v1.py

Dependencies:
    pip install tensorflow numpy pandas matplotlib

Notes:
    This is a configurable proof-of-concept. Swap dataset, model backbone, or pruning/regrowth rules to taste.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import datetime

# -----------------------------
# Config
# -----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CONFIG = {
    'seq_len': 50,
    'train_size': 3000,
    'val_size': 600,
    'batch_size': 64,
    'epochs_per_phase': 10,
    'cycles': 4,                # grow/prune cycles
    'prune_fraction': 0.30,     # fraction of neurons to prune each cycle
    'regrow_fraction': 0.15,    # fraction of pruned neurons to regrow
    'hidden_units': 2048,       # per-layer hidden units (overparameterized)
    'hidden_layers': 3,
    'learning_rate': 1e-3,
    'log_dir': './logs/adaptive_overparam_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
}

os.makedirs(CONFIG['log_dir'], exist_ok=True)

# -----------------------------
# Synthetic dataset (time series next-step prediction)
# -----------------------------

def make_dataset(n, seq_len):
    X = np.zeros((n, seq_len, 1), dtype=np.float32)
    Y = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        base_freq = np.random.uniform(0.5, 2.0)
        amp = np.random.uniform(0.6, 1.4)
        phase = np.random.uniform(0, 2*np.pi)
        t = np.linspace(0, 6, seq_len + 1)
        sig = amp * np.sin(2*np.pi*base_freq * t + phase)
        sig += 0.08 * np.random.randn(len(t))
        X[i, :, 0] = sig[:-1]
        Y[i, 0] = sig[-1]
    return X, Y

# Create datasets
X_train, Y_train = make_dataset(CONFIG['train_size'], CONFIG['seq_len'])
X_val, Y_val = make_dataset(CONFIG['val_size'], CONFIG['seq_len'])

train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(2000).batch(CONFIG['batch_size'])
val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(CONFIG['batch_size'])

# -----------------------------
# Masked Dense layer (supports neuron-level masking for pruning/regrowth)
# -----------------------------
class MaskedDense(layers.Layer):
    def __init__(self, units, activation=None, name=None):
        super().__init__(name=name)
        self.units = units
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
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False
        )

    def call(self, inputs):
        masked_kernel = self.kernel * tf.reshape(self.mask, (1, -1))
        masked_bias = self.bias * self.mask
        x = tf.matmul(inputs, masked_kernel) + masked_bias
        if self.activation is not None:
            x = self.activation(x)
        return x

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

# -----------------------------
# Build overparameterized model
# -----------------------------

def build_overparam_model(seq_len, hidden_units, hidden_layers):
    inp = layers.Input(shape=(seq_len, 1))
    x = layers.Flatten()(inp)
    for i in range(hidden_layers):
        x = MaskedDense(hidden_units, activation='relu', name=f'masked_dense_{i}')(x)
    out = layers.Dense(1, name='head')(x)
    model = keras.Model(inp, out)
    return model

model = build_overparam_model(CONFIG['seq_len'], CONFIG['hidden_units'], CONFIG['hidden_layers'])
optim = keras.optimizers.Adam(CONFIG['learning_rate'])
model.compile(optim, loss='mse', metrics=['mae'])

# -----------------------------
# Utilities: activation capture & per-neuron utility calculation
# -----------------------------

def capture_activations(model, dataset, layer_names=None):
    if layer_names is None:
        layer_names = [l.name for l in model.layers if isinstance(l, MaskedDense)]

    submodels = {name: keras.Model(model.input, model.get_layer(name).output) for name in layer_names}
    acts = {name: [] for name in layer_names}
    for x, y in dataset:
        for name, sm in submodels.items():
            a = sm(x, training=False).numpy()
            acts[name].append(a)
    for name in layer_names:
        acts[name] = np.concatenate(acts[name], axis=0)
    return acts


def compute_activation_entropy(activations, eps=1e-9):
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


def approx_fisher_information(model, dataset, loss_fn):
    fisher = {}
    md_layers = [l for l in model.layers if isinstance(l, MaskedDense)]
    for l in md_layers:
        units = l.units
        fisher[l.name] = np.zeros((units,), dtype=np.float32)

    for x, y in dataset:
        with tf.GradientTape() as tape:
            preds = model(x, training=False)
            loss = loss_fn(y, preds)
        grads = tape.gradient(loss, [l.kernel for l in md_layers] + [l.bias for l in md_layers])
        k_grads = grads[:len(md_layers)]
        b_grads = grads[len(md_layers):]
        for i, l in enumerate(md_layers):
            gk = k_grads[i].numpy()
            gb = b_grads[i].numpy()
            unit_grads = np.sqrt(np.sum(gk**2, axis=0) + gb**2)
            fisher[l.name] += unit_grads**2
    for name in fisher:
        fisher[name] /= (len(list(dataset)) + 1e-9)
    return fisher

# -----------------------------
# Evolutionary Compression Controller
# -----------------------------
class EvoCompressor:
    def __init__(self, model, val_ds, prune_fraction=0.3, regrow_fraction=0.15, log_dir=None):
        self.model = model
        self.val_ds = val_ds
        self.prune_fraction = prune_fraction
        self.regrow_fraction = regrow_fraction
        self.log_dir = log_dir
        self.history = []

    def evaluate_and_compress(self):
        acts = capture_activations(self.model, self.val_ds)
        entropies = {name: compute_activation_entropy(a) for name, a in acts.items()}
        fisher = approx_fisher_information(self.model, self.val_ds, keras.losses.MeanSquaredError())
        utilities = {}
        for name in entropies:
            e = entropies[name]
            f = fisher.get(name, np.zeros_like(e))
            e_norm = (e - e.mean()) / (e.std() + 1e-9)
            f_norm = (f - f.mean()) / (f.std() + 1e-9)
            score = 0.5 * e_norm + 0.5 * f_norm
            utilities[name] = score
        for name, score in utilities.items():
            layer = self.model.get_layer(name)
            mask = layer.mask.numpy()
            alive_idx = np.where(mask > 0.5)[0]
            if len(alive_idx) == 0:
                continue
            scores_alive = score[alive_idx]
            n_prune = max(1, int(len(alive_idx) * self.prune_fraction))
            prune_pos = np.argsort(scores_alive)[:n_prune]
            prune_idx = alive_idx[prune_pos]
            layer.prune_units(prune_idx)
            n_regrow = int(n_prune * self.regrow_fraction)
            if n_regrow > 0:
                regrow_idx = np.random.choice(prune_idx, size=n_regrow, replace=False)
                layer.regrow_units(regrow_idx)
        summary = {
            'alive_counts': {name: int(layer.mask.numpy().sum()) for name, layer in [(n, self.model.get_layer(n)) for n in utilities.keys()]},
            'utilities_mean': {name: float(score.mean()) for name, score in utilities.items()}
        }
        self.history.append(summary)
        rows = []
        for name, layer in [(n, self.model.get_layer(n)) for n in utilities.keys()]:
            rows.append({'layer': name, 'alive': int(layer.mask.numpy().sum()), 'utility_mean': float(utilities[name].mean())})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.log_dir, f'compressor_snapshot_{len(self.history)}.csv'), index=False)
        return summary

# -----------------------------
# Training + cycles
# -----------------------------

def run_experiment(model, train_ds, val_ds, config):
    log_dir = config['log_dir']
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_dir)
    chkpt = keras.callbacks.ModelCheckpoint(os.path.join(log_dir, 'best_model.keras'), save_best_only=True, monitor='val_loss')

    compressor = EvoCompressor(model, val_ds, prune_fraction=config['prune_fraction'], regrow_fraction=config['regrow_fraction'], log_dir=log_dir)

    print('Warmup training...')
    model.fit(train_ds, validation_data=val_ds, epochs=config['epochs_per_phase'], callbacks=[tb_cb, chkpt], verbose=2)

    for cycle in range(config['cycles']):
        print(f'--- Cycle {cycle+1}/{config["cycles"]} ---')
        summary = compressor.evaluate_and_compress()
        print('Compressor summary:', summary)
        model.fit(train_ds, validation_data=val_ds, epochs=config['epochs_per_phase'], callbacks=[tb_cb, chkpt], verbose=2)

    return model, compressor

if __name__ == '__main__':
    trained_model, compressor = run_experiment(model, train_ds, val_ds, CONFIG)
    print('Final alive neuron counts per masked layer:')
    for l in trained_model.layers:
        if isinstance(l, MaskedDense):
            print(l.name, int(l.mask.numpy().sum()), '/', l.units)
    print('Logs and snapshots saved to', CONFIG['log_dir'])
