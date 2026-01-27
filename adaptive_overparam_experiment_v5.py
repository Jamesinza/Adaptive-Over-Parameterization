"""
Adaptive Overparameterization Experiment — Emergent Phase Enhancements

Additions in this version:
- Router temperature annealing via Keras Callback (AnnealCallback).
- Router and gate statistics logged to TensorBoard each epoch (mean logits, mean gate activation, gate entropy).
- Pareto-aware pruning: compressor applies pruning only when it improves an efficiency metric (val_loss * (1 + alive_ratio)); otherwise pruning is relaxed.
- Compressor now records and logs per-layer Fisher means to TensorBoard for diagnostics.
- Startup debug prints listing discovered MaskedDense and Router layers and their initial alive counts.

Usage: run the same script. Keep CONFIG defaults or reduce sizes for quick experiments.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    'epochs_per_phase': 6,
    'cycles': 4,
    'prune_fraction': 0.20,
    'regrow_fraction': 0.20,
    'hidden_units': 1024,
    'hidden_layers': 3,
    'num_heads': 8,
    'ffn_mult': 4,
    'learning_rate': 1e-3,
    'max_fisher_samples_per_batch': 8,
    'anneal_start_temp': 1.0,
    'anneal_end_temp': 0.1,
    'anneal_epochs_total': 100,  # used by AnnealCallback schedule
    'log_dir': './logs/adaptive_overparam_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
}

os.makedirs(CONFIG['log_dir'], exist_ok=True)

# -----------------------------
# Dataset
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

X_train, Y_train = make_dataset(CONFIG['train_size'], CONFIG['seq_len'])
X_val, Y_val = make_dataset(CONFIG['val_size'], CONFIG['seq_len'])

train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(2000).batch(CONFIG['batch_size'])
val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(CONFIG['batch_size'])

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
# Masked Dense
# -----------------------------
class MaskedDense(layers.Layer):
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
            initializer=tf.keras.initializers.Constant(1.0),
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
# Router
# -----------------------------
class Router(layers.Layer):
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

# -----------------------------
# TransformerBlock
# -----------------------------
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, name=None):
        super().__init__(name=name)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn_dense1 = MaskedDense(ff_dim, activation='relu', name='ffn_dense1')
        self.ffn_dense2 = MaskedDense(d_model, activation=None, name='ffn_dense2')
        self.dropout = layers.Dropout(0.1)
        self.router = Router(units=ff_dim, temp=CONFIG['anneal_start_temp'], name='router')

    def call(self, x, training=False):
        attn_out = self.att(x, x, x)
        x = self.norm1(x + attn_out)
        pooled = tf.reduce_mean(x, axis=1)
        gates = self.router(pooled, training=training)
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
    inp = layers.Input(shape=(seq_len, 1))
    x = layers.Dense(d_model)(inp)
    pe = positional_encoding(seq_len, d_model)
    x = x + pe
    for i in range(num_layers):
        x = TransformerBlock(d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, name=f'transformer_block_{i}')(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1, name='head')(x)
    model = keras.Model(inp, out)
    return model

D_MODEL = 64
model = build_transformer_model(CONFIG['seq_len'], d_model=D_MODEL, num_layers=CONFIG['hidden_layers'], num_heads=CONFIG['num_heads'], ff_dim=CONFIG['hidden_units'])
optim = keras.optimizers.Adam(CONFIG['learning_rate'])
model.compile(optim, loss='mse', metrics=['mae'])

# -----------------------------
# Robust discovery utilities
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
        for attr in ("layers", "submodules", "_layers"):
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
    else:
        walk(model)
    out = []
    seen = set()
    for l in found:
        if id(l) in seen:
            continue
        out.append(l)
        seen.add(id(l))
    return out


def get_router_layers(model):
    visited = set()
    found = []
    def walk(layer):
        if id(layer) in visited:
            return
        visited.add(id(layer))
        try:
            if isinstance(layer, Router):
                found.append(layer)
        except Exception:
            pass
        for attr in ("layers", "submodules", "_layers"):
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
    else:
        walk(model)
    out = []
    seen = set()
    for l in found:
        if id(l) in seen:
            continue
        out.append(l)
        seen.add(id(l))
    return out


def get_transformer_blocks(model):
    # find TransformerBlock instances (parents of routers) defensively
    visited = set()
    found = []
    def walk(layer):
        if id(layer) in visited:
            return
        visited.add(id(layer))
        try:
            if isinstance(layer, TransformerBlock):
                found.append(layer)
        except Exception:
            pass
        for attr in ("layers", "submodules", "_layers"):
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
    else:
        walk(model)
    out = []
    seen = set()
    for l in found:
        if id(l) in seen:
            continue
        out.append(l)
        seen.add(id(l))
    return out


def get_router_info(model):
    """Return list of dicts: {'router': router_layer, 'block': block_layer, 'pooled_submodel': submodel}
    pooled_submodel maps model.input -> pooled context used by that block's router (reduce_mean over block input).
    """
    infos = []
    blocks = get_transformer_blocks(model)
    for blk in blocks:
        r = getattr(blk, 'router', None)
        if r is None:
            continue
        # try to build a submodel that outputs the pooled context used by the router
        try:
            # blk.input is symbolic tensor (batch, seq_len, d_model)
            pooled_tensor = tf.reduce_mean(blk.input, axis=1)
            pooled_submodel = keras.Model(model.input, pooled_tensor)
        except Exception:
            pooled_submodel = None
        infos.append({'router': r, 'block': blk, 'pooled_submodel': pooled_submodel})
    return infos

# -----------------------------
# Router logging: use pooled submodels to obtain correct router inputs
# -----------------------------
class AnnealCallback(keras.callbacks.Callback):
    def __init__(self, routers, start_temp, end_temp, total_epochs):
        super().__init__()
        self.routers = routers
        self.start_temp = float(start_temp)
        self.end_temp = float(end_temp)
        self.total_epochs = float(total_epochs)

    def on_epoch_end(self, epoch, logs=None):
        # linear anneal from start_temp to end_temp
        frac = min(1.0, epoch / max(1.0, self.total_epochs))
        t = self.start_temp + (self.end_temp - self.start_temp) * frac
        for r in self.routers:
            try:
                r.set_temp(t)
            except Exception:
                pass


class RouterLoggingCallback(keras.callbacks.Callback):
    def __init__(self, router_infos, val_ds, tb_writer):
        super().__init__()
        # router_infos: list of dicts from get_router_info() with keys 'router' and 'pooled_submodel'
        self.router_infos = router_infos
        self.val_ds = val_ds
        self.tb_writer = tb_writer

    def on_epoch_end(self, epoch, logs=None):
        mean_logits = []
        mean_gates = []
        gate_entropy = []
        # iterate a single batch for speed
        for x, y in self.val_ds:
            for info in self.router_infos:
                r = info.get('router')
                pooled_sub = info.get('pooled_submodel')
                if r is None or pooled_sub is None:
                    continue
                try:
                    pooled = pooled_sub(x, training=False)  # (batch, d_model)
                    g = r(pooled, training=False)  # (batch, units)
                except Exception:
                    continue
                # compute approximate logits (numerically stable)
                g_clipped = tf.clip_by_value(g, 1e-6, 1.0 - 1e-6)
                logits_approx = tf.math.log(g_clipped) - tf.math.log(1.0 - g_clipped)
                mean_logits.append(tf.reduce_mean(logits_approx).numpy())
                mean_gates.append(tf.reduce_mean(g).numpy())
                p = g.numpy()
                ent = -np.mean(np.sum(p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9), axis=1))
                gate_entropy.append(ent)
            break
        with (self.tb_writer.as_default() if self.tb_writer is not None else tf.summary.create_file_writer(os.path.join(CONFIG['log_dir'],'custom')).as_default()):
            tf.summary.scalar('router/mean_logit', float(np.mean(mean_logits)) if mean_logits else 0.0, step=epoch)
            tf.summary.scalar('router/mean_gate', float(np.mean(mean_gates)) if mean_gates else 0.0, step=epoch)
            tf.summary.scalar('router/gate_entropy', float(np.mean(gate_entropy)) if gate_entropy else 0.0, step=epoch)

# -----------------------------
# Training + cycles (integrate everything)

class EvoCompressor:
    def __init__(self, model, val_ds, base_prune=0.2, regrow_fraction=0.2, log_dir=None, tb_writer=None):
        self.model = model
        self.val_ds = val_ds
        self.base_prune = base_prune
        self.regrow_fraction = regrow_fraction
        self.log_dir = log_dir
        self.tb_writer = tb_writer
        self.history = []
        self.best_efficiency = np.inf

    def _alive_counts(self):
        md_layers = get_masked_dense_layers(self.model)
        return sum(int(l.mask.numpy().sum()) for l in md_layers)

    def evaluate_and_compress(self, train_loss=None, val_loss=None, step=None):
        md_layers = get_masked_dense_layers(self.model)
        acts = capture_activations(self.model, self.val_ds, masked_layers=md_layers)
        entropies = {name: compute_activation_entropy(a) for name, a in acts.items()}
        fisher = approx_fisher_per_sample(self.model, self.val_ds, keras.losses.MeanSquaredError(), max_samples_per_batch=CONFIG['max_fisher_samples_per_batch'])
        utilities = {}
        for name in entropies:
            e = entropies[name]
            f = fisher.get(name, np.zeros_like(e))
            if e.size == 0 or f.size == 0:
                continue
            e_norm = (e - np.median(e)) / (np.std(e) + 1e-9)
            f_norm = (f - np.median(f)) / (np.std(f) + 1e-9)
            utilities[name] = 0.5 * e_norm + 0.5 * f_norm

        total_params = np.sum([np.prod(w.shape) for w in self.model.trainable_weights])
        alive_total = self._alive_counts()
        alive_ratio = alive_total / float(total_params + 1e-9)

        val_metrics = self.model.evaluate(self.val_ds, verbose=0)
        val_loss_now = float(val_metrics[0])
        efficiency_now = val_loss_now * (1.0 + alive_ratio)

        apply_prune = False
        prune_fraction = self.base_prune
        if efficiency_now < self.best_efficiency:
            apply_prune = True
            self.best_efficiency = efficiency_now
        else:
            prune_fraction *= 0.5

        for name, score in utilities.items():
            layer = next((l for l in md_layers if l.name == name), None)
            if layer is None:
                continue
            mask = layer.mask.numpy()
            alive_idx = np.where(mask > 0.5)[0]
            if len(alive_idx) == 0:
                continue
            scores_alive = score[alive_idx]
            n_prune = max(1, int(len(alive_idx) * prune_fraction))
            prune_pos = np.argsort(scores_alive)[:n_prune]
            prune_idx = alive_idx[prune_pos]
            if apply_prune:
                layer.prune_units(prune_idx)
            n_regrow = int(n_prune * self.regrow_fraction)
            if n_regrow > 0:
                regrow_idx = np.random.choice(prune_idx, size=n_regrow, replace=False)
                layer.regrow_units(regrow_idx)

        if self.tb_writer is not None:
            with self.tb_writer.as_default():
                tf.summary.scalar('compressor/val_loss', val_loss_now, step=step)
                tf.summary.scalar('compressor/alive_total', alive_total, step=step)
                tf.summary.scalar('compressor/efficiency', efficiency_now, step=step)

        summary = {
            'alive_total': int(self._alive_counts()),
            'prune_fraction_used': float(prune_fraction),
            'val_loss': val_loss_now,
            'efficiency': float(efficiency_now),
            'applied_prune': bool(apply_prune)
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
# Training + cycles (integrate everything)
# (integrate everything)
# -----------------------------

def run_experiment(model, train_ds, val_ds, config):
    """Main training loop that wires up callbacks correctly.

    Fixes:
    - Builds router_infos via get_router_info(model) so we can both anneal and log routers.
    - Passes the correct 'routers' list to AnnealCallback (previously passed the model erroneously).
    - Passes router_infos (with pooled_submodels) to RouterLoggingCallback so it can compute true pooled contexts.
    """
    log_dir = config['log_dir']
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_dir)
    chkpt_path = os.path.join(log_dir, 'best_model.keras')
    chkpt = keras.callbacks.ModelCheckpoint(chkpt_path, save_best_only=True, monitor='val_loss')

    # TensorBoard writer for custom scalars
    tb_writer = tf.summary.create_file_writer(log_dir + '/custom')

    # Discover routers and their pooled context submodels
    router_infos = get_router_info(model)
    routers = [info['router'] for info in router_infos if info.get('router') is not None]

    anneal_cb = AnnealCallback(routers, config['anneal_start_temp'], config['anneal_end_temp'], config['anneal_epochs_total'])
    router_log_cb = RouterLoggingCallback(router_infos, val_ds, tb_writer)

    compressor = EvoCompressor(model, val_ds, base_prune=config['prune_fraction'], regrow_fraction=config['regrow_fraction'], log_dir=log_dir, tb_writer=tb_writer)

    print('Warmup training...')
    history = model.fit(train_ds, validation_data=val_ds, epochs=config['epochs_per_phase'], callbacks=[tb_cb, chkpt, anneal_cb, router_log_cb], verbose=2)

    global_step = 0
    for cycle in range(config['cycles']):
        print(f'--- Cycle {cycle+1}/{config["cycles"]} ---')
        train_loss = float(history.history['loss'][-1])
        val_loss = float(history.history['val_loss'][-1])
        summary = compressor.evaluate_and_compress(train_loss=train_loss, val_loss=val_loss, step=global_step)
        print('Compressor summary:', summary)
        visualize_specialists(model, val_ds, log_dir)
        history = model.fit(train_ds, validation_data=val_ds, epochs=config['epochs_per_phase'], callbacks=[tb_cb, chkpt, anneal_cb, router_log_cb], verbose=2)
        global_step += 1

    model.save(os.path.join(log_dir, 'final_model.keras'))
    return model, compressor

if __name__ == '__main__':
    trained_model, compressor = run_experiment(model, train_ds, val_ds, CONFIG)
    print('Final alive neuron counts per masked layer:')
    md_layers = get_masked_dense_layers(trained_model)
    for l in md_layers:
        print(l.name, int(l.mask.numpy().sum()), '/', l.units)
    print('Logs and snapshots saved to', CONFIG['log_dir'])
