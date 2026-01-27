"""adaptive_overparam_experiment_v39.py"""

import os
import math
import random
import logging
import datetime
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from contextlib import contextmanager
from sklearn.decomposition import PCA
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List, Iterable
import tensorflow as tf
import keras
import tensorflow.nest as tfnest

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
    'warm_up_epochs': 100,
    'epochs_per_phase': 1000,
    'cycles': 1000,
    'regrow_finetune_steps': 3,
    'mask_thresh_min': 0.1,
    'mask_thresh_multi': 0.05,
    'prune_fraction': 0.4,
    'regrow_fraction': 0.5,
    'connection_rewire_fraction': 0.5,
    'd_model': 8,
    'hidden_units': 32,
    'hidden_layers': 2,
    'num_heads': 8,
    'learning_rate': 1e-3,
    'max_fisher_samples_per_batch': 8,
    'max_grad_samples_per_batch': 8,
    'cycle_patience': 30,
    'cycle_min_delta': 1e-10,
    'cycle_monitor': 'val_loss_final',
    'save_best_cycle_model': True,
    'cycle_min_rel': 1e-3,   # relative improvement (default 0.1%)
    'cycle_min_abs': 1e-12,  # absolute fallback
    'log_dir': './logs/adaptive_overparam_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),    
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

# ---------------------------------------------------------------------------
# Configuration dataclass (extended for layer-wise control + quality metrics)
# ---------------------------------------------------------------------------
@dataclass
class EvoConfig:
    base_prune: float = CONFIG['prune_fraction']
    regrow_fraction: float = CONFIG['regrow_fraction']
    connection_rewire_fraction: float = CONFIG['connection_rewire_fraction']
    regrow_loss_tol: float = 0.01
    prune_search: bool = False
    mask_thresh_multi: float = CONFIG['mask_thresh_multi']
    mask_thresh_min: float = CONFIG['mask_thresh_min']
    log_dir: Optional[str] = CONFIG['log_dir']
    debug: bool = False
    max_prune_frac: float = 0.7
    min_prune_frac: float = 0.1
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
    # grad_unit_weight: float = 0.3                # how much grad contributes to unit score
    # fisher_weight: float = 0.3                   # how much fisher contributes to unit score
    max_grad_samples: int = CONFIG['max_grad_samples_per_batch'] # cap number of val batches to use for gradient-based scoring

    # Ablation / EMA / regrow controls
    ablation_max_batches: int = 32               # val batches to use for ablation scoring
    ablation_max_neurons: int = 256              # max neurons to test per layer
    # ablation_weight: float = 0.3                 # how much ablation contributes to unit score
    ema_decay: float = 0.7                       # EMA smoothing factor for unit utilities
    deterministic_regrow_seed: int = SEED        # base seed for deterministic regrow reinit
    regrow_reinit_std: float = 0.01              # relative std when reinitializing regrown connections

    # Global finetune controls
    regrow_finetune_steps: int = CONFIG['regrow_finetune_steps']
    finetune_batch_size: Optional[int] = CONFIG['batch_size']//2    

    # Per-layer short finetune controls
    layer_finetune_steps: int = 3                    # number of train_on_batch steps after accepting a layer prune
    layer_finetune_batch_size: Optional[int] = CONFIG['batch_size']//4  # fallback to config finetune_batch_size if None
    layer_finetune_lr_mul: float = 0.5               # scale applied to gradients during per-layer finetune (<=1 for conservative)

    # CREC-specific defaults
    crec_init_regrow_fraction = 0.5
    crec_max_regrow_fraction = 0.2
    crec_fisher_target_scale = 0.8
    crec_entropy_target = 0.5
    crec_fisher_k = 50.0
    crec_val_k = 1.0
    crec_entropy_k = 1.0
    crec_compression_tradeoff = 1.0
    crec_momentum = 0.9
    crec_history_window = 8
    crec_max_delta_per_step = 0.05

    # Controller (online neural subcontroller) defaults
    use_controller = True
    controller_lr = 1e-5
    controller_hidden_units = (8, 4)
    controller_ent_coef = 1e-3
    controller_max_delta_regrow = 0.05
    controller_max_delta_comp = 0.2
    controller_baseline_decay = 0.99
    

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


# ---------------------------
# Online neural subcontroller (tiny policy network)
# ---------------------------
class OnlineNeuralSubcontroller(tf.keras.Model):
    """
    Stochastic Gaussian policy returning two actions:
      - delta_regrow (applied to regrow_fraction)
      - delta_compression (applied to compression_tradeoff)
    Trained online with REINFORCE (policy gradient) using a running baseline (EMA).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_units: Tuple[int, ...] = (64, 32),
        lr: float = 1e-4,
        ent_coef: float = 1e-3,
        max_delta_regrow: float = 0.05,
        max_delta_comp: float = 0.2,
        baseline_decay: float = 0.99,
        name: str = "online_subcontroller",
    ):
        super().__init__(name=name)
        self.max_delta_regrow = float(max_delta_regrow)
        self.max_delta_comp = float(max_delta_comp)
        self.ent_coef = float(ent_coef)
        self.baseline_decay = float(baseline_decay)

        # small MLP -> means
        self._mlp = tf.keras.Sequential(name=f"{name}_mlp")
        for i, u in enumerate(hidden_units):
            self._mlp.add(tf.keras.layers.Dense(u, activation="elu", name=f"dense_{i}"))
        self._mean_head = tf.keras.layers.Dense(2, activation=None, name="mean_head")
        # learnable logstd
        init_logstd = -3.0
        self.logstd = tf.Variable(tf.ones((2,)) * init_logstd, trainable=True, name="logstd")

        self.opt = tf.keras.optimizers.AdamW(learning_rate=lr)

        # running baseline for advantage
        self.reward_baseline = 0.0
        self._baseline_initialized = False

        # last stored sampling info for online update
        self._last_state = None
        self._last_action = None
        self._last_logp = None

    def call(self, state: tf.Tensor):
        x = tf.convert_to_tensor(state, dtype=tf.float32)
        if len(x.shape) == 1:
            x = tf.expand_dims(x, 0)
        h = self._mlp(x)
        mean = self._mean_head(h)  # (1,2)
        std = tf.exp(self.logstd)  # (2,)
        return tf.squeeze(mean, axis=0), std  # (2,), (2,)

    def sample_action(self, state: np.ndarray):
        mean, std = self.call(state)
        mean_np = mean.numpy()
        std_np = std.numpy()
        eps = np.random.randn(*mean_np.shape).astype(np.float32)
        raw_action = mean_np + eps * std_np
        var = std_np ** 2 + 1e-12
        logp = -0.5 * np.sum(((raw_action - mean_np) ** 2) / var + np.log(2 * np.pi * var))
        # bound and scale via tanh
        a0 = np.tanh(raw_action[0]) * self.max_delta_regrow
        a1 = np.tanh(raw_action[1]) * self.max_delta_comp
        action = np.array([a0, a1], dtype=np.float32)
        self._last_state = np.asarray(state, dtype=np.float32)
        self._last_action = raw_action.astype(np.float32)
        self._last_logp = float(logp)
        return action, float(logp), mean_np, std_np

    def learn_from_reward(self, reward: float, entropy_bonus: float = None):
        if self._last_state is None:
            return None, None

        if not self._baseline_initialized:
            self.reward_baseline = float(reward)
            self._baseline_initialized = True

        advantage = float(reward) - float(self.reward_baseline)
        # ema update
        self.reward_baseline = float(self.baseline_decay * self.reward_baseline + (1.0 - self.baseline_decay) * float(reward))

        raw_action = tf.convert_to_tensor(self._last_action, dtype=tf.float32)
        state = tf.convert_to_tensor(self._last_state[None, :], dtype=tf.float32)  # (1,dim)

        with tf.GradientTape() as tape:
            h = self._mlp(state)
            mean = self._mean_head(h)[0]
            std = tf.exp(self.logstd)
            var = std ** 2 + 1e-12
            log_prob = -0.5 * tf.reduce_sum(((raw_action - mean) ** 2) / var + tf.math.log(2.0 * np.pi * var))
            loss_pg = - (advantage) * log_prob
            if entropy_bonus is None:
                entropy_bonus = self.ent_coef
            entropy = 0.5 * tf.reduce_sum(tf.math.log(2.0 * np.pi * np.e * var))
            loss = loss_pg - entropy_bonus * entropy

        grads = tape.gradient(loss, self.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, self.trainable_variables) if g is not None]
        if grads_and_vars:
            self.opt.apply_gradients(grads_and_vars)

        self._last_state = None
        self._last_action = None
        self._last_logp = None

        return float(advantage), float(loss.numpy())


# ------------------------------------------------------
# NeuroAdaptiveCREC: compression & regrowth controller
# ------------------------------------------------------
class NeuroAdaptiveCREC:
    """
    NeuroAdaptiveCREC:
    Self-regulating compression–regrowth controller with dynamic Fisher and entropy targets.
    Balances pruning, regrowth, and model sensitivity automatically.
    """

    def __init__(self,
                 init_regrow_fraction=0.05,
                 max_regrow_fraction=0.25,
                 fisher_target_scale=0.8,
                 entropy_target=0.5,
                 fisher_k=50.0,
                 val_k=20.0,
                 entropy_k=10.0,
                 compression_tradeoff=1.0,
                 momentum=0.9,
                 history_window=20,
                 max_delta_per_step=0.05
                ):

        # Core parameters
        self.regrow_fraction = init_regrow_fraction
        self.max_regrow_fraction = max_regrow_fraction
        self.fisher_target_scale = fisher_target_scale
        self.entropy_target = entropy_target
        self.fisher_k = fisher_k
        self.val_k = val_k
        self.entropy_k = entropy_k
        self.compression_tradeoff = compression_tradeoff
        self.momentum = momentum
        self.history_window = history_window
        self.max_delta_per_step = float(max_delta_per_step)

        # Rolling histories
        self.fisher_history = []
        self.entropy_history = []
        self.val_history = []

        # Internal state
        self.smoothed_regrow = init_regrow_fraction
        self.fisher_target = None
        self.entropy_target_dynamic = entropy_target

    # ------------------------------------------------------
    # Utility
    # ------------------------------------------------------
    def _mask_entropy(self, mask):
        # mask is expected to be float in [0,1] per-unit. compute fraction active p in [0,1].
        if getattr(mask, 'size', 0) == 0:
            return 0.0
        p = float(np.mean(mask))
        # numerical guard
        if p <= 0.0 or p >= 1.0:
            return 0.0
        # normalized entropy (divide by ln(2) so max = 1.0)
        return (-(p * np.log(p) + (1 - p) * np.log(1 - p))) / np.log(2.0)

    # ------------------------------------------------------
    # Calibration
    # ------------------------------------------------------
    def calibrate_targets(self, fisher_metric, mask_entropies):
        """Update Fisher and entropy targets using rolling median/mean history."""
        # robust append
        self.fisher_history.append(float(fisher_metric))
        self.entropy_history.append(float(np.mean(mask_entropies)) if len(mask_entropies) > 0 else 0.0)

        # keep window
        if len(self.fisher_history) > self.history_window:
            self.fisher_history.pop(0)
            self.entropy_history.pop(0)

        # Dynamic Fisher target = scaled median
        median_f = float(np.median(self.fisher_history)) if len(self.fisher_history) > 0 else 0.0
        self.fisher_target = median_f * self.fisher_target_scale

        # Dynamic entropy target: blend between observed mean and 0.5 (normalized domain)
        mean_ent = float(np.mean(self.entropy_history)) if len(self.entropy_history) > 0 else 0.0
        # keep in [0,1]
        self.entropy_target_dynamic = float(np.clip(0.8 * mean_ent + 0.2 * 0.5, 0.0, 1.0))


    # ------------------------------------------------------
    # Core adaptive update
    # ------------------------------------------------------
    def update(self, fisher_metric, val_delta, masks):
        """Update regrow_fraction and compression_tradeoff adaptively.
        `masks` can be either a list of mask arrays (as before) or a list of per-layer
        aggregated entropy-like scalars in [0,1]. We accept both for backwards compatibility.
        """
        # accept scalars (already normalized entropy in [0,1]) or arrays
        mask_entropies = []
        for m in masks:
            if m is None:
                continue
            # scalar path: assumed to be pre-computed entropy in [0,1]
            if np.isscalar(m):
                try:
                    mask_entropies.append(float(m))
                except Exception:
                    mask_entropies.append(0.0)
                continue
            # array-like path: compute neuron-level alive fraction and entropy normalized by ln(2)
            try:
                arr = np.asarray(m)
                if arr.size == 0:
                    mask_entropies.append(0.0)
                    continue
                p = float(np.mean(arr))
                if p <= 0.0 or p >= 1.0:
                    mask_entropies.append(0.0)
                else:
                    mask_entropies.append((-(p * np.log(p) + (1 - p) * np.log(1 - p))) / np.log(2.0))
            except Exception:
                mask_entropies.append(0.0)
    
        mean_entropy = np.mean(mask_entropies) if len(mask_entropies) > 0 else 0.0
    
        # --- self-calibrate targets before using them ---
        self.calibrate_targets(fisher_metric, mask_entropies)
    
        # Compute deviations
        fisher_term = (self.fisher_target - fisher_metric) * self.fisher_k
        val_term = -val_delta * self.val_k
        entropy_term = (self.entropy_target_dynamic - mean_entropy) * self.entropy_k
    
        delta = fisher_term + val_term + entropy_term
    
        # Update regrow fraction with smoothing
        new_r = np.clip(self.regrow_fraction + delta, 0.0, self.max_regrow_fraction)
        self.smoothed_regrow = (
            self.momentum * self.smoothed_regrow + (1 - self.momentum) * new_r
        )
        self.regrow_fraction = float(self.smoothed_regrow)
    
        # --- Update compression_tradeoff ---
        compression_adjust = (
            (fisher_metric - self.fisher_target) * 0.05
            - (mean_entropy - self.entropy_target_dynamic) * 0.1
        )
        self.compression_tradeoff = np.clip(
            self.compression_tradeoff + compression_adjust, 0.1, 10.0
        )
    
        # Save val metric
        self.val_history.append(val_delta)
        if len(self.val_history) > self.history_window:
            self.val_history.pop(0)
    
        return {
            "regrow_fraction": self.regrow_fraction,
            "compression_tradeoff": self.compression_tradeoff,
            "fisher_target": self.fisher_target,
            "entropy_target": self.entropy_target_dynamic,
            "mean_entropy": mean_entropy,
        }


    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------
    def summary(self):
        return {
            "regrow_fraction": round(self.regrow_fraction, 5),
            "compression_tradeoff": round(self.compression_tradeoff, 3),
            "fisher_target": round(self.fisher_target, 6) if self.fisher_target else None,
            "entropy_target": round(self.entropy_target_dynamic, 4),
        }


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
        self.pareto_logger = defaultdict(list)
        self.conn_age_log = defaultdict(list)

        # store last layerwise decisions for regrowth logic / debugging
        self.last_layer_decisions: Dict[str, Dict[str, Any]] = {}

        # internal
        self._setup_logger()

        # integrate NeuroAdaptiveCREC instance
        self.crec = NeuroAdaptiveCREC(
            init_regrow_fraction=getattr(self.config, 'crec_init_regrow_fraction', 0.05),
            max_regrow_fraction=getattr(self.config, 'crec_max_regrow_fraction', 0.25),
            fisher_target_scale=getattr(self.config, 'crec_fisher_target_scale', 0.8),
            entropy_target=getattr(self.config, 'crec_entropy_target', 0.5),
            fisher_k=getattr(self.config, 'crec_fisher_k', 50.0),
            val_k=getattr(self.config, 'crec_val_k', 20.0),
            entropy_k=getattr(self.config, 'crec_entropy_k', 10.0),
            compression_tradeoff=getattr(self.config, 'crec_compression_tradeoff', 1.0),
            momentum=getattr(self.config, 'crec_momentum', 0.9),
            history_window=getattr(self.config, 'crec_history_window', 20),
        )

        # optional online controller for NeuroAdaptiveCREC
        if getattr(self.config, 'use_controller', False):
            state_dim = 8  # fisher, mean_layer_entropy, val_delta, crec_regrow, crec_comp, mean_util_var, mean_conn_density, prune_fraction
            self.controller = OnlineNeuralSubcontroller(
                state_dim=state_dim,
                hidden_units=getattr(self.config, 'controller_hidden_units', (64, 32)),
                lr=getattr(self.config, 'controller_lr', 1e-4),
                ent_coef=getattr(self.config, 'controller_ent_coef', 1e-3),
                max_delta_regrow=getattr(self.config, 'controller_max_delta_regrow', 0.05),
                max_delta_comp=getattr(self.config, 'controller_max_delta_comp', 0.2),
                baseline_decay=getattr(self.config, 'controller_baseline_decay', 0.99),
            )
        else:
            self.controller = None
        self._controller_last = None

        # track last observed val loss for delta computations
        self.last_val_loss: Optional[float] = None

    def _setup_logger(self):
        name = 'EvoCompressor'
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            level = logging.DEBUG if self.config.debug else logging.INFO
            self.logger.setLevel(level)
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            ch.setLevel(level)
            self.logger.addHandler(ch)
        self.log = self.logger

    # ------------ hook points inserted into evaluate_and_compress ------------
    def evaluate_and_compress(self, train_loss: Optional[float] = None, val_loss: Optional[float] = None, do_regrow: bool = True) -> Tuple[Dict[str, Any], float]:
        self.log.info('Starting evaluate_and_compress cycle (CREC-enabled)')

        md_layers = get_masked_dense_layers(self.model)

        # compute utilities (unchanged behavior)
        utilities = self.compute_utilities(md_layers)

        # --- CREC: derive scalar fisher metric and richer mask/conn entropies from layer objects ---
        fisher_vals = []
        layer_entropy_metrics = []   # aggregated per-layer entropy-like signals (normalized 0..1)
        debug_layer_info = []
        
        for l in md_layers:
            # fisher (mean abs)
            f = getattr(l, '_last_fisher', None)
            if f is None or getattr(f, 'size', 0) == 0:
                continue
            fisher_vals.append(float(np.mean(np.abs(f))))
        
            # neuron-level alive fraction (one value per output neuron mask)
            m = l.mask.numpy() if hasattr(l, 'mask') else np.array([])
            p_neuron = float(np.mean(m)) if m.size else 0.0   # already in [0,1]
        
            # connection-level alive fraction (conn_mask shape = (in_dim, out_dim))
            conn = l.conn_mask.numpy() if hasattr(l, 'conn_mask') else np.array([])
            p_conn = float(np.mean(conn)) if conn.size else 0.0
        
            # normalized entropy helper (divided by ln(2) to map max->1.0)
            def _norm_entropy(p):
                if p <= 0.0 or p >= 1.0:
                    return 0.0
                return (-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))) / np.log(2.0)
        
            ent_neuron = _norm_entropy(p_neuron)   # 0..1
            ent_conn = _norm_entropy(p_conn)       # 0..1
        
            # utility variance signal (use EMA unit score if available)
            util = getattr(l, '_ema_unit_score', None)
            if util is None or util.size == 0:
                util_var = 0.0
            else:
                util_arr = np.asarray(util, dtype=np.float32)
                util_var = float(np.clip(np.var(util_arr) / (np.mean(np.abs(util_arr)) + 1e-9), 0.0, 1.0))
        
            # aggregate into a single per-layer entropy-like metric
            layer_entropy = 0.6 * ent_conn + 0.3 * ent_neuron + 0.1 * util_var
            layer_entropy = float(np.clip(layer_entropy, 0.0, 1.0))
        
            layer_entropy_metrics.append(layer_entropy)
            debug_layer_info.append({
                'name': f"{l.name}_{hex(id(l))}",
                'p_neuron': p_neuron,
                'p_conn': p_conn,
                'ent_neuron': ent_neuron,
                'ent_conn': ent_conn,
                'util_var': util_var,
                'agg_entropy': layer_entropy,
                'fisher_mean_abs': float(np.mean(np.abs(f))),
            })
        
        # scalar fisher metric: median of per-layer mean-abs fisher (robust)
        fisher_metric = float(np.median(fisher_vals)) if fisher_vals else 0.0

        # baseline val loss
        baseline_val_loss = self._obtain_baseline_loss(val_loss)        
        
        # compute val_delta relative to last observed val
        if self.last_val_loss is None:
            val_delta = 0.0
        else:
            # positive = improvement (previous -> current lowered loss)
            val_delta = float(self.last_val_loss - baseline_val_loss)
        
        # Pass aggregated per-layer scalars to CREC (CREC.update handles scalar entropies)
        try:
            crec_info = self.crec.update(fisher_metric, val_delta, layer_entropy_metrics)
            self.log.debug(f'CREC update: {crec_info}')
        except Exception as e:
            self.log.warning(f'CREC update failed: {e}')
            crec_info = self.crec.summary()
        
        # helpful debug printing
        mean_entropy = float(np.mean(layer_entropy_metrics)) if layer_entropy_metrics else 0.0
        self.log.debug(f'CREC input: fisher_metric={fisher_metric:.3e}, mean_layer_entropy={mean_entropy:.4f}')
        for info in debug_layer_info:
            self.log.debug(
                f"layer={info['name']} p_neuron={info['p_neuron']:.3f} p_conn={info['p_conn']:.3f} "
                f"ent_conn={info['ent_conn']:.3f} ent_neuron={info['ent_neuron']:.3f} util_var={info['util_var']:.3f} agg={info['agg_entropy']:.3f}"
            )

        # decide dynamic prune fraction (CREC may influence future regrow)
        prune_fraction = self._compute_dynamic_prune_fraction(train_loss, val_loss)
        self.log.info(f'Computed prune_fraction={prune_fraction:.4f}')

        # controller integration: build compact state and let controller propose small deltas
        # first fetch CREC's suggestions
        crec_regrow = float(crec_info.get('regrow_fraction', self.crec.regrow_fraction))
        crec_comp = float(crec_info.get('compression_tradeoff', self.crec.compression_tradeoff))

        # compute a couple of extra signals for the state
        mean_util_var = 0.0
        mean_conn_density = 1.0
        util_vars = []
        conn_ds = []
        for l in md_layers:
            util = getattr(l, '_ema_unit_score', None)
            if util is not None and getattr(util, 'size', 0) > 0:
                util_vars.append(float(np.var(np.asarray(util, dtype=np.float32))))
            cm = getattr(l, 'conn_mask', None)
            if cm is not None:
                try:
                    conn_ds.append(float(np.mean(cm.numpy())))
                except Exception:
                    pass
        if util_vars:
            mean_util_var = float(np.mean(util_vars))
        if conn_ds:
            mean_conn_density = float(np.mean(conn_ds))

        prune_frac_for_state = float(prune_fraction) if 'prune_fraction' in locals() else 0.0

        state = np.array([
            fisher_metric,
            mean_entropy,
            val_delta,
            crec_regrow,
            crec_comp,
            mean_util_var,
            mean_conn_density,
            prune_frac_for_state,
        ], dtype=np.float32)

        # sample controller action (if controller enabled)
        if getattr(self, 'controller', None) is not None:
            action, logp, mean, std = self.controller.sample_action(state)
            # action = [delta_regrow, delta_comp]
            applied_regrow = float(np.clip(crec_regrow + float(action[0]), 0.0, self.crec.max_regrow_fraction))
            applied_comp = float(np.clip(crec_comp + float(action[1]), 0.1, 10.0))

            # stash overrides and original values so we can restore and learn later
            self._controller_last = {
                'state': state,
                'action': action,
                'logp': logp,
                'crec_regrow': crec_regrow,
                'crec_comp': crec_comp,
                'applied_regrow': applied_regrow,
                'applied_comp': applied_comp,
            }

            # apply overrides for the downstream regrow decision logic
            self._controller_old_regrow = getattr(self.config, 'regrow_fraction', 0.05)
            self._controller_old_comp = getattr(self.crec, 'compression_tradeoff', self.config.compression_tradeoff)
            self.config.regrow_fraction = applied_regrow
            self.crec.compression_tradeoff = applied_comp
        else:
            self._controller_last = None

        # compute prune candidates
        stored_prune_info, total_alive_pre, mask_thresh_min = self.compute_prune_candidates(md_layers, utilities, prune_fraction)
        self.log.info(f'Total alive neurons pre-prune: {total_alive_pre}')

        # proceed with prune evaluation and regrowth as before
        best_scenario, pruned_val_loss, baseline_weights = self._search_or_apply_scenarios(stored_prune_info, total_alive_pre)
        self.log.info(f'Chosen pruning scenario: {best_scenario} => pruned_val_loss={pruned_val_loss:.6g}')

        regrow_triggered = False
        val_loss_after_regrow = pruned_val_loss

        if do_regrow and (pruned_val_loss > baseline_val_loss * (1.0 + self.config.regrow_loss_tol)):
            self.log.info('Pruned model degraded beyond tolerance; evaluating finetune before regrow')
            val_after_ft = self._finetune_short()
            self.log.info(f'Val loss after short finetune: {val_after_ft:.6g}')

            if val_after_ft <= baseline_val_loss * (1.0 + self.config.regrow_loss_tol):
                self.log.info('Short finetune recovered performance; skipping regrowth')
                val_loss_after_regrow = val_after_ft
            else:
                regrow_triggered = True

                # use CREC regrow_fraction override if provided
                target_regrow_frac = crec_info.get('regrow_fraction', getattr(self.config, 'regrow_fraction', 0.05))

                # choose target layers based on stored per-layer rel_delta and CREC-guided regrow fraction
                target_layers = []
                for lname, info in stored_prune_info.items():
                    rel = info.get('rel_delta', 0.0)
                    # if layer caused notable relative degradation, target it
                    if rel >= self.config.layer_regrow_min_rel_loss:
                        target_layers.append(lname)

                if not target_layers:
                    self.log.debug('No per-layer delta exceeded threshold; regrowing across all layers')
                    target_layers = list(stored_prune_info.keys())

                # temporarily override config.regrow_fraction for deterministic regrow counts
                old_rf = getattr(self.config, 'regrow_fraction', 0.05)
                try:
                    self.config.regrow_fraction = float(np.clip(target_regrow_frac, 0.0, self.crec.max_regrow_fraction))
                    self.perform_regrowth(stored_prune_info, utilities, target_layers=target_layers)
                finally:
                    # restore config value
                    self.config.regrow_fraction = old_rf

                val_metrics_after = self._evaluate_val()
                val_loss_after_regrow = float(val_metrics_after[0])
                self.log.info(f'Val loss after regrowth: {val_loss_after_regrow:.6g}')

        # finalize stats and compute final loss
        df_rows = self.finalize_stats()
        val_metrics_final = self._evaluate_val()
        val_loss_final = float(val_metrics_final[0])
        self.log.info(f'Final validation loss: {val_loss_final:.6g}')

        # post-compression decision - inject CREC's compression_tradeoff as suggestion
        compression_tradeoff_for_decision = crec_info.get('compression_tradeoff', getattr(self.config, 'compression_tradeoff', 0.15))
        action, reason = self._post_compression_decision(
            baseline_val_loss=baseline_val_loss,
            val_loss_final=val_loss_final,
            baseline_alive=total_alive_pre,
            final_alive=int(sum(r['alive_neurons'] for r in df_rows)),
            baseline_weights=baseline_weights,
            policy=self.config.compress_policy if hasattr(self.config, 'compress_policy') else 'rollback',
            rel_tol=0.01,
            compression_tradeoff=compression_tradeoff_for_decision,
            long_retrain_epochs=self.config.long_retrain_epochs,
            train_ds=self.train_ds,
            finetune_bs=self.config.finetune_batch_size
        )
        self.log.info(f'Post-compression decision: action={action}; reason={reason}')

        if action.startswith('rolled_back'):
            df_rows = self.finalize_stats()
            val_metrics_final = self._evaluate_val()
            val_loss_final = float(val_metrics_final[0])

        total_alive_now = int(sum(r['alive_neurons'] for r in df_rows))
        # record CREC summary in history
        crec_summary = self.crec.summary()

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
            'crec': crec_summary,
        }
        self.history.append(summary)

        # ----- Controller learning (online) -----
        if getattr(self, 'controller', None) is not None and getattr(self, '_controller_last', None) is not None:
            try:
                baseline_alive = int(total_alive_pre) if 'total_alive_pre' in locals() else int(0)
                total_alive_now = int(total_alive_now) if 'total_alive_now' in locals() else int(sum(r['alive_neurons'] for r in df_rows))
                compression_gain = (baseline_alive - total_alive_now) / (baseline_alive + 1e-12)

                # normalized val improvement (positive is better)
                val_improve_norm = (baseline_val_loss - val_loss_final) / (abs(baseline_val_loss) + 1e-12)

                # reward shaping: you can tune weights w_val and w_comp
                w_val = 1.0
                w_comp = 0.5
                reward = float(w_val * val_improve_norm + w_comp * compression_gain)

                adv, loss = self.controller.learn_from_reward(reward)
                self.log.debug(f'Controller learn: adv={adv} loss={loss} reward={reward:.6g}')

            except Exception as e:
                self.log.warning(f'Controller learning failed: {e}')
            finally:
                # restore config/CREC original values so next cycle starts clean
                if hasattr(self, '_controller_old_regrow'):
                    try:
                        self.config.regrow_fraction = self._controller_old_regrow
                    except Exception:
                        pass
                if hasattr(self, '_controller_old_comp'):
                    try:
                        self.crec.compression_tradeoff = self._controller_old_comp
                    except Exception:
                        pass
                self._controller_last = None

        # update last_val_loss for next cycle
        self.last_val_loss = val_loss_final

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

        # return summary and mask threshold minimum used earlier
        return summary, mask_thresh_min

    # ---------------------- Low-cost approximate ablation -----------------------------
    def approx_ablation_scores(self, md_layers):
        """
        Compute inexpensive per-unit ablation deltas (loss_after_zeroing - baseline_loss).
        Returns dict unique_layer_name -> delta array (shape = units, dtype=float32).
        Only tests up to config.ablation_max_neurons per layer and config.ablation_max_batches from val_ds.
        """
        self.log.debug('Running approx_ablation_scores()')
        max_batches = max(1, int(self.config.ablation_max_batches))
        max_neurons = max(1, int(self.config.ablation_max_neurons))
        loss_fn = keras.losses.get(keras.losses.Huber())
    
        # collect small set of validation batches
        batches = []
        if isinstance(self.val_ds, tuple) and isinstance(self.val_ds[0], np.ndarray):
            X_val, y_val = self.val_ds
            vb_iter = numpy_batch_iter(X_val, y_val, CONFIG['batch_size'])
        else:
            vb_iter = iter(self.val_ds)
    
        for _ in range(max_batches):
            try:
                xb, yb = next(vb_iter)
            except StopIteration:
                break
            batches.append((tf.convert_to_tensor(xb), tf.convert_to_tensor(yb)))
        if len(batches) == 0:
            # fallback: try a single evaluation call to get baseline
            self.log.warning('No validation batches available for ablation scoring; returning zeros')
            return {f"{l.name}_{hex(id(l))}": np.zeros((l.units,), dtype=np.float32) for l in md_layers}
    
        # baseline averaged loss across selected batches
        baseline_losses = []
        for xb, yb in batches:
            preds = self.model(xb, training=False)
            baseline_losses.append(float(loss_fn(yb, preds).numpy()))
        baseline = float(np.mean(baseline_losses))
    
        ablation_scores = {}
        for layer in md_layers:
            unique_name = f"{layer.name}_{hex(id(layer))}"
            units = int(layer.units)
    
            # get alive indices
            try:
                alive_mask = (layer.mask.numpy() > 0.5)
            except Exception:
                alive_mask = np.ones((units,), dtype=bool)
            alive_idx = np.where(alive_mask)[0]
            if alive_idx.size == 0:
                ablation_scores[unique_name] = np.zeros((units,), dtype=np.float32)
                continue
    
            # choose subset deterministically: lowest grad-importance first if available, else first N
            prior_unit_imp = getattr(layer, '_last_unit_score', None)
            if prior_unit_imp is not None and prior_unit_imp.size == units:
                # pick the units with smallest prior importance (most likely prunable) for test coverage
                idx_order = np.argsort(prior_unit_imp)  # ascending
                test_idx = [i for i in idx_order if i in alive_idx][:max_neurons]
            else:
                # deterministic subsample: take first max_neurons alive indices
                test_idx = alive_idx[:max_neurons]
    
            delta = np.zeros((units,), dtype=np.float32)
    
            # perform cheap ablation per tested unit
            for u in test_idx:
                orig_mask = layer.mask.numpy()
                tmp_mask = orig_mask.copy()
                tmp_mask[u] = 0.0
                layer.mask.assign(tmp_mask)
                # evaluate same small batch set
                losses = []
                for xb, yb in batches:
                    preds = self.model(xb, training=False)
                    losses.append(float(loss_fn(yb, preds).numpy()))
                loss_after = float(np.mean(losses))
                delta[u] = loss_after - baseline
                # restore mask
                layer.mask.assign(orig_mask)
    
            # for untested units, set safe fallback to median of tested (prevents accidental pruning of untested units)
            if len(test_idx) < units and len(test_idx) > 0:
                safe_val = float(np.median(delta[test_idx]))
                for u in range(units):
                    if u not in test_idx:
                        delta[u] = safe_val
            ablation_scores[unique_name] = delta
    
        return ablation_scores

        
    # ---------------------- utility computation -----------------------------
    def compute_utilities(self, md_layers) -> Dict[str, np.ndarray]:
        """
        Compute per-layer per-unit utilities combining:
          - fisher (approx second-order importance)
          - grad-based importance: mean(abs(act * dL/dact)) computed per-sample
          - weight-grad importance: mean(abs(kernel * dkernel)) per-connection
          - ablation scores (from self.approx_ablation_scores)
        Uses vmapped per-example gradients when possible and monkeypatch capture fallback otherwise.
        Returns utilities: dict unique_layer_name -> per-unit utility (EMA-smoothed).
        """

        # Adaptive signal weighting (reliability-based)
        def adaptive_weights(G, F, A, eps=1e-8):
            """Compute dynamic α, β, γ based on signal reliability (1 / variance)."""
            sigs = [G, F, A]
            reliabilities = []
            for sig in sigs:
                if sig is None or np.all(sig == 0):
                    reliabilities.append(0.0)
                else:
                    reliabilities.append(1.0 / (np.var(sig) + eps))
            reliabilities = np.array(reliabilities, dtype=np.float32)
            total = np.sum(reliabilities) + eps
            return reliabilities / total

        # normalization helper
        def _norm_arr(x):
            x = np.asarray(x, dtype=np.float32)
            if x.size == 0:
                return x
            x = x - np.median(x)
            s = np.std(x) + 1e-9
            return (x / s)            
        
        self.log.debug('Computing quality-aware utilities (grad-based + fisher)')
    
        # 1) activation stats
        acts = capture_activations(self.model, self.val_ds, masked_layers=md_layers)
        entropies = {name: compute_activation_entropy(a) for name, a in acts.items()}
    
        # 2) fisher
        fisher = approx_fisher_per_sample(
            self.model, self.val_ds, keras.losses.Huber(),
            max_samples_per_batch=int(CONFIG.get('max_fisher_samples_per_batch', 4))
        )
    
        utilities: Dict[str, np.ndarray] = {}
    
        # gradient-based parts
        use_grad = getattr(self.config, 'use_grad_importance', True)
        use_wgrad = getattr(self.config, 'use_weight_grad', True)
        max_batches = max(1, int(getattr(self.config, 'max_grad_samples', 1)))
    
        # get val iterator
        if isinstance(self.val_ds, tuple) and isinstance(self.val_ds[0], np.ndarray):
            X_val, y_val = self.val_ds
            val_iter = numpy_batch_iter(X_val, y_val, CONFIG.get('batch_size', 32))
        else:
            val_iter = iter(self.val_ds)
    
        # accumulators
        grad_unit_accum = {f"{l.name}_{hex(id(l))}": None for l in md_layers}
        grad_unit_count = {k: 0 for k in grad_unit_accum}
        grad_weight_accum = {k: None for k in grad_unit_accum}
    
        # build submodels where possible and mark failures
        layer_submodels = {}
        failed_layers = []
        for layer in md_layers:
            unique_name = f"{layer.name}_{hex(id(layer))}"
            try:
                layer_submodels[unique_name] = keras.Model(self.model.inputs, [layer.output, self.model.output])
            except Exception as e:
                self.log.debug(f'Failed to build submodel for {unique_name}: {e}')
                layer_submodels[unique_name] = None
                failed_layers.append(layer)
    
        # per-example prefix size
        max_samples_per_batch = int(min(max(1, int(CONFIG.get('max_grad_samples_per_batch', 4))), 8))
    
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
    
            xb_tf = tf.convert_to_tensor(xb, dtype=tf.float32)
            yb_tf = tf.convert_to_tensor(yb, dtype=tf.float32)
            B = int(tf.shape(xb_tf)[0])
            if B == 0:
                continue
            n_samples = int(min(B, max_samples_per_batch))
            x_small = xb_tf[:n_samples]
            y_small = yb_tf[:n_samples]
    
            # for each layer try fast vmapped path when submodel exists else fallback to monkeypatch per-batch
            # Fast per-layer vmapped
            for layer in md_layers:
                unique_name = f"{layer.name}_{hex(id(layer))}"
                subm = layer_submodels.get(unique_name)
                if subm is not None and use_grad:
                    # per-example function: returns (units_vec, kernel_grad_tensor_or_empty)
                    def per_example_fn(xi, yi, _subm=subm, _layer=layer):
                        xi_b = tf.expand_dims(xi, 0)
                        yi_b = tf.expand_dims(yi, 0)
                        with tf.GradientTape(persistent=True) as tape:
                            act_b, pred_b = _subm(xi_b, training=False)
                            tape.watch(act_b)
                            loss_b = tf.reshape(loss_fn(yi_b, pred_b), [])
                        g_act = tape.gradient(loss_b, act_b)
                        if g_act is None:
                            units = tf.shape(act_b)[-1]
                            unit_mag = tf.zeros((units,), dtype=tf.float32)
                        else:
                            g0 = g_act[0]
                            a0 = act_b[0]
                            prod = tf.abs(a0 * g0)
                            if tf.rank(prod) == 1:
                                unit_mag = prod
                            else:
                                sq = tf.square(prod)
                                reduce_axes = tf.range(tf.rank(prod) - 1)
                                unit_mag = tf.sqrt(tf.reduce_sum(sq, axis=reduce_axes))
                        # compute kernel grad if possible
                        if use_wgrad and hasattr(_layer, 'kernel') and _layer.kernel is not None:
                            gk = tape.gradient(loss_b, _layer.kernel)
                            if gk is None:
                                gk = tf.zeros_like(_layer.kernel)
                        else:
                            gk = tf.constant([], dtype=tf.float32)
                        del tape
                        return unit_mag, gk
    
                    # run vectorized_map safely
                    try:
                        mapped = tf.vectorized_map(lambda elems: per_example_fn(elems[0], elems[1]), (x_small, y_small))
                        per_sample_unit = mapped[0]  # (n_samples, units)
                        per_sample_k = mapped[1]     # (n_samples, *kernel_shape) or empty
                    except Exception:
                        # fallback python loop
                        per_unit_list = []
                        per_k_list = []
                        for i in range(n_samples):
                            u, kgrad = per_example_fn(x_small[i], y_small[i])
                            per_unit_list.append(u)
                            per_k_list.append(kgrad)
                        per_sample_unit = tf.stack(per_unit_list, axis=0)
                        per_sample_k = tf.stack(per_k_list, axis=0)
    
                    # aggregate
                    per_unit_avg = tf.reduce_mean(per_sample_unit, axis=0).numpy()
                    if grad_unit_accum[unique_name] is None:
                        grad_unit_accum[unique_name] = per_unit_avg
                    else:
                        grad_unit_accum[unique_name] += per_unit_avg
                    grad_unit_count[unique_name] += 1
    
                    if use_wgrad and hasattr(layer, 'kernel') and layer.kernel is not None:
                        avg_grad_k = tf.reduce_mean(per_sample_k, axis=0).numpy()
                        try:
                            kernel_np = layer.kernel.numpy()
                            w_imp = np.abs(kernel_np * avg_grad_k)
                        except Exception:
                            w_imp = np.abs(avg_grad_k)
                        if grad_weight_accum[unique_name] is None:
                            grad_weight_accum[unique_name] = w_imp
                        else:
                            grad_weight_accum[unique_name] += w_imp
    
            # Now handle failed layers via monkeypatch-instrumented single-tape forward
            if failed_layers and use_grad:
                # filter which failed layers we actually care about for this batch
                layers_to_patch = failed_layers
                prev = bool(getattr(tf.config, "functions_run_eagerly", lambda: False)())
                tf.config.run_functions_eagerly(True)
                try:
                    with monkeypatch_capture_layers(layers_to_patch, record_tensors=True, force_eager=True) as records:
                        with tf.GradientTape(persistent=True) as tape:
                            # run forward once
                            preds = self.model(x_small, training=False)
                            loss_vec = tf.reshape(loss_fn(y_small, preds), [-1])  # (n_samples,)
                            outs = {}
                            for l in layers_to_patch:
                                key = f"{l.name}_{hex(id(l))}"
                                rec = records.get(key, [])
                                if not rec:
                                    continue
                                out_t = _first_tensor_from_nested(rec[0])
                                if out_t is None:
                                    continue
                                outs[key] = out_t
                                tape.watch(out_t)
                        # now compute jacobians per captured output
                        for key, out_t in outs.items():
                            # jac shape: (n_samples, n_samples, *out_tail)
                            jac = tape.jacobian(loss_vec, out_t)
                            if jac is None:
                                continue
                            n = n_samples
                            per_sample_flat, tail_units = _gather_diag_from_jac(jac, n, out_t)
                            # attempt to reshape into (n, time_prod, units) as before
                            try:
                                units = tf.shape(out_t)[-1] if tf.rank(out_t) > 0 else 1
                                P = tf.shape(per_sample_flat)[1]
                                time_prod = tf.maximum(P // units, 1)
                                per_sample_resh = tf.reshape(per_sample_flat, (n, time_prod, units))
                                sq = tf.square(per_sample_resh)
                                unit_sq = tf.reduce_sum(sq, axis=1)  # (n, units)
                            except Exception:
                                unit_sq = tf.square(per_sample_flat)
                            batch_sum = tf.reduce_sum(unit_sq, axis=0)
                            try:
                                # convert to numpy and add
                                if grad_unit_accum.get(key) is None:
                                    grad_unit_accum[key] = batch_sum.numpy()
                                else:
                                    grad_unit_accum[key] += batch_sum.numpy()
                            except Exception:
                                if grad_unit_accum.get(key) is None:
                                    grad_unit_accum[key] = np.array(batch_sum)
                                else:
                                    grad_unit_accum[key] += np.array(batch_sum)
                            grad_unit_count[key] += 1
    
                            # weight-grad via jacobian isn't trivial for monkeypatch path,
                            # we skip per-weight grads here (caller can rely on vmapped path or layer.kernel grads)
                finally:
                    tf.config.run_functions_eagerly(prev)

        try:
            ablation = self.approx_ablation_scores(md_layers)
        except Exception as e:
            self.log.debug(f'Ablation scoring failed: {e}')
            ablation = {}

        # finalize accumulators and assemble utilities
        for layer in md_layers:
            unique_name = f"{layer.name}_{hex(id(layer))}"
    
            if grad_unit_accum.get(unique_name) is not None and grad_unit_count.get(unique_name, 0) > 0:
                gunit = grad_unit_accum[unique_name] / float(max(1, grad_unit_count[unique_name]))
            else:
                gunit = np.zeros((getattr(layer, 'units', 0),), dtype=np.float32)
    
            if grad_weight_accum.get(unique_name) is not None:
                gweight = grad_weight_accum[unique_name] / float(max(1, grad_unit_count[unique_name]))
            else:
                try:
                    gweight = np.zeros_like(layer.kernel.numpy())
                except Exception:
                    gweight = np.array([])
    
            gunit_n = _norm_arr(gunit)
            
            f = fisher.get(unique_name, np.zeros_like(gunit))
            f_n = _norm_arr(f)

            a = ablation.get(unique_name, None)
            if a is None:
                a_n = np.zeros_like(gunit_n)
            else:
                arra = np.asarray(a, dtype=np.float32)
                arra = arra - np.median(arra)
                s3 = np.std(arra) + 1e-9
                a_n = (arra / s3)             

            setattr(layer, '_last_grad_weight_importance', gweight)
            setattr(layer, '_last_grad_unit_importance', gunit)
            setattr(layer, '_last_fisher', f)
            
            # Get dynamic alpha, beta, gamma per layer
            alpha, beta, gamma = adaptive_weights(gunit, f, a)
    
            # Combine normalized signals
            unit_score = alpha * gunit_n + beta * f_n + gamma * a_n              
    
            setattr(layer, '_last_adaptive_weights', (alpha, beta, gamma))
            setattr(layer, '_last_unit_score', unit_score.astype(np.float32))
            
            # ---------------------------------------------------------------
            # EMA smoothing across cycles
            # ---------------------------------------------------------------
            decay = float(self.config.ema_decay)
            prev = getattr(layer, '_ema_unit_score', None)
            if prev is None:
                ema = unit_score.copy()
            else:
                ema = decay * np.asarray(prev, dtype=np.float32) + (1.0 - decay) * unit_score
            setattr(layer, '_ema_unit_score', ema.astype(np.float32))
    
            # Final per-layer utility exposed to pruning logic
            utilities[unique_name] = ema.astype(np.float32)
    
            # Debug logging
            self.log.debug(
                f'Adaptive utilities for {unique_name}: '
                f'α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, '
                f'units={unit_score.size}'
                
            )
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
            mask_threshold = min(mask_thresh_min, 0.7)
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

    def _apply_prune_scenario(self, layer, stored_prune_info, scenario: str):
        """
        Apply pruning scenario. If 'layer' is provided, expects stored_prune_info to contain
        only that layer's entry. If layer is None, we will locate the layer object for each
        stored_prune_info entry.
        """
        md_layers = get_masked_dense_layers(self.model)
    
        for unique_name, info in stored_prune_info.items():
            # locate layer object if not given
            if layer is None:
                layer_obj = next((l for l in md_layers if f"{l.name}_{hex(id(l))}" == unique_name), None)
            else:
                # caller provided a specific layer object to operate on
                # ensure it's the same unique_name (caller responsibility)
                layer_obj = layer
    
            if layer_obj is None:
                self.log.debug(f'No layer object found for {unique_name}; skipping')
                continue
    
            if scenario == 'neurons':
                if info.get('prune_idx', None) is not None and getattr(info['prune_idx'], 'size', 0) > 0:
                    self.log.debug(f'Applying neuron prune to {unique_name}: n={len(info["prune_idx"])}')
                    layer_obj.prune_units(info['prune_idx'])
            elif scenario == 'connections':
                if info.get('conn_low_pairs', None) is not None and getattr(info['conn_low_pairs'], 'size', 0) > 0:
                    self.log.debug(f'Applying connection prune to {unique_name}: n_pairs={len(info["conn_low_pairs"])}')
                    layer_obj.prune_connections(info['conn_low_pairs'])
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
        self.log.debug(f'[LAYERWISE] initial baseline_val={baseline_val:.6g}')

        md_layers = get_masked_dense_layers(self.model)
        for unique_name, info in items:
            layer = next((l for l in md_layers if f"{l.name}_{hex(id(l))}" == unique_name), None)
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
            self._apply_prune_scenario(layer, {unique_name: info}, 'neurons')
            metrics_neu = self._evaluate_val()
            v_neu = float(metrics_neu[0])
            self.log.debug(f'[LAYERWISE] {unique_name} neuron-only val_loss={v_neu:.6g}')

            # revert to baseline before testing connections
            self.model.set_weights(current_baseline)

            # Test connection-only pruning
            self._apply_prune_scenario(layer, {unique_name: info}, 'connections')
            metrics_conn = self._evaluate_val()
            v_conn = float(metrics_conn[0])
            self.log.debug(f'[LAYERWISE] {unique_name} connection-only val_loss={v_conn:.6g}')

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
            self._apply_prune_scenario(layer, {unique_name: info}, chosen_mode)

            if chosen_val <= baseline_val * (1.0 + rel_tol):
                # accept this layer's prune and update baseline to include it
                accepted.append(unique_name)
                # current model already has chosen prune applied (we applied it earlier)
                # Run immediate short per-layer finetune to recover/solidify gains
                try:
                    # get the actual layer object (we already have `layer` in outer scope)
                    # but if not available, look it up
                    target_layer = layer
                    finetune_steps = int(getattr(self.config, 'layer_finetune_steps', 0))
                    finetune_bs = getattr(self.config, 'layer_finetune_batch_size', None)
                    lr_mul = float(getattr(self.config, 'layer_finetune_lr_mul', 1.0))
                    if finetune_steps > 0:
                        self.log.debug(f'[LAYERWISE] Running per-layer finetune for {unique_name}: steps={finetune_steps}, bs={finetune_bs}, lr_mul={lr_mul}')
                        val_after_ft = self._finetune_layer(target_layer, steps=finetune_steps, batch_size=finetune_bs, lr_mul=lr_mul)
                        # if finetune improves chosen_val, use it as the canonical chosen_val
                        if val_after_ft <= chosen_val:
                            chosen_val = val_after_ft
                            self.log.info(f'[LAYERWISE] Per-layer finetune improved {unique_name}: new_val={chosen_val:.6g}')
                        else:
                            # keep chosen_val as is (finetune didn't help)
                            self.log.debug(f'[LAYERWISE] Per-layer finetune did not improve {unique_name}: pre={chosen_val:.6g} post={val_after_ft:.6g}')
                except Exception as e:
                    self.log.warning(f'[LAYERWISE] per-layer finetune failed for {unique_name}: {e}')

                # update baseline snapshot and baseline_val to include the accepted prune + any finetune
                current_baseline = self.model.get_weights()
                baseline_val = chosen_val
                self.log.info(f'[LAYERWISE] ACCEPTED {unique_name} ({chosen_mode}): rel_delta={delta:.6g} -> new baseline_val={baseline_val:.6g}')

            else:
                # reject this layer's prune; revert
                rejected.append(unique_name)
                self.model.set_weights(current_baseline)
                self.log.info(f'[LAYERWISE] REJECTED {unique_name} ({chosen_mode}): rel_delta={delta:.6g}')

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
            self._apply_prune_scenario(None, stored_prune_info, scen)
            metrics = self._evaluate_val()
            scenario_val_loss = float(metrics[0])
            scenario_results[scen] = {'val_loss': scenario_val_loss}
            self.log.info(f'[PRUNE-SEARCH] scenario={scen}, val_loss={scenario_val_loss:.6g}')

        best_scenario = min(scenario_results.keys(), key=lambda k: scenario_results[k]['val_loss'])
        pruned_val_loss = float(scenario_results[best_scenario]['val_loss'])

        # ensure the model is left pruned according to best scenario
        self.model.set_weights(baseline_weights)
        self._apply_prune_scenario(None, stored_prune_info, best_scenario)

        return best_scenario, pruned_val_loss, baseline_weights

    # ---------------------- global short finetune ---------------------------------
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

    # ---------------------- per-layer short finetune ---------------------------------
    def _finetune_layer(self, layer: keras.layers.Layer, steps: int = 2, batch_size: Optional[int] = None, lr_mul: float = 1.0):
        """
        Short finetune that updates ONLY variables belonging to `layer`.
        - steps: number of train steps (mini-batches)
        - batch_size: batch size for train batches (None => config.finetune_batch_size or global CONFIG)
        - lr_mul: multiplier applied to gradients (conservative when <1.0)
        Returns validation loss after finetune.
        """
        if self.train_ds is None:
            self.log.debug(f'No training data available for per-layer finetune on {layer.name}; skipping')
            return float(self._evaluate_val()[0])
    
        # prepare batch iterator
        bs = int(batch_size or self.config.layer_finetune_batch_size or self.config.finetune_batch_size or CONFIG['batch_size'])
        if isinstance(self.train_ds, tuple) and isinstance(self.train_ds[0], np.ndarray):
            train_iter = numpy_batch_iter(self.train_ds[0], self.train_ds[1], bs)
        else:
            train_iter = iter(self.train_ds)
    
        # loss and optimizer
        loss_fn = keras.losses.get(keras.losses.Huber())
        opt = getattr(self.model, 'optimizer', None)
        if opt is None:
            # fallback optimizer (no stateful accumulators for main opt)
            opt = keras.optimizers.AdamW(learning_rate=1e-4)
    
        vars_to_update = [v for v in layer.trainable_variables if v.trainable]
        if not vars_to_update:
            self.log.debug(f'Layer {layer.name} has no trainable variables; skipping per-layer finetune')
            return float(self._evaluate_val()[0])
    
        steps_done = 0
        for xb, yb in train_iter:
            xb_tf = tf.convert_to_tensor(xb)
            yb_tf = tf.convert_to_tensor(yb)
            with tf.GradientTape() as tape:
                preds = self.model(xb_tf, training=True)
                loss = loss_fn(yb_tf, preds)
            grads = tape.gradient(loss, vars_to_update)
            # filter None grads and apply lr multiplier if requested
            grads_and_vars = []
            for g, v in zip(grads, vars_to_update):
                if g is None:
                    continue
                g_scaled = g * float(lr_mul) if (lr_mul != 1.0) else g
                grads_and_vars.append((g_scaled, v))
            if grads_and_vars:
                opt.apply_gradients(grads_and_vars)
            steps_done += 1
            if steps_done >= steps:
                break
    
        # evaluate and return validation loss
        metrics = self._evaluate_val()
        val_loss = float(metrics[0])
        self.log.info(f'[LAYER-FINETUNE] layer={layer.name} steps={steps_done} val_loss={val_loss:.6g}')
        return val_loss
        

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

            # neuron regrowth (deterministic)
            n_prune = info['n_prune']
            if n_prune <= 0 or self.config.regrow_fraction <= 0:
                n_regrow = 0
            else:
                n_regrow = int(np.random.binomial(n_prune, self.config.regrow_fraction))
    
            if n_regrow > 0:
                dead_indices = np.where(layer.mask.numpy() == 0)[0]
                # deterministic selection: pick top-desirability using last_unit_score or fallback
                if dead_indices.size == 0:
                    regrow_idx = np.array([], dtype=int)
                else:
                    # compute desirability for dead neurons: prefer those with higher downstream desirability (utilities)
                    util = utilities.get(unique_name, None)
                    if util is None:
                        # deterministic default: smallest indices
                        cand_order = dead_indices
                    else:
                        # prefer regrowing positions that, if active, would have high utility (descending)
                        # For dead neurons util might be filled with safe median; still deterministic
                        cand_scores = util[dead_indices]
                        order = np.argsort(-cand_scores)  # descending
                        cand_order = dead_indices[order]
                    k = min(n_regrow, len(cand_order))
                    regrow_idx = np.array(cand_order[:k], dtype=int)
    
                if regrow_idx.size:
                    # deterministic reinit based on alive columns statistics + seeded noise
                    k_arr = layer.kernel.numpy()
                    b_arr = layer.bias.numpy()
                    conn_age = layer.conn_age.numpy()
                    alive_cols = np.where(layer.mask.numpy() > 0.5)[0]
                    if alive_cols.size:
                        base = np.mean(k_arr[:, alive_cols], axis=1)
                        col_std = np.std(k_arr[:, alive_cols], axis=1)
                        # avoid zero std
                        col_std = np.where(col_std <= 1e-8, 0.02, col_std)
                    else:
                        base = np.zeros((k_arr.shape[0],), dtype=np.float32)
                        col_std = np.ones_like(base) * 0.02
    
                    for idx in regrow_idx:
                        seed = int(self.config.deterministic_regrow_seed + idx)
                        rng = np.random.RandomState(seed)
                        noise = rng.normal(loc=0.0, scale=self.config.regrow_reinit_std, size=base.shape) * col_std
                        k_arr[:, idx] = base + noise
                        b_arr[idx] = 0.0
                        conn_age[:, idx] = 0.0
    
                    # assign back
                    layer.kernel.assign(k_arr)
                    layer.bias.assign(b_arr)
                    layer.conn_age.assign(conn_age)
                    # make mask live
                    m = layer.mask.numpy()
                    m[regrow_idx] = 1.0
                    layer.mask.assign(m)
                    self.log.info(f'[REGROW] {layer.name}: deterministically regrew {len(regrow_idx)} neurons')
    
            # connection regrowth (deterministic)
            regrew_conn_count = 0
            n_rewire = info['n_rewire']
            if n_rewire > 0:
                W = layer.kernel.numpy()
                out_util = utilities.get(unique_name, np.zeros((layer.units,)))
                if out_util.size == layer.units:
                    out_u = out_util.copy().astype(np.float32)
                    # normalize [0,1]
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
                    # deterministic tie-breaking: add tiny deterministic jitter based on index
                    jitter = np.array([((p * 1315423911) % 1000) for p in range(len(desirability))], dtype=np.float32) * 1e-9
                    desirability = desirability + jitter
                    k = min(n_rewire, len(dead_pairs))
                    if k > 0:
                        choose_idx = np.argpartition(-desirability, k - 1)[:k]
                        regrow_pairs = dead_pairs[choose_idx]
                        # deterministic weight init using alive columns stats
                        k_arr = layer.kernel.numpy()
                        conn_age = layer.conn_age.numpy()
                        for (i, j) in regrow_pairs:
                            seed = int(self.config.deterministic_regrow_seed + i * 31 + j * 97)
                            rng = np.random.RandomState(seed)
                            # init with tiny noise around existing row/col statistics
                            row_mean = np.mean(k_arr[i, :][layer.mask.numpy() > 0.5]) if np.any(layer.mask.numpy() > 0.5) else 0.0
                            init_val = row_mean + rng.normal(scale=self.config.regrow_reinit_std)
                            k_arr[i, j] = init_val
                            conn_age[i, j] = 0.0
                        # layer.conn_mask.assign(layer.conn_mask.numpy())  # ensure shape consistent
                        layer.conn_mask.assign(np.where(layer.conn_mask.numpy() < 0.5, layer.conn_mask.numpy(), layer.conn_mask.numpy()))
                        # apply changes
                        layer.regrow_connections(regrow_pairs)
                        regrew_conn_count = len(regrow_pairs)
                        self.log.info(f'[REGROW] {layer.name}: deterministically regrew {regrew_conn_count} connections')


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
            gap = float(abs(val_loss - train_loss))
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
def _snapshot_layer_masks(model):
    """Return a dict of layer_name -> dict of mask arrays for masked layers."""
    masks = {}
    for layer in model.layers:
        # adjust checks to match your masked-dense implementation
        if hasattr(layer, 'mask') or hasattr(layer, 'conn_mask'):
            masks[layer.name] = {}
            if hasattr(layer, 'mask'):
                try:
                    masks[layer.name]['mask'] = layer.mask.numpy()
                except Exception:
                    masks[layer.name]['mask'] = np.array(layer.mask)  # fallback
            if hasattr(layer, 'conn_mask'):
                try:
                    masks[layer.name]['conn_mask'] = layer.conn_mask.numpy()
                except Exception:
                    masks[layer.name]['conn_mask'] = np.array(layer.conn_mask)
            if hasattr(layer, 'conn_age'):
                try:
                    masks[layer.name]['conn_age'] = layer.conn_age.numpy()
                except Exception:
                    masks[layer.name]['conn_age'] = np.array(layer.conn_age)
    return masks

def _restore_layer_masks(model, masks_snapshot):
    """Restore mask arrays back into layers (use assign if tf.Variable)."""
    for layer in model.layers:
        if layer.name not in masks_snapshot:
            continue
        snap = masks_snapshot[layer.name]
        if 'mask' in snap and hasattr(layer, 'mask'):
            try:
                if isinstance(layer.mask, tf.Variable):
                    layer.mask.assign(snap['mask'])
                else:
                    layer.mask = snap['mask']
            except Exception:
                layer.mask = snap['mask']
        if 'conn_mask' in snap and hasattr(layer, 'conn_mask'):
            try:
                if isinstance(layer.conn_mask, tf.Variable):
                    layer.conn_mask.assign(snap['conn_mask'])
                else:
                    layer.conn_mask = snap['conn_mask']
            except Exception:
                layer.conn_mask = snap['conn_mask']
        if 'conn_age' in snap and hasattr(layer, 'conn_age'):
            try:
                if isinstance(layer.conn_age, tf.Variable):
                    layer.conn_age.assign(snap['conn_age'])
                else:
                    layer.conn_age = snap['conn_age']
            except Exception:
                layer.conn_age = snap['conn_age']


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
# Transformer Block
# -----------------------------
@keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, **kwargs):
        # Accept name/trainable/dtype/etc via kwargs and forward to base Layer
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)

        # build sublayers in __init__ is fine (they will be built later on first call)
        self.att = MaskedMultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads, name='masked_mha')
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
        self.ffn_dense1 = MaskedDense(self.ff_dim, activation='gelu', name='ffn_dense1')
        self.ffn_dense2 = MaskedDense(self.d_model, activation=None, name='ffn_dense2')

    def call(self, x, training=None):
        # accept training as optional kwarg (Keras may call with training=True/False)
        attn_out = self.att(x, x, x, training=training)
        x = self.norm1(x + attn_out)
        seq_len = tf.shape(x)[1]
        # flatten per-token to apply MaskedDense which expects 2D
        x_flat = tf.reshape(x, (-1, tf.shape(x)[-1]))
        pre, h = self.ffn_dense1(x_flat, return_pre_activation=True)
        h2 = self.ffn_dense2(h)
        h2 = tf.reshape(h2, (-1, seq_len, self.d_model))
        x = self.norm2(x + h2)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'd_model': int(self.d_model),
            'num_heads': int(self.num_heads),
            'ff_dim': int(self.ff_dim),
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        # Keras will call this during deserialization
        return cls(**config)


# -----------------------------
# Custom Masked Multihead Attention layer
# -----------------------------
@keras.utils.register_keras_serializable(package="Custom")
class MaskedMultiHeadAttention(keras.layers.Layer):
    """
    Multi-head attention implemented with MaskedDense linear projections so that
    q/k/v and output projections are prunable/regrowable via MaskedDense.mask/conn_mask.
    Keeps API similar to keras.layers.MultiHeadAttention: call(query, key, value, training=None, mask=None)
    """

    def __init__(self, d_model, num_heads, dropout=0.0, name="masked_mha", **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        if self.d_model % max(1, self.num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = self.d_model // max(1, self.num_heads)
        self.dropout = float(dropout)

        # Use MaskedDense for projections so they are part of your pruning/regrow system
        self.q_proj = MaskedDense(self.d_model, activation=None, name=f"{name}_q_proj")
        self.k_proj = MaskedDense(self.d_model, activation=None, name=f"{name}_k_proj")
        self.v_proj = MaskedDense(self.d_model, activation=None, name=f"{name}_v_proj")
        self.out_proj = MaskedDense(self.d_model, activation=None, name=f"{name}_out_proj")
        self._softmax = tf.keras.layers.Softmax(axis=-1)
        if self.dropout > 0.0:
            self._drop = tf.keras.layers.Dropout(self.dropout)
        else:
            self._drop = None

    def build(self, input_shape):
        # MaskedDense will initialize weights during its own build when called the first time
        super().build(input_shape)

    def _split_heads(self, x):
        # x: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)
        b, seq = tf.shape(x)[0], tf.shape(x)[1]
        x = tf.reshape(x, (b, seq, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def _combine_heads(self, x):
        # x: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        b = tf.shape(x)[0]; seq = tf.shape(x)[1]
        return tf.reshape(x, (b, seq, self.d_model))

    def call(self, query, key, value, training=None, mask=None):
        """
        query/key/value: (batch, seq_len, d_model)
        mask: optional attention mask compatible with scores shape (batch, num_heads, seq_q, seq_k)
        returns: (batch, seq_len_q, d_model)
        """
        # project
        q = self.q_proj(query)   # (batch, seq_q, d_model)
        k = self.k_proj(key)     # (batch, seq_k, d_model)
        v = self.v_proj(value)   # (batch, seq_k, d_model)

        # split heads
        qh = self._split_heads(q)  # (b, nh, seq_q, head_dim)
        kh = self._split_heads(k)  # (b, nh, seq_k, head_dim)
        vh = self._split_heads(v)  # (b, nh, seq_k, head_dim)

        # scaled dot-product
        scale = tf.cast(tf.math.sqrt(tf.cast(self.head_dim, tf.float32)), tf.float32)
        # scores: (b, nh, seq_q, seq_k)
        scores = tf.matmul(qh, kh, transpose_b=True) / (scale + 1e-12)

        if mask is not None:
            # Assume mask is broadcastable to (b, nh, seq_q, seq_k)
            scores = tf.where(tf.cast(mask, dtype=tf.bool), scores, tf.fill(tf.shape(scores), -1e9))

        attn = self._softmax(scores)  # (b, nh, seq_q, seq_k)
        if self._drop is not None:
            attn = self._drop(attn, training=training)

        # attention output
        context = tf.matmul(attn, vh)  # (b, nh, seq_q, head_dim)
        combined = self._combine_heads(context)  # (b, seq_q, d_model)

        out = self.out_proj(combined)  # (b, seq_q, d_model)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'d_model': int(self.d_model), 'num_heads': int(self.num_heads), 'dropout': float(self.dropout)})
        return cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)
        

# -----------------------------
# Custom Masked Dense layer
# -----------------------------
@keras.utils.register_keras_serializable(package="Custom")
class MaskedDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        # accept name/trainable/dtype/etc and forward
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        w_init = tf.random_normal_initializer(stddev=0.02)
        self.kernel = self.add_weight(
            name='kernel', shape=(last_dim, self.units),
            initializer=w_init, trainable=True
        )
        self.bias = self.add_weight(
            name='bias', shape=(self.units,), initializer='zeros', trainable=True
        )
        # masks and age tracked as non-trainable variables (will be saved in weights)
        self.mask = self.add_weight(
            name='mask', shape=(self.units,), initializer='ones', trainable=False
        )
        self.conn_mask = self.add_weight(
            name='conn_mask', shape=(last_dim, self.units), initializer='ones', trainable=False
        )
        self.conn_age = self.add_weight(
            name='conn_age', shape=(last_dim, self.units), initializer='zeros', trainable=False
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
        cfg = super().get_config()
        # serialize activation via keras serializer
        cfg.update({
            'units': int(self.units),
            'activation': keras.activations.serialize(self.activation)
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        # activation will be serialized; ensure it's passed correctly
        return cls(**config)


# -----------------------------
# Model builder
# -----------------------------
def build_transformer_model(seq_len, d_model, num_layers, num_heads, ff_dim):
    inp = keras.layers.Input(shape=(seq_len, 1))
    x = norm(inp)
    # x = keras.layers.Dense(d_model)(x)
    x = MaskedDense(d_model, activation=None, name='proj_dense')(x)
    pe = positional_encoding(seq_len, d_model)
    x = x + pe
    for i in range(num_layers):
        x = TransformerBlock(d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, name=f'transformer_block_{i}')(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    out = keras.layers.Dense(1, name='head')(x)
    model = keras.Model(inp, out)
    return model


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

# ------------------- monkeypatch capture manager --------------------------------
@contextmanager
def monkeypatch_capture_layers(layers: Iterable[keras.layers.Layer],
                               *, record_tensors: bool = True, force_eager: bool = True):
    """
    Temporarily monkeypatch layer.call to capture raw outputs produced in a forward pass.
    Returns a dict unique_name -> list-of-captured-outputs (in call order).
    """
    originals = {}
    records = {}

    for l in layers:
        originals[l] = getattr(l, 'call')
        records[f"{l.name}_{hex(id(l))}"] = []

    # create patched closures
    for l in layers:
        orig = originals[l]
        uniq = f"{l.name}_{hex(id(l))}"

        def _make_patched(_orig, _uniq):
            @functools.wraps(_orig)
            def patched(*args, **kwargs):
                out = _orig(*args, **kwargs)
                # best-effort record
                try:
                    if record_tensors:
                        records[_uniq].append(out)
                    else:
                        # try numpy copy for safety if requested
                        if isinstance(out, (list, tuple)):
                            records[_uniq].append([o.numpy() for o in out])
                        else:
                            records[_uniq].append(out.numpy())
                except Exception:
                    # if `.numpy()` fails or nested structures are tricky, store raw
                    records[_uniq].append(out)
                return out
            return patched

        patched_fn = _make_patched(orig, uniq)
        setattr(l, 'call', patched_fn)

    # optionally force eager to ensure subclassed layers are executed concretely
    prev_eager = None
    if force_eager:
        prev_eager = bool(getattr(tf.config, "functions_run_eagerly", lambda: False)())
        tf.config.run_functions_eagerly(True)

    try:
        yield records
    finally:
        # restore originals
        for l, orig in originals.items():
            try:
                setattr(l, 'call', orig)
            except Exception:
                pass
        if force_eager and prev_eager is not None:
            tf.config.run_functions_eagerly(prev_eager)


def _first_tensor_from_nested(x):
    """
    Return the first tf.Tensor found inside a nested structure `x` (list/tuple/dict/etc),
    or None if none found.
    """
    flat = tfnest.flatten(x)
    for el in flat:
        if isinstance(el, tf.Tensor):
            return el
    return None


def _gather_diag_from_jac(jac, n_samples, out_tensor):
    """
    Given jacobian `jac` (typically shape (n_samples, n_samples, *out_tail)),
    extract diagonal [i,i,...] entries and return per-sample flattened vectors (n_samples, P_flat).
    Handles dynamic shapes robustly by reshaping to (n_samples, n_samples, -1) first.
    """
    # safe reshape to (n_samples, n_samples, -1)
    jac_flat = tf.reshape(jac, (n_samples, n_samples, -1))
    # gather diagonal entries
    idx = tf.stack([tf.range(n_samples), tf.range(n_samples)], axis=1)  # (n,2)
    per_sample_flat = tf.gather_nd(jac_flat, idx)  # (n_samples, P_flat)
    # attempt to recover units if possible (last dim of out_tensor)
    try:
        out_shape = tf.shape(out_tensor)
        tail_units = out_shape[-1] if tf.size(out_shape) > 0 else 1
    except Exception:
        tail_units = 1
    return per_sample_flat, tail_units  # caller will reshape/aggregate as needed


# ------------------- approx_fisher_per_sample --------------------------
def approx_fisher_per_sample(model: keras.Model,
                             dataset,
                             loss_fn,
                             max_samples_per_batch: int = 8,
                             batch_size: int = None) -> Dict[str, np.ndarray]:
    """
    Approx per-layer per-unit Fisher using per-sample grads.
    - Fast path: builds functional submodels per-layer and vmaps a per-example function that
      computes dL/d(out) for that layer's activation.
    - Fallback path: monkeypatch-capture the raw activation tensors for subclassed/lazy layers,
      run a single forward inside one GradientTape and use tape.jacobian(loss_vec, out_t).
    Returns dict unique_layer_name -> per-unit fisher (averaged over processed samples).
    """
    md_layers = get_masked_dense_layers(model)
    if batch_size is None:
        batch_size = CONFIG.get('batch_size', 32)

    # Attempt to create functional submodels
    submodels = {}
    functional_layers = []
    functional_keys = []
    for l in md_layers:
        key = f"{l.name}_{hex(id(l))}"
        try:
            submodels[key] = keras.Model(model.inputs, l.output)
            functional_layers.append(l)
            functional_keys.append(key)            
        except Exception:
            submodels[key] = None

    multi_subm = None
    key_to_index = {}
    if functional_layers:
        try:
            outputs = [model.output] + [l.output for l in functional_layers]
            multi_subm = keras.Model(model.inputs, outputs)
            # build mapping from key to index in outputs (1-based because 0 is pred)
            for idx, key in enumerate(functional_keys, start=1):
                key_to_index[key] = idx
        except Exception:
            multi_subm = None
            key_to_index = {}            

    # --- Create vmapped_fns only for those layers we can extract from multi_subm ---
    vmapped_fns = {}
    seq_len = int(CONFIG.get('seq_len', 1))
    feature_dim = int(CONFIG.get('feature_dim', 1)) if CONFIG.get('feature_dim') else 1
    
    if multi_subm is not None:
        # single per-example fn that uses multi_subm and extracts desired activation by index
        @tf.function(input_signature=[
            tf.TensorSpec(shape=(seq_len, feature_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        ])
        def per_example_multi(x_i, y_i):
            x_b = tf.expand_dims(x_i, 0)
            y_b = tf.expand_dims(y_i, 0)
            with tf.GradientTape() as tape:
                outs = multi_subm(x_b, training=False)  # list: [pred, act1, act2, ...]
                pred_b = outs[0]
                # we won't watch all activations; caller will request the index it cares about
                loss_b = tf.reshape(loss_fn(y_b, pred_b), [])
            # return both loss and list of activations (we compute gradient for a requested activation later)
            return outs, loss_b
    
        # wrapper that given key returns function that computes per-example unit magnitudes
        def make_vmapped_for_key(idx):
            # idx is index in outs for this layer (1-based)
            def call_vmapped(x_small, y_small):
                # vectorized_map over examples
                def per_e(elems):
                    x_e, y_e = elems[0], elems[1]
                    outs, loss_scalar = per_example_multi(x_e, y_e)
                    # outs is a list-like structure; pick element idx
                    out_t = outs[idx]
                    # create tape to compute gradient of scalar w.r.t out_t
                    with tf.GradientTape() as t2:
                        # we need out_t to be watched and available in t2 context
                        t2.watch(out_t)
                        # loss already computed but we must re-evaluate or reuse: recompute pred/loss cheaply
                        # —— to keep it simple, recompute pred & loss here using a small call (one forward)
                        pred_again = outs[0]
                        # loss_scalar already available
                        loss_val = loss_scalar
                    g = t2.gradient(loss_val, out_t)
                    # print(f'\ng is:\n{g}\n')
                    if g is None:
                        units = tf.shape(out_t)[-1] if tf.rank(out_t) > 0 else 1
                        return tf.zeros((units,), dtype=tf.float32)
                    g0 = g[0] if tf.rank(g) > 0 else g
                    if tf.rank(g0) == 1:
                        unit_mag = tf.abs(g0)
                    else:
                        sq = tf.square(g0)
                        reduce_axes = tf.range(tf.rank(g0) - 1)
                        reduced = tf.sqrt(tf.reduce_sum(sq, axis=reduce_axes))
                        unit_mag = reduced
                    return tf.cast(unit_mag, tf.float32)
    
                try:
                    mapped = tf.vectorized_map(lambda elems: per_e((elems[0], elems[1])), (x_small, y_small))
                    return mapped
                except Exception:
                    lst = []
                    for i in range(int(tf.shape(x_small)[0])):
                        lst.append(per_e((x_small[i], y_small[i])))
                    return tf.stack(lst, axis=0)
            return call_vmapped
    
        # populate vmapped_fns for all keys present in key_to_index
        for key, idx in key_to_index.items():
            vmapped_fns[key] = make_vmapped_for_key(idx)
            # print(f'\nvmapped_fns: {vmapped_fns}\n')

    # accumulators
    fisher = {f"{l.name}_{hex(id(l))}": np.zeros((getattr(l, 'units', 0),), dtype=np.float32) for l in md_layers}
    count = 0

    # choose iterator
    if isinstance(dataset, tuple) and isinstance(dataset[0], np.ndarray):
        batch_iter = numpy_batch_iter(dataset[0], dataset[1], batch_size)
    else:
        batch_iter = iter(dataset)

    max_samples_per_batch = int(min(max(1, max_samples_per_batch), 8))
    for x_batch, y_batch in batch_iter:
        x_batch_tf = tf.convert_to_tensor(x_batch, dtype=tf.float32)
        y_batch_tf = tf.convert_to_tensor(y_batch, dtype=tf.float32)
        B = int(tf.shape(x_batch_tf)[0])
        n_samples = int(min(B, max_samples_per_batch))
        if n_samples == 0:
            continue
        x_small = x_batch_tf[:n_samples]
        y_small = y_batch_tf[:n_samples]

        # partition
        have_vmapped = []
        need_fallback = []
        for l in md_layers:
            key = f"{l.name}_{hex(id(l))}"
            # if key in vmapped_fns:
            #     have_vmapped.append((l, key))
            # else:
            need_fallback.append((l, key))

        # fast vmapped path
        if have_vmapped:
            prev_run_eager = bool(getattr(tf.config, "functions_run_eagerly", lambda: False)())
            if prev_run_eager:
                tf.config.run_functions_eagerly(False)
            try:
                for l, key in have_vmapped:
                    try:
                        per_example_units = vmapped_fns[key](x_small, y_small)  # (n_samples, units)
                    except Exception:
                        # fallback: iterate
                        per_list = []
                        for i in range(n_samples):
                            per_list.append(vmapped_fns[key](x_small[i:i+1, ...], y_small[i:i+1, ...])[0])
                        per_example_units = tf.stack(per_list, axis=0)
                    batch_sum = tf.reduce_sum(tf.square(per_example_units), axis=0)  # (units,)
                    try:
                        fisher[key] += batch_sum.numpy()
                    except Exception:
                        fisher[key] += np.array(batch_sum)
                count += n_samples * len(have_vmapped)
            finally:
                tf.config.run_functions_eagerly(prev_run_eager)

        # fallback using monkeypatch capture + one tape
        if need_fallback:
            prev = bool(getattr(tf.config, "functions_run_eagerly", lambda: False)())
            tf.config.run_functions_eagerly(True)
            try:
                layers_to_patch = [l for (l, _) in need_fallback]
                with monkeypatch_capture_layers(layers_to_patch, record_tensors=True, force_eager=True) as records:
                    with tf.GradientTape(persistent=True) as tape:
                        preds = model(x_small, training=False)
                        loss_vec = tf.reshape(loss_fn(y_small, preds), [-1])  # (n_samples,)
                        # gather captured outs (first recorded item per layer)
                        outs = {}
                        for (l, key) in need_fallback:
                            rec = records.get(key, [])
                            if not rec:
                                continue
                            # pick the first tensor-like item
                            out_t = _first_tensor_from_nested(rec[0])
                            if out_t is None:
                                continue
                            outs[key] = out_t
                            tape.watch(out_t)
                    # compute jacobians and extract per-sample per-unit magnitudes
                    for key, out_t in outs.items():
                        jac = tape.jacobian(loss_vec, out_t)
                        if jac is None:
                            continue
                        n = n_samples
                        per_sample_flat, tail_units = _gather_diag_from_jac(jac, n, out_t)  # (n, P_flat)
                        # attempt to reshape per_sample_flat into (n, time_prod, units) if tail_units known
                        try:
                            units = tf.shape(out_t)[-1] if tf.rank(out_t) > 0 else 1
                            # compute time_prod = P_flat / units
                            P = tf.shape(per_sample_flat)[1]
                            time_prod = tf.maximum(P // units, 1)
                            per_sample_resh = tf.reshape(per_sample_flat, (n, time_prod, units))
                            sq = tf.square(per_sample_resh)
                            unit_sq = tf.reduce_sum(sq, axis=1)  # (n, units)
                        except Exception:
                            # fallback: treat per_sample_flat as already per-unit
                            unit_sq = tf.square(per_sample_flat)
                        batch_sum = tf.reduce_sum(unit_sq, axis=0)  # (units_or_flat,)
                        try:
                            fisher[key] += batch_sum.numpy()
                        except Exception:
                            fisher[key] += np.array(batch_sum)
                        count += n
            finally:
                tf.config.run_functions_eagerly(prev)

    # normalize
    if count == 0:
        return fisher
    for name in fisher:
        fisher[name] = fisher[name] / float(count + 1e-9)
    return fisher


def run_experiment(model, train_ds, val_ds, config, mask_thresh_min):
    os.makedirs(config.get('log_dir', './logs'), exist_ok=True)
    log_dir = config['log_dir']

    # cycle-level early stopping config (use relative threshold by default)
    cycle_patience = int(config.get('cycle_patience', 3))
    cycle_min_rel = float(config.get('cycle_min_rel', 1e-3))   # relative improvement (default 0.1%)
    cycle_min_abs = float(config.get('cycle_min_abs', 1e-12))  # absolute fallback
    cycle_monitor = config.get('cycle_monitor', 'val_loss_final')
    save_best_cycle = bool(config.get('save_best_cycle_model', True))

    # callbacks for warmup
    chkpt_path = os.path.join(log_dir, 'best_base_model.keras')
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=log_dir),
        keras.callbacks.ModelCheckpoint(chkpt_path, save_best_only=True, monitor='val_loss'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=5, cooldown=10),
    ]
    print('\nWarmup training...\n')
    model.summary()
    history = model.fit(
        train_ds[0], train_ds[1],
        batch_size=config['batch_size'],
        validation_data=(val_ds[0], val_ds[1]),
        epochs=CONFIG['warm_up_epochs'],
        callbacks=callbacks,
        verbose=2
    )

    best_cycle_metric = np.inf
    best_cycle_idx = -1
    best_cycle_weights = None
    best_cycle_masks = None
    best_cycle_model_path = os.path.join(log_dir, 'best_cycle_overall.keras')
    cycles_no_improve = 0

    total_cycles = int(config.get('cycles', 10))
    for cycle in range(total_cycles):
        # create compressor with fresh config per-cycle if desired
        cfg = EvoConfig(mask_thresh_min=mask_thresh_min, log_dir=log_dir, debug=True, prune_search=True)
        compressor = EvoCompressor(model, val_ds, config=cfg, train_ds=train_ds)

        print(f'\n--- Cycle {cycle + 1}/{total_cycles} ---\n')
        chkpt_path = os.path.join(log_dir, f'best_cycle_{cycle+1}_model.keras')
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=log_dir),
            keras.callbacks.ModelCheckpoint(chkpt_path, save_best_only=True, monitor='val_loss'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.99, patience=3, cooldown=0, min_lr=1e-7),
        ]
        # train -> history
        history = model.fit(
            train_ds[0], train_ds[1],
            batch_size=config['batch_size'],
            validation_data=(val_ds[0], val_ds[1]),
            epochs=config['epochs_per_phase'],
            callbacks=callbacks,
            verbose=2
        )

        train_loss = float(history.history['loss'][-1])
        val_loss = float(history.history['val_loss'][-1])
        summary, mask_thresh_min = compressor.evaluate_and_compress(train_loss=train_loss, val_loss=val_loss)

        # pick metric
        if cycle_monitor == 'val_loss_final':
            current_metric = float(summary.get('val_loss_final', np.inf))
        elif cycle_monitor == 'pruned_val_loss':
            current_metric = float(summary.get('pruned_val_loss', np.inf))
        elif cycle_monitor == 'val_loss':
            current_metric = val_loss
        else:
            current_metric = float(summary.get('val_loss_final', np.inf))

        # RELATIVE improvement check:
        if best_cycle_metric == np.inf:
            improved = True
        else:
            rel_improv = (best_cycle_metric - current_metric) / (abs(best_cycle_metric) + 1e-12)
            abs_improv = best_cycle_metric - current_metric
            improved = (rel_improv > cycle_min_rel) or (abs_improv > cycle_min_abs)

        if improved:
            best_cycle_metric = current_metric
            best_cycle_idx = cycle
            # snapshot both weights and masks
            best_cycle_weights = [w.copy() for w in model.get_weights()]
            best_cycle_masks = _snapshot_layer_masks(model)
            cycles_no_improve = 0
            if save_best_cycle:
                try:
                    model.save(best_cycle_model_path, overwrite=True, include_optimizer=True)
                except Exception as e:
                    print(f'Warning: failed to save full model to disk: {e}')
            print(f'[CYCLE-ES] New best cycle #{cycle+1}: {cycle_monitor}={current_metric:.6g}')
        else:
            cycles_no_improve += 1
            print(f'[CYCLE-ES] No improvement on cycle {cycle+1} ({cycle_monitor}={current_metric:.6g}); '
                  f'no_improve_count={cycles_no_improve}/{cycle_patience}')

        if cycles_no_improve >= cycle_patience:
            print(f'[CYCLE-ES] Early stopping after {cycle+1} cycles. Restoring best cycle #{best_cycle_idx+1} (metric={best_cycle_metric:.6g})')
            # Prefer full-disk load (best for custom-layer state); fallback to in-memory masks+weights
            restored = False
            if save_best_cycle and os.path.exists(best_cycle_model_path):
                try:
                    loaded = keras.models.load_model(best_cycle_model_path, compile=False)
                    # Replace model weights & attributes in-place if feasible:
                    model.set_weights(loaded.get_weights())
                    _restore_layer_masks(model, _snapshot_layer_masks(loaded))
                    restored = True
                    print('[CYCLE-ES] Restored model state from disk saved .keras file.')
                except Exception as e:
                    print(f'[CYCLE-ES] Warning: full-model disk restore failed ({e}); will try in-memory masks+weights.')

            if not restored and best_cycle_weights is not None:
                model.set_weights(best_cycle_weights)
                if best_cycle_masks is not None:
                    _restore_layer_masks(model, best_cycle_masks)
                print('[CYCLE-ES] Restored best weights + mask snapshot from memory.')
            break

    # if we exhausted cycles, restore best if exists
    if cycles_no_improve < cycle_patience and best_cycle_weights is not None:
        print(f'[CYCLE-ES] Completed all cycles. Restoring best cycle #{best_cycle_idx+1}.')
        model.set_weights(best_cycle_weights)
        if best_cycle_masks is not None:
            _restore_layer_masks(model, best_cycle_masks)

    # final save
    model.save(os.path.join(log_dir, 'final_model.keras'), overwrite=True, include_optimizer=True)


def main():
    optim = keras.optimizers.AdamW(CONFIG['learning_rate'])
    
    model = build_transformer_model(CONFIG['seq_len'], d_model=CONFIG['d_model'], num_layers=CONFIG['hidden_layers'],
                                    num_heads=CONFIG['num_heads'], ff_dim=CONFIG['hidden_units'])
    model.compile(optim, loss='huber', metrics=['mae'])
    
    run_experiment(model, train_ds, val_ds, CONFIG, MASK_THRESH_MIN)    

# ------------ Main ------------ #
if __name__ == '__main__':

    main()

