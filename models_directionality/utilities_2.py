"""Model definitions for the trading‑bot project.

This file contains two Keras/TensorFlow 2.x model builders:
    • build_tcn_model: a residual, dilated Temporal Convolutional Network (TCN)
    • build_cnn_lstm_mk_model: a causal CNN + LSTM with multi‑kernel and residual blocks
Both models expect inputs with shape (seq_len, n_features) where
seq_len = 18 (for a 90‑min look‑back of 5‑min candles).
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, Add, Activation, LayerNormalization, SpatialDropout1D,
    GlobalAveragePooling1D, Dense, Dropout, Concatenate, LSTM
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


import tensorflow as tf

class MacroF1(tf.keras.metrics.Metric):
    """Macro-averaged F1 for un ≤ 2 binary classes (0 y 1)."""
    def __init__(self, threshold=0.5, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold

        # Contadores por clase
        self.tp0 = self.add_weight(name="tp0", initializer="zeros")
        self.fp0 = self.add_weight(name="fp0", initializer="zeros")
        self.fn0 = self.add_weight(name="fn0", initializer="zeros")

        self.tp1 = self.add_weight(name="tp1", initializer="zeros")
        self.fp1 = self.add_weight(name="fp1", initializer="zeros")
        self.fn1 = self.add_weight(name="fn1", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # binariza con el umbral
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.int32)

        # Clase 0
        self.tp0.assign_add(tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 0), self.dtype)))
        self.fp0.assign_add(tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 0), self.dtype)))
        self.fn0.assign_add(tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 1), self.dtype)))

        # Clase 1
        self.tp1.assign_add(tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), self.dtype)))
        self.fp1.assign_add(tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 1), self.dtype)))
        self.fn1.assign_add(tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 0), self.dtype)))

    def result(self):
        # Precision y Recall por clase
        prec0 = tf.math.divide_no_nan(self.tp0, self.tp0 + self.fp0)
        rec0  = tf.math.divide_no_nan(self.tp0, self.tp0 + self.fn0)

        prec1 = tf.math.divide_no_nan(self.tp1, self.tp1 + self.fp1)
        rec1  = tf.math.divide_no_nan(self.tp1, self.tp1 + self.fn1)

        # F1 por clase
        f1_0 = tf.math.divide_no_nan(2 * prec0 * rec0, prec0 + rec0)
        f1_1 = tf.math.divide_no_nan(2 * prec1 * rec1, prec1 + rec1)

        return (f1_0 + f1_1) / 2.0      # macro-average

    def reset_states(self):
        for var in (
                self.tp0, self.fp0, self.fn0,
                self.tp1, self.fp1, self.fn1
        ):
            var.assign(0.0)

# ────────────────────────────────────────────────
# 1)  Temporal Convolutional Network (TCN)
# ────────────────────────────────────────────────

def _residual_block_tcn(x: tf.Tensor, filters: int, dilation: int, dropout_rate: float = 0.2) -> tf.Tensor:
    """A single residual block for a TCN with two causal dilated convolutions."""
    shortcut = x

    # First dilated conv
    x = Conv1D(filters, kernel_size=3, padding="causal", dilation_rate=dilation,
               activation="relu")(x)
    x = SpatialDropout1D(dropout_rate)(x)

    # Second dilated conv
    x = Conv1D(filters, kernel_size=3, padding="causal", dilation_rate=dilation,
               activation="relu")(x)
    x = SpatialDropout1D(dropout_rate)(x)

    # Match dimensions if necessary for residual sum
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding="same")(shortcut)

    return Add()([shortcut, x])


def build_tcn_model(seq_len: int, n_features: int) -> Model:
    """Builds a lightweight residual‑dilated TCN.

    * receptive field: 18 steps (dilations 1‑2‑4 for kernel_size=3)
    * total parameters ≈ 110 k (for filters=64)
    """
    inp = Input(shape=(seq_len, n_features))
    x = inp

    for dilation in (1, 2, 4):  # receptive field 3 * (1+2+4) = 21 > 18
        x = _residual_block_tcn(x, filters=64, dilation=dilation)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out, name="TCN_residual")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[MacroF1(threshold=0.5),
                 tf.keras.metrics.AUC(name="roc_auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model


# ────────────────────────────────────────────────
# 2)  CNN + LSTM (causal) with multi‑kernel & residual blocks
# ────────────────────────────────────────────────

def _conv_branch(x: tf.Tensor, filters: int, kernel_size: int) -> tf.Tensor:
    """Single causal Conv1D branch followed by LayerNorm."""
    y = Conv1D(filters, kernel_size, padding="causal", activation="relu")(x)
    y = LayerNormalization()(y)
    return y


def _residual_conv_block(x: tf.Tensor, filters: int, kernel_size: int) -> tf.Tensor:
    """Two causal conv layers with skip connection."""
    shortcut = x
    y = Conv1D(filters, kernel_size, padding="causal", activation="relu")(x)
    y = LayerNormalization()(y)
    y = Conv1D(filters, kernel_size, padding="causal", activation="relu")(y)
    y = LayerNormalization()(y)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding="same")(shortcut)
    return Add()([shortcut, y])


def build_cnn_lstm_mk_model(seq_len: int, n_features: int) -> Model:
    """Builds a causal CNN + LSTM with multi‑kernel conv branches and residual blocks."""
    inp = Input(shape=(seq_len, n_features))

    # Multi‑kernel parallel convolutions (3, 5, 7)
    branches = [_conv_branch(inp, filters=64, kernel_size=k) for k in (3, 5, 7)]
    x = Concatenate()(branches)

    # Residual conv block to fuse features
    x = _residual_conv_block(x, filters=128, kernel_size=3)
    x = Dropout(0.3)(x)

    # Causal LSTM (unidirectional)
    x = LSTM(64, return_sequences=True, dropout=0.3)(x)

    # Multi‑head causal self‑attention
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out, name="CNN_LSTM_MultiKernel")


    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[MacroF1(threshold=0.5),
                 tf.keras.metrics.AUC(name="roc_auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model




