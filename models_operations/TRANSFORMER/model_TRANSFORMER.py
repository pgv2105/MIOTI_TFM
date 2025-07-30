import numpy as np
import tensorflow as tf
from keras import Layer
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.metrics import Precision, Recall
from keras.src.optimizers import AdamW
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, BatchNormalization, Add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

import keras_tuner as kt

from models.LSTM.models_LSTM import get_best_LSTM_model_5
from services.services import show_results


def custom_loss_fp(weight_for_false_positive=1.15, label_smoothing=0.0):
    """
    Loss function personalized to penalize false positives.

    Parameters:
      - weight_for_false_positive: Factor to increment penalization hwen y_true is 0 and y_pred is high (False Positive).
      - label_smoothing: value between 0 and 1. Set 0 by default to deactivate.

    Loss is calculated as a binary crossentropy modified:
      loss = - [ y_true * log(y_pred) + (1 - y_true) * weight_for_false_positive * log(1 - y_pred) ]
    """
    def loss_fn(y_true, y_pred):
        # Optional: apply label smoothing to avoid overconfidence
        if label_smoothing > 0:
            y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing

        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        # Penalize errors in class 0 (False positives)
        loss = - (y_true * tf.math.log(y_pred) +
                  (1 - y_true) * weight_for_false_positive * tf.math.log(1 - y_pred))
        return tf.reduce_mean(loss)
    return loss_fn

class PositionalEncoding(Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'position': self.position,
            'd_model': self.d_model,
        })
        return config

    def call(self, inputs):
        # inputs: (batch_size, seq_len, d_model)
        seq_len = tf.shape(inputs)[1]
        pos_encoding = self.positional_encoding(self.position, self.d_model)
        # Only use first 'seq_len' positions
        pos_encoding = pos_encoding[:, :seq_len, :]
        return inputs + pos_encoding

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        # Apply sin function to pair values and cos tu odd values
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates


# Transformer (Encoder)
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Normalization Layer and Multi Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Add()([attn_output, inputs])

    # Feed-forward layer
    x_ff = LayerNormalization(epsilon=1e-6)(x)
    x_ff = Dense(ff_dim, activation="swish")(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x_ff, x])
    return x


def build_transformer_model(input_shape, head_size=64, num_heads=4, ff_dim=128, num_transformer_blocks=2, mlp_units=[64, 32], dropout=0.3):
    inputs = Input(shape=input_shape)

    # Agregar codificación posicional
    x = PositionalEncoding(position=input_shape[0], d_model=input_shape[-1])(inputs)

    # Apilar bloques Transformer
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Pooling global para aplanar la secuencia
    x = GlobalAveragePooling1D()(x)

    # Capas densas finales similares a tu modelo original
    for units in mlp_units:
        x = Dense(units, activation='swish', kernel_regularizer=regularizers.l2(1e-3))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model


def get_best_TRANSFORMER_model(X_train, y_train, X_val, y_val):


    # Supongamos que X_train, y_train, X_val, y_val ya están definidos y preprocesados
    input_shape = (X_train.shape[1], X_train.shape[2])

    model = build_transformer_model(input_shape,
                                    head_size=96,
                                    num_heads=8,
                                    ff_dim=128,
                                    num_transformer_blocks=3,
                                    mlp_units=[32, 32],
                                    dropout=0.2)

    # Compilamos el modelo usando el mismo optimizador y función de pérdida que en tu modelo LSTM
    model.compile(
        optimizer=AdamW(learning_rate=0.0009),
        loss=custom_loss_fp(weight_for_false_positive=1, label_smoothing=0.0),
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )

    # Configuración de callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.35, patience=5, min_lr=1e-6, mode='min')

    # Entrenamiento del modelo
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=128,
        callbacks=[early_stopping, reduce_lr],
        validation_data=(X_val, y_val),
        verbose=1
    )
    return model, history


def build_model(hp, input_shape):
    # Definición de hiperparámetros a explorar:
    head_size = hp.Int("head_size", min_value=32, max_value=128, step=32)
    num_heads = hp.Choice("num_heads", values=[2, 4, 8])
    ff_dim = hp.Int("ff_dim", min_value=64, max_value=256, step=64)
    num_transformer_blocks = hp.Int("num_transformer_blocks", min_value=1, max_value=3, step=1)

    # Para las unidades del MLP, se definen dos capas
    mlp_units = [
        hp.Int("mlp_units_0", min_value=32, max_value=128, step=32),
        hp.Int("mlp_units_1", min_value=16, max_value=64, step=16)
    ]
    dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-3, sampling="log")

    # Construimos el modelo usando la función build_transformer_model definida previamente
    model = build_transformer_model(input_shape,
                                    head_size=head_size,
                                    num_heads=num_heads,
                                    ff_dim=ff_dim,
                                    num_transformer_blocks=num_transformer_blocks,
                                    mlp_units=mlp_units,
                                    dropout=dropout)

    # Compilamos el modelo con los hiperparámetros definidos
    model.compile(
        optimizer=AdamW(learning_rate=learning_rate),
        loss=custom_loss_fp(weight_for_false_positive=1.1, label_smoothing=0.0),
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )
    return model

def study_hyperparameters_TRANSFORMER(X_train, y_train, X_test, y_test):

    input_shape = (X_train.shape[1], X_train.shape[2])
    y_test = y_test.reshape(-1)

    length = int(X_train.shape[0]*0.8)
    #for train_index, val_index in tscv.split(X_train):
    X_tr, X_val = X_train[:length], X_train[length:]
    y_tr, y_val = y_train[:length], y_train[length:]

    # Configuración de callbacks (los mismos que en tu entrenamiento original)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.35, patience=5, min_lr=1e-6, mode='min')

    # Inicialización de KerasTuner: aquí usamos búsqueda aleatoria (RandomSearch)
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, input_shape),
        objective='val_loss',  # Puedes cambiar el objetivo si prefieres optimizar otra métrica
        max_trials=20,         # Número de combinaciones a probar
        executions_per_trial=1,
        directory='hp_tuning',
        project_name='transformer_tuning'
    )

    # Ejecución de la búsqueda de hiperparámetros
    tuner.search(X_tr, y_tr,
                 epochs=100,
                 batch_size=128,
                 validation_data=(X_val, y_val),
                 callbacks=[early_stopping, reduce_lr],
                 verbose=1)

    # Una vez finalizada la búsqueda, puedes ver el resumen de los mejores hiperparámetros:
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Mejores hiperparámetros encontrados:")
    print(best_hp.values)


def cross_validation_TRANSFORMER(X_train, y_train, X_test, y_test):
    tscv = TimeSeriesSplit(n_splits=5)
    ensemble_preds = []
    y_test = y_test.reshape(-1)

    length = int(X_train.shape[0]*0.8)
    for train_index, val_index in tscv.split(X_train):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        model, history = get_best_TRANSFORMER_model(X_tr, y_tr, X_val, y_val)
        #model, history = get_best_LSTM_model_5(X_tr, y_tr, X_val, y_val)

        test_predictions = model.predict(X_test).flatten()
        ensemble_preds.append(test_predictions)

        # Print aux results
        y_pred_lstm = (test_predictions > 0.5).astype(int)
        show_results(y_test,y_pred_lstm)

    # Get cross validation average for final predictions
    ensemble_preds = np.array(ensemble_preds)
    y_pred_final = np.mean(ensemble_preds, axis=0)


    return y_pred_final.flatten()
