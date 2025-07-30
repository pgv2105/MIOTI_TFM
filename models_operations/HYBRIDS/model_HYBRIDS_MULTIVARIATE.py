import numpy as np
import tensorflow as tf
from keras import Layer
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.metrics import Precision, Recall
from keras.src.optimizers import AdamW
from keras.src.saving import load_model
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, BatchNormalization, Add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import recall_score
from sklearn.utils import class_weight

import keras_tuner as kt

from models.LSTM.models_LSTM import get_best_LSTM_model_5
from services.services import show_results

@tf.keras.utils.register_keras_serializable()
def custom_loss_fp(weight_for_false_positive, label_smoothing=0.0):
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


@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model

    def get_config(self):
        config = super().get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos_encoding = self.positional_encoding(self.position, self.d_model)
        pos_encoding = pos_encoding[:, :seq_len, :]
        return inputs + pos_encoding

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def compute_output_shape(self, input_shape):
        return input_shape



class BalancedAccCallback(Callback):
    def __init__(self, X_val, y_val_onehot):
        super().__init__()
        self.X_val = X_val
        # ----> convierte y_val a vector int aquí
        self.y_val = np.argmax(y_val_onehot, axis=1)

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        recalls = recall_score(self.y_val, y_pred,
                               labels=[0,1,2], average=None)
        bal_acc = np.mean(recalls)
        if logs is not None:
            logs['val_balanced_accuracy'] = bal_acc
        print(f" — val_balanced_accuracy: {bal_acc:.4f} "
              f"(recall0={recalls[0]:.3f}, recall1={recalls[1]:.3f}, recall2={recalls[2]:.3f})")


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




def build_transformer_model(input_shape, head_size=32, num_heads=2, ff_dim=128, num_transformer_blocks=1, mlp_units=[64, 32], dropout=0.3):
    inputs = Input(shape=input_shape)

    # LSTM to capture local dependencies
    x = LSTM(32, return_sequences=True)(inputs)

    # Add positional encoding for transformer
    x = PositionalEncoding(position=input_shape[0], d_model=32)(x)

    # Stack transformers blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Pooling global in order to get flat sequence
    x = GlobalAveragePooling1D()(x)

    # Add dense layers
    for units in mlp_units:
        x = Dense(units, activation='swish', kernel_regularizer=regularizers.l2(1e-3))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


def get_best_HYBRID_model_multivariate(X_train, y_train, X_val, y_val):


    # Define input_shape
    input_shape = (X_train.shape[1], X_train.shape[2])

    model = build_transformer_model(input_shape,
                                    head_size=96,
                                    num_heads=8,
                                    ff_dim=128,
                                    num_transformer_blocks=3,
                                    mlp_units=[32, 32],
                                    dropout=0.2)

    # Compile model with specific metrics
    model.compile(
        optimizer=AdamW(learning_rate=0.0009),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            Precision(name='prec_long', class_id=1),
            Recall   (name='recall_long',   class_id=1),
            Precision(name='prec_short', class_id=2),
            Recall   (name='recall_short',   class_id=2)
        ]
    )

    y_int = np.argmax(y_train, axis=1)

    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_int),
        y=y_int
    )
    class_weight_dict = {i: w for i, w in enumerate(weights)}

    sample_weight = np.take(
        [class_weight_dict[0], class_weight_dict[1], class_weight_dict[2]],
        y_int
    )

    # Configure callbacks
    early_stop = EarlyStopping(
        monitor='val_balanced_accuracy',
        mode='max',
        patience=10,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_balanced_accuracy',
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    bal_acc_cb = BalancedAccCallback(X_val, y_val)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        sample_weight=sample_weight,
        batch_size=64,
        callbacks=[bal_acc_cb, early_stop, reduce_lr],
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


def cross_validation_HYBRID_MODEL_MULTIVARIATE(X_train, y_train, X_test, y_test):
    tscv = TimeSeriesSplit(n_splits=5)
    ensemble_preds = []
    # y_test = y_test.reshape(-1)

    length = int(X_train.shape[0]*0.8)
    #for train_index, val_index in tscv.split(X_train):
    X_tr, X_val = X_train[:length], X_train[length:]
    y_tr, y_val = y_train[:length], y_train[length:]

    y_train_cat = to_categorical(y_tr, num_classes=3)
    y_val_cat   = to_categorical(y_val,   num_classes=3)

    model, history = get_best_HYBRID_model_multivariate(X_tr, y_train_cat, X_val, y_val_cat)

    # -- SAVE TRAINED MODEL

    # Reassign loss function name
    #loss_function = custom_loss_fp(1.1, 0.0)
    #loss_function.__name__ = "loss_fn"  # Se debe registrar con el nombre "loss_fn"


    model.save('model_PLUS_MULTIVARIATE.keras')
    #model = load_model('model_PLUS_MULTIVARIATE.keras', custom_objects={
    #    'loss_fn': loss_function,
    #    'PositionalEncoding': PositionalEncoding
    #})
    test_predictions = model.predict(X_test).flatten()
    ensemble_preds.append(test_predictions)

    # Print aux results
    y_pred_lstm = (test_predictions > 0.5).astype(int)
    show_results(y_test,y_pred_lstm)

    # Get cross validation average for final predictions
    ensemble_preds = np.array(ensemble_preds)
    y_pred_final = np.mean(ensemble_preds, axis=0)


    return y_pred_final.flatten()
