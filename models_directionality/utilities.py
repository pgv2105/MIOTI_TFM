import pickle
import numpy as np
import joblib
import matplotlib.pyplot as plt

import pandas as pd
from keras import Regularizer
from keras.src.layers import Conv1D, SpatialDropout1D, Concatenate
from keras.src.losses import BinaryCrossentropy
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from backtesting.model_directionality.get_model_classifier_data import get_train_data_through_strategy
from backtesting.model_directionality.get_model_nn_data import get_nn_train_data_through_strategy
from backtesting.model_directionality.get_model_nn_tendency_data import get_nn_train_tendency_data_through_strategy
from backtesting.model_directionality.utilities_2 import build_cnn_lstm_mk_model, MacroF1
from models.HYBRIDS.model_HYBRIDS import custom_loss_fp, PositionalEncoding, get_best_HYBRID_model, \
    get_best_HYBRID_model_tendency

from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.metrics import precision_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scikeras.wrappers import KerasClassifier
from xgboost import XGBClassifier
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
# from xgboost.callback import EarlyStopping
from keras.src.metrics import Precision, Recall, AUC
from keras.src.optimizers import AdamW, Adam
from keras.src.saving import load_model
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, BatchNormalization, Add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import register_keras_serializable, to_categorical
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dropout,
    Bidirectional, LSTM, Dense, Layer, Multiply, Permute, MultiHeadAttention,
    Activation, Softmax, Lambda, GRU, MultiHeadAttention, LayerNormalization, Add
)
from tcn import TCN

import tensorflow.keras.backend as K



import xgboost as xgb


from services.services import show_results


best_feat = [
    'kf_slope_short', 'kf_slope_long',
    'zl_ema5',
    'kama10',        'srsi5',
    'fisher10',
    'delta',         'accel'
]

feature_names = [
    'kf_slope_fast', 'kf_slope_slow',
    'kf_trend_fast', 'kf_trend_slow',
    'kf_res_fast',   'kf_res_slow',
    'rsi_fast', 'rsi_slow',
    'vol_fast', 'vol_slow',
    'macd',
    'ema_fast_slow',
    'sg_trend', 'sg_slope', 'sg_res',
    'ss_fast_price', 'ss_slow_price',
    'zl_ema', 'kama', 'srsi',
    'fisher', 'sup_tr',
    'delta', 'accel'
]

def focal_loss(alpha=0.25, gamma=2.0):
    def loss_fn(y_true, y_pred):
        # Aseguramos flotantes y clip
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())

        # 1) Binary cross‑entropy manual por muestra
        bce = - (y_true * tf.math.log(y_pred) +
                 (1. - y_true) * tf.math.log(1. - y_pred))

        # 2) p_t = probabilidad asignada a la clase real
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1. - y_pred)

        # 3) alpha factor
        alpha_factor = tf.where(tf.equal(y_true, 1),
                                alpha, 1. - alpha)

        # 4) modulating factor
        modulating_factor = tf.pow(1. - p_t, gamma)

        # 5) focal loss
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)
    return loss_fn


class F1ThresholdCallback(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.X_val).flatten()
        best_f1, best_thr = 0, 0.5
        for thr in np.linspace(0.1, 0.9, 17):
            f1 = f1_score(self.y_val, probs >= thr)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        logs['val_f1'] = best_f1
        logs['best_thr'] = best_thr
        print(f" — val_f1: {best_f1:.4f} @ thr={best_thr:.2f}")


class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super().build(input_shape)
    def call(self, x):
        # x: (batch, timesteps, features)
        e = K.tanh(K.dot(x, self.W))          # (batch, timesteps, 1)
        e = K.squeeze(e, -1)                  # (batch, timesteps)
        alpha = K.softmax(e)                  # (batch, timesteps)
        alpha = K.expand_dims(alpha, -1)      # (batch, timesteps, 1)
        return K.sum(x * alpha, axis=1)       # (batch, features)


class Attention2(Layer):
    def build(self, input_shape):
        # input_shape = (batch, timesteps, features)
        self.W = self.add_weight(
            name="att_weight",                # ← ahora explícito
            shape=(input_shape[-1], 1),       # (features, 1)
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)            # buena práctica

    def call(self, x):
        # x → (B, T, F)
        e     = tf.tanh(tf.matmul(x, self.W))     # (B, T, 1)
        e     = tf.squeeze(e, -1)                 # (B, T)
        alpha = tf.nn.softmax(e)                  # (B, T)
        alpha = tf.expand_dims(alpha, -1)         # (B, T, 1)
        return tf.reduce_sum(x * alpha, axis=1)   # (B, F)

# ---------- Weighted BCE ----------
def weighted_bce(pos_weight):
    bce = BinaryCrossentropy(from_logits=False)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        w      = 1.0 + y_true * (pos_weight - 1.0)
        return bce(y_true, y_pred, sample_weight=w)
    return loss

class F1Callback(Callback):
    def __init__(self, X_val, y_val, threshold=0.5):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        y_pred = (self.model.predict(self.X_val) >= self.threshold).astype(int)
        f1 = f1_score(self.y_val, y_pred)
        logs = logs or {}
        logs['val_f1'] = f1
        print(f' — val_f1: {f1:.4f}')


def set_strategy_output(model_nn, X, Y, mode):

    if mode == 'nn':
        features = get_nn_train_data_through_strategy(model_nn, X, Y)
    else:
        features = get_train_data_through_strategy(model_nn, X, Y)

    features = pd.DataFrame(features)

    # features = pd.get_dummies(features, columns=['session'], prefix='session')

    features.set_index('time_entry', inplace = True)

    return features



def create_lag_columns(df_features):
    df_features = df_features.sort_index()

    # Create los lags for our specified columns
    for col in ['kalman_slope', 'ema', 'macd', 'dmi_diff', 'rsi_val']:
        df_features[f'{col}_lag1'] = df_features[col].shift(1)  # 1 set candle (10 min)
        df_features[f'{col}_lag2'] = df_features[col].shift(2)  # 2 set candle (20 min)
        df_features[f'{col}_lag3'] = df_features[col].shift(3)  # 3 set candle (30 min)

    # 3) Drop those entries which are not useful
    df_features = df_features.dropna().reset_index(drop=True)

    return df_features

def calculate_classifier_data(model, X_train, X_test, y_train, y_test):
    # Store train features data
    features = set_strategy_output(model, X_train, y_train,'classifier')

    mask = (features['signal'] != -1)
    features = features.loc[mask]
    features.to_pickle('classifier_features')

    # Store test features data
    features_test = set_strategy_output(model, X_test, y_test,'classifier')
    features_test.to_pickle('classifier_test_features')



def calculate_nn_data(model_nn, X_train, X_test, y_train, y_test):
    # Store train features data
    features = set_strategy_output(model_nn, X_train, y_train,'nn')
    features.to_pickle('data/nn_features_tendency')

    # Store test
    features_test = set_strategy_output(model_nn, X_test, y_test,'nn')
    features_test.to_pickle('data/nn_test_features_tendency')

def train_and_eval_classifier(X_train, X_val, y_train, y_val, X_test, y_test):

    # 1) Convierte tus datos a DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    # 2) Define tus hiperparámetros (idénticos a los del wrapper)
    params = {
        "eta": 0.03,
        "max_depth": 4,            # de 3 a 4
        "min_child_weight": 10,    # de 20 a 10
        "subsample": 0.7,          # de 0.5 a 0.7
        "colsample_bytree": 0.7,   # de 0.5 a 0.7
        "gamma": 0.3,              # de 0.5 a 0.3
        "alpha": 0.7,              # de 1.0 a 0.7
        "lambda": 1.5,             # de 2.0 a 1.5
        "seed": 42
    }

    # 3) Entrena con early stopping
    evals = [(dtrain, "train"), (dval, "validation")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=200,
        verbose_eval=10       # imprime cada 10 iteraciones
    )

    print(f"Mejor iteración: {model.best_iteration}")
    print(f"Mejor {model.best_score=}")

    # 4) Predicción sobre test
    probs = model.predict(dtest)

    # probs = model.predict(X_test_model)
    preds = (probs > 0.5).astype(int)
    show_results(y_test.astype(int),preds)
    #preds = probs


    xgb.plot_importance(model, max_num_features=15)
    plt.tight_layout()
    plt.show()

def train_and_eval_TCN(X_train, X_val, y_train, y_val, X_test, y_test):

    n_samples, timesteps, n_features = X_train.shape
    n_samples_test, _, _ = X_test.shape
    n_samples_val, _, _ = X_val.shape

    scaler = StandardScaler()

    X_reshaped = X_train.reshape(-1, n_features)
    X_scaled_reshaped = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(n_samples, timesteps, n_features)

    X_val_reshaped = X_val.reshape(-1, n_features)
    X_scaled_val_reshaped = scaler.transform(X_val_reshaped)
    X_val_scaled = X_scaled_val_reshaped.reshape(n_samples_val, timesteps, n_features)

    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled_reshaped.reshape(n_samples_test, timesteps, n_features)

    # --- 3. Construcción del modelo ---
    tcn_kwargs = dict(
        nb_filters=64,
        kernel_size=5,              # de 3 → 5
        dilations=[1, 2, 4, 8, 16], # ampliamos receptive field
        dropout_rate=0.1,
        return_sequences=False,
        activation='relu'
    )

    model = Sequential([
        Input(shape=(timesteps, n_features)),
        TCN(**tcn_kwargs),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(2, activation='softmax')  # 2 salidas para softmax
    ])

    # --- 4. Compilación con label smoothing ---
    loss = CategoricalCrossentropy(label_smoothing=0.05)
    optimizer = Adam(learning_rate=3e-4, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['AUC', 'accuracy']
    )

    # --- 5. Callbacks de entrenamiento ---
    callbacks = [
        EarlyStopping(monitor='val_AUC', mode='max', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5, min_lr=1e-5),
    ]

    y_train_cat = to_categorical(y_train, num_classes=2)
    y_val_cat   = to_categorical(y_val,   num_classes=2)
    y_test_cat  = to_categorical(y_test,  num_classes=2)

    # --- 7. Entrenamiento ---
    history = model.fit(
        X_scaled,
        y_train_cat,
        epochs=60,
        batch_size=128,
        validation_data=(X_val_scaled, y_val_cat),
        callbacks=callbacks
    )

    # 6. Evaluation
    test_predictions = model.predict(X_test_scaled).flatten()
    y_pred_tcn = (test_predictions > 0.5).astype(int)
    show_results(y_test, y_pred_tcn)
    y_pred_tcn = (test_predictions > 0.5).astype(int)




def train_and_eval_LSTM(X_train, X_val, y_train, y_val, X_test, y_test):

    n_samples, timesteps, n_features = X_train.shape
    n_samples_test, _, _ = X_test.shape
    n_samples_val, _, _ = X_val.shape

    scaler = StandardScaler()

    X_reshaped = X_train.reshape(-1, n_features)
    X_scaled_reshaped = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(n_samples, timesteps, n_features)

    X_val_reshaped = X_val.reshape(-1, n_features)
    X_scaled_val_reshaped = scaler.transform(X_val_reshaped)
    X_val_scaled = X_scaled_val_reshaped.reshape(n_samples_val, timesteps, n_features)

    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled_reshaped.reshape(n_samples_test, timesteps, n_features)

    model = Sequential([
        LSTM(64, input_shape=(timesteps, n_features),
             recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.1),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dense(1, activation='sigmoid')
    ])

    opt = Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(opt, loss='binary_crossentropy', metrics=['AUC','accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_AUC', mode='max', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5, min_lr=1e-5)
    ]

    model.fit(X_scaled, y_train,
              epochs=100, batch_size=128,
              validation_data=(X_val_scaled, y_val),  # o split temporal
              callbacks=callbacks,
              shuffle=False)
    # -- Test evaluation
    # loss, acc = model.evaluate(X_test_scaled, y_test)

    test_predictions = model.predict(X_test_scaled).flatten()

    # Print aux results
    y_pred_lstm = (test_predictions > 0.5).astype(int)
    show_results(y_test,y_pred_lstm)
    y_pred_lstm = (test_predictions > 0.5).astype(int)


def train_and_eval_CNN_LSTM(X_train, X_val, y_train, y_val, X_test, y_test):

    n_samples, timesteps, n_features = X_train.shape
    n_samples_test, _, _ = X_test.shape
    n_samples_val, _, _ = X_val.shape

    scaler = StandardScaler()

    X_reshaped = X_train.reshape(-1, n_features)
    X_scaled_reshaped = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(n_samples, timesteps, n_features)

    joblib.dump(scaler, '../data/models/standard_scaler_model_directionality_1.pkl')

    X_val_reshaped = X_val.reshape(-1, n_features)
    X_scaled_val_reshaped = scaler.transform(X_val_reshaped)
    X_val_scaled = X_scaled_val_reshaped.reshape(n_samples_val, timesteps, n_features)

    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled_reshaped.reshape(n_samples_test, timesteps, n_features)

    inp = Input(shape=(timesteps, n_features))



    # 1) CNN causal para patrones locales
    x = Conv1D(64, kernel_size=3, padding='causal', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, kernel_size=3, padding='causal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # 2) BiLSTM para dinámica bidireccional
    x = LSTM(64, return_sequences=True, activation='tanh')(x)
    x = Dropout(0.3)(x)

    # 3) Atención sobre la secuencia de salidas
    x = Attention()(x)


    # 4) Capa fully‑connected final
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=AdamW(learning_rate=3e-3),
        loss='binary_crossentropy',
        metrics=[MacroF1(threshold=0.5),
                 AUC(curve='PR', name='pr_auc'),
                 Precision(name='precision'),
                 Recall(name='recall')]
    )

    callbacks = [
        EarlyStopping(monitor='val_macro_f1', mode='max', patience=20, restore_best_weights=True, verbose = 1),
        ReduceLROnPlateau(monitor='val_pr_auc', patience=5, factor=0.35, min_lr=1e-6, mode='max')
    ]

    history = model.fit(
        X_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=70,
        batch_size=64,
        callbacks=callbacks,
        verbose = True
    )


    #model = load_model('../data/models/model_DIRECTIONALITY_0_4942.keras')




    test_predictions = model.predict(X_test_scaled).flatten()

    # Print aux results
    y_pred_lstm = (test_predictions > 0.5).astype(int)
    show_results(y_test,y_pred_lstm)

    #fpr, tpr, thresholds = roc_curve(y_test, lstm_test_predictions)
    #optimal_idx = np.argmax(tpr - fpr)
    #optimal_threshold = thresholds[optimal_idx]