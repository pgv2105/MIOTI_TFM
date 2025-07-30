import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import false
from tensorflow.keras import backend as K
from keras import Layer, Input, Model
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras import regularizers
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.src.layers import SpatialDropout1D, Conv1D, MaxPooling1D
from keras.src.metrics import Precision, Recall

from services.services import show_results


# -------------------------------------------
# AUXILIAR FUNCTION TO CALCULATE FOCAL LOSS
# -------------------------------------------
def focal_loss(gamma=2., alpha=0.25):
    """
    Implementation focal loss for binary representation models.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = weight * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(loss)
    return focal_loss_fixed

# -------------------------------
# Attention layer personalized
# -------------------------------
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weight for each characteristic
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform",
                                 trainable=True)
        # Bias for each time step
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x : (batch_size, time_steps, features)
        e = K.tanh(K.dot(x, self.W) + self.b)  # (batch_size, time_steps, 1)
        a = K.softmax(e, axis=1)               # Attention weights
        output = x * a                         # Weighted output
        return K.sum(output, axis=1)           # Weighted sum over time

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])




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




# -------------------------------------------------------------------------------------- #
#            CREATION DIFFERENT LSTM MODELS TO EVALUATE ITS PERFORMANCE                  #
# -------------------------------------------------------------------------------------- #


def get_LSTM_model(X_train, y_train):
    # Create LSTM model

    nn_model = Sequential([
        Bidirectional(LSTM(160, return_sequences=True, recurrent_dropout=0.375),
                      input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.234),
        Bidirectional(LSTM(80 // 2, return_sequences=False, recurrent_dropout=0.375)),
        Dropout(0.234),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dense(1, activation='sigmoid')
    ])


    # Compile model
    nn_model.compile(optimizer=AdamW(learning_rate=0.0004, weight_decay=1e-5),
                     loss='binary_crossentropy', metrics=['accuracy'])


    # nn_model.fit(X_train, y_train, class_weight=class_weights, epochs=50, batch_size=32, validation_split=0.2)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    nn_model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)

    return nn_model


def get_best_LSTM_model(X_train, y_train):
    # Create LSTM model

    nn_model = Sequential([
        Bidirectional(LSTM(150, return_sequences=True, recurrent_dropout=0.325),
                      input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.30),
        Bidirectional(LSTM(50 , return_sequences=False, recurrent_dropout=0.325)),
        Dropout(0.30),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    nn_model.compile(optimizer=AdamW(learning_rate=0.0004, weight_decay=1e-5),
                     loss='binary_crossentropy', metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    class_weight = {0: 1.0, 1: 1.75}

    # Train model
    nn_model.fit(X_train, y_train,
                 epochs=100,
                 batch_size=128,
                 callbacks=[early_stopping],
                 validation_split=0.2,
                 class_weight = class_weight,
                 verbose=1)

    return nn_model


def get_best_LSTM_model_2(X_train, y_train, X_val, y_val):

    # Define model
    nn_model = Sequential([
        Bidirectional(LSTM(150, return_sequences=True, recurrent_dropout=0.325,
                           recurrent_regularizer=regularizers.l2(1e-4)),
                      input_shape=(X_train.shape[1], X_train.shape[2])),
        SpatialDropout1D(0.30),

        Bidirectional(LSTM(50, return_sequences=False, recurrent_dropout=0.325,
                           recurrent_regularizer=regularizers.l2(1e-4))),
        Dropout(0.30),  # Se reemplaza SpatialDropout1D por Dropout normal

        Dense(64, activation='swish', kernel_regularizer=regularizers.l2(1e-3)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='swish', kernel_regularizer=regularizers.l2(1e-3)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])

    # Compile model
    nn_model.compile(optimizer=AdamW(learning_rate=0.0004, weight_decay=1e-5),
                     loss='binary_crossentropy',
                     metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Weight classes
    class_weight = {0: 1.0, 1: 1.15}

    # Train model
    history = nn_model.fit(X_train, y_train,
                 epochs=100,
                 batch_size=128,
                 callbacks=[early_stopping, reduce_lr],
                 class_weight=class_weight,
                 validation_data=(X_val, y_val),
                 verbose=1)

    return nn_model, history


def get_best_LSTM_model_3(X_train, y_train, X_val, y_val):

    nn_model = Sequential([
        Bidirectional(LSTM(150, return_sequences=True, recurrent_dropout=0.325,
                           recurrent_regularizer=regularizers.l2(1e-4)),
                      input_shape=(X_train.shape[1], X_train.shape[2])),
        SpatialDropout1D(0.30),

        LSTM(50, return_sequences=True, recurrent_dropout=0.325,
             recurrent_regularizer=regularizers.l2(1e-4)),

        AttentionLayer(),
        Dropout(0.30),  # Se reemplaza SpatialDropout1D por Dropout normal
        Dense(64, activation='swish', kernel_regularizer=regularizers.l2(1e-3)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='swish', kernel_regularizer=regularizers.l2(1e-3)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])

    # Compile model
    nn_model.compile(optimizer=AdamW(learning_rate=0.0001),
                     loss='binary_crossentropy',
                     metrics=['accuracy',
                              Precision(name='precision'),
                              Recall(name='recall')])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.35, patience=5, min_lr=1e-6, mode='min')

    # Adjust classes weights
    class_weight = {0: 1, 1: 1.15}

    # Train model
    history = nn_model.fit(X_train, y_train,
                 epochs=100,
                 batch_size=128,
                 validation_data=(X_val, y_val),
                 callbacks=[early_stopping, reduce_lr],
                 class_weight=class_weight,
                 verbose=1)
    return nn_model, history


def get_best_LSTM_model_4(X_train, y_train, X_val, y_val):

    nn_model = Sequential([
        Bidirectional(LSTM(150, return_sequences=True, recurrent_dropout=0.325,
                           recurrent_regularizer=regularizers.l2(1e-4)),
                      input_shape=(X_train.shape[1], X_train.shape[2])),
        SpatialDropout1D(0.30),
        LSTM(50, return_sequences=True, recurrent_dropout=0.325,
             recurrent_regularizer=regularizers.l2(1e-4)),
        AttentionLayer(),
        Dropout(0.30),
        Dense(64, activation='swish', kernel_regularizer=regularizers.l2(1e-3)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='swish', kernel_regularizer=regularizers.l2(1e-3)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Compile model with new loss function
    nn_model.compile(
        optimizer=AdamW(learning_rate=0.0004),
        loss=custom_loss_fp(weight_for_false_positive=1.2, label_smoothing=0.0),
        metrics=['accuracy',
                 Precision(name='precision'),
                 Recall(name='recall')]
    )

    # Configurate callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.35, patience=5, min_lr=1e-6, mode='min')

    # Train model
    history = nn_model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=128,
        callbacks=[early_stopping, reduce_lr],
        validation_data=(X_val, y_val),
        verbose=1
    )
    return nn_model, history



def get_best_LSTM_model_5(X_train, y_train, X_val, y_val):

    nn_model = Sequential([
        # CNN block to extract local features
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
               input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(pool_size=2),
        SpatialDropout1D(0.30),

        # LSTM block to capture temporal dependencies
        Bidirectional(LSTM(150, return_sequences=True, recurrent_dropout=0.325,
                           recurrent_regularizer=regularizers.l2(1e-4))),
        LSTM(50, return_sequences=True, recurrent_dropout=0.325,
             recurrent_regularizer=regularizers.l2(1e-4)),
        AttentionLayer(),
        Dropout(0.30),
        Dense(64, activation='swish', kernel_regularizer=regularizers.l2(1e-3)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='swish', kernel_regularizer=regularizers.l2(1e-3)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model with optimizer and custom loss function
    nn_model.compile(
        optimizer=AdamW(learning_rate=0.0004),
        loss=custom_loss_fp(weight_for_false_positive=1.1, label_smoothing=0.0),
        metrics=['accuracy',
                 Precision(name='precision'),
                 Recall(name='recall')]
    )

    # Callback configuration
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.35, patience=5, min_lr=1e-6, mode='min')

    # Model training
    history = nn_model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=128,
        callbacks=[early_stopping, reduce_lr],
        validation_data=(X_val, y_val),
        verbose=1
    )

    return nn_model, history



def cross_validation(X_train, y_train, X_test, y_test):
    tscv = TimeSeriesSplit(n_splits=5)
    ensemble_preds = []
    y_test = y_test.reshape(-1)

    for train_index, val_index in tscv.split(X_train):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        model, history = get_best_LSTM_model_5(X_tr, y_tr, X_val, y_val)

        test_predictions = model.predict(X_test).flatten()
        ensemble_preds.append(test_predictions)

        # Print auxiliary results
        y_pred_lstm = (test_predictions > 0.5).astype(int)
        show_results(y_test, y_pred_lstm)

    # Get cross-validation average for final predictions
    ensemble_preds = np.array(ensemble_preds)
    y_pred_final = np.mean(ensemble_preds, axis=0)

    return y_pred_final.flatten()


def walk_forward_validation(X_train, y_train, X_test, y_test):
    """
    Performs walk-forward validation: at each iteration, the model is trained with
    all available data and predicts the next sample from the test set.

    Assumes that:
      - get_best_LSTM_model_5 is a function that takes (X_tr, y_tr, X_val, y_val)
        and returns the trained model and its history.
      - show_results is a function to visualize or print the results.

    Parameters:
        X_train, y_train: Initial training data.
        X_test, y_test: Test data to be progressively added.

    Returns:
        predictions: Array of predictions on X_test.
    """
    predictions = []
    y_test = y_test.reshape(-1)

    # Iterate over each sample in the test set
    for i in range(len(X_test)):
        # Optionally split the current X_train into training and validation
        # For example, 80% for training and 20% for validation
        split_index = int(0.8 * len(X_train))
        X_tr, X_val = X_train[:split_index], X_train[split_index:]
        y_tr, y_val = y_train[:split_index], y_train[split_index:]

        # Train the model with current data
        model, history = get_best_LSTM_model_5(X_tr, y_tr, X_val, y_val)

        # Predict the next sample (one per iteration)
        test_sample = X_test[i:i+1]
        test_prediction = model.predict(test_sample).flatten()
        predictions.append(test_prediction[0])

        # Show results for the current sample (you can adjust threshold or metrics)
        y_pred_lstm = (test_prediction > 0.5).astype(int)
        show_results(y_test[i:i+1], y_pred_lstm)

        # Update the training set: add the now-known test sample
        X_train = np.concatenate([X_train, test_sample])
        y_train = np.concatenate([y_train, y_test[i:i+1]])

    return np.array(predictions)

