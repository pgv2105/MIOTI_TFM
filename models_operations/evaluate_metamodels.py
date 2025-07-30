import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier

from services.services import show_results

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def eval_LGBM_metamodel(X_meta_train, X_meta_test, y_train, y_test):

    meta_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    meta_model.fit(X_meta_train, y_train)
    meta_predictions = meta_model.predict(X_meta_test)

    print("----------------  META-MODEL:  LGBM   --------------")
    show_results(y_test, meta_predictions)


def eval_XGB_metamodel(X_meta_train, X_meta_test, y_train, y_test):

    meta_model = XGBClassifier(n_estimators=100,
                                learning_rate=0.1,
                                random_state=42,
                                use_label_encoder=False,
                                eval_metric='logloss')
    meta_model.fit(X_meta_train, y_train)
    meta_predictions = meta_model.predict(X_meta_test)

    print("----------------  META-MODEL:  XGB   --------------")
    show_results(y_test, meta_predictions)


def eval_MLP_metamodel(X_meta_train, X_meta_test, y_train, y_test):

    meta_model = Sequential()
    meta_model.add(Dense(8, activation='relu', input_dim=X_meta_train.shape[1]))
    meta_model.add(Dense(4, activation='relu'))
    meta_model.add(Dense(1, activation='sigmoid'))

    meta_model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    # Ajusta la red (puedes modificar epochs y batch_size según tu dataset)
    meta_model.fit(X_meta_train, y_train, epochs=50, batch_size=32, verbose=0)

    y_prob = meta_model.predict(X_meta_test).flatten()

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Optimal Threshold:", optimal_threshold)

    meta_predictions = (y_prob > 0.5).astype(int)

    print("----------------  META-MODEL:  MLP   --------------")
    show_results(y_test, meta_predictions)