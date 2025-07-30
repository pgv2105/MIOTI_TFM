import numpy as np
import pandas as pd
import shap

from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam, AdamW
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import optuna as opt


import xgboost as xgb
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from sklearn.linear_model import LogisticRegression

from .HYBRIDS.model_HYBRIDS import cross_validation_HYBRID_MODEL
from .HYBRIDS.model_HYBRIDS_MULTIVARIATE import cross_validation_HYBRID_MODEL_MULTIVARIATE
from .TRANSFORMER.model_TRANSFORMER import cross_validation_TRANSFORMER, study_hyperparameters_TRANSFORMER
from .evaluate_metamodels import (
    eval_LGBM_metamodel,
    eval_XGB_metamodel,
    eval_MLP_metamodel
)

from services.services import create_sequences, show_results
from .LSTM.models_LSTM import get_best_LSTM_model, get_best_LSTM_model_2, get_best_LSTM_model_3, get_best_LSTM_model_4, \
    cross_validation, walk_forward_validation
from .XGBOOST.models_XGBOOST import get_best_xgBoost_model

sequence_length = 5  # Define window length




def evaluate_with_LSTM(X, Y):


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

    # -------------   Step 1: TRAIN DATA WITH OUR DIFFERENT MODELS AVAILABLE   -------------

    # lstm_test_predictions = cross_validation(X_train, y_train, X_test, y_test)
    lstm_test_predictions = cross_validation_HYBRID_MODEL(X_train, y_train, X_test, y_test)
    # lstm_test_predictions = cross_validation_HYBRID_MODEL_MULTIVARIATE(X_train, y_train, X_test, y_test)
    # lstm_test_predictions = walk_forward_validation(X_train, y_train, X_test, y_test)

    # Get optimal threshold
    #fpr, tpr, thresholds = roc_curve(y_test, lstm_test_predictions)
    #optimal_idx = np.argmax(tpr - fpr)
    #optimal_threshold = thresholds[optimal_idx]

    y_test = y_test.reshape(-1)
    y_pred_lstm = (lstm_test_predictions > 0.5).astype(int)
    show_results(y_test,y_pred_lstm)



    '''
    --------------  CODE TO STACK LSTM-TRANFORMER PREDICTIONS WITH X META-CLASSIFIERS  -------------
    
       # -- Step 2: Prepare Data for XGBoost --
    # Combine LSTM predictions with original features

    #X_train_combined = np.hstack([X_train.reshape(X_train.shape[0], -1), lstm_train_predictions.reshape(-1, 1)])
    #X_test_combined = np.hstack([X_test.reshape(X_test.shape[0], -1), lstm_test_predictions.reshape(-1, 1)])

    X_train_xgboost = X_train.reshape(X_train.shape[0], -1)
    y_train = y_train.reshape(-1)
    X_test_xgboost = X_test.reshape(X_test.shape[0], -1)

    # -- Step 3: Train XGBoost
    xgb_model = get_best_xgBoost_model(X_train_xgboost, y_train)

    # -- Step 4: Evaluate the Combined Model
    xgb_pred = xgb_model.predict(X_test_xgboost)
    xgb_prob_train = xgb_model.predict_proba(X_train_xgboost)[:, 1]
    xgb_prob_test = xgb_model.predict_proba(X_test_xgboost)[:, 1]

    show_results(y_test, xgb_pred)



    # Combinamos las predicciones OOF de ambos modelos para formar las nuevas características
    lstm_train_predictions= []
    X_meta_train = np.column_stack((lstm_train_predictions, xgb_prob_train))
    X_meta_test = np.column_stack((lstm_test_predictions, xgb_prob_test))

    #  --------- METAMODELS PREDICTIONS -----------
    eval_LGBM_metamodel(X_meta_train, X_meta_test, y_train, y_test)
    eval_XGB_metamodel(X_meta_train, X_meta_test, y_train, y_test)
    eval_MLP_metamodel(X_meta_train, X_meta_test, y_train, y_test)

    # ROC-AUC Score
    # print("ROC-AUC Score:", roc_auc_score(y_test, y_xgb_prob))
    
    '''


