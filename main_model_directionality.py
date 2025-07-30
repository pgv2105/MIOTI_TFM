import pickle
import numpy as np

import pandas as pd

from models_directionality.utilities import train_and_eval_LSTM, train_and_eval_CNN_LSTM, create_lag_columns, \
    calculate_nn_data, calculate_classifier_data, train_and_eval_classifier
from models_operations.HYBRIDS.model_HYBRIDS import custom_loss_fp, PositionalEncoding, get_best_HYBRID_model
from keras.src.saving import load_model

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib


ALL_feature_names_2 = [
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

ALL_feature_names = [
    'kf_slope_short', 'kf_slope_long',
    'kf_trend_short', 'kf_trend_long',
    'kf_res_short',   'kf_res_long',
    'rsi100',        'rsi240',
    'vol150',        'vol500',
    'macd_hist',     'ema12_26',
    'sg151_trend',   'sg151_slope',
    'sg151_res',     'ss61_price',
    'ss420_price',   'ss61_slope',
    'ss420_slope',   'zl_ema5',
    'kama10',        'srsi5',
    'fisher10',      'sup_tr7',
    'delta',         'accel'
]




if __name__ == '__main__':


    loss_function = custom_loss_fp(1.1, 0.0)
    loss_function.__name__ = "loss_fn"  # Se debe registrar con el nombre "loss_fn"

    # MODEL PARA LONG Y SHORT ACTUAL
    model_nn = load_model('data/model_PLUS_EXTENDED.keras', custom_objects={
        'loss_fn': loss_function,
        'PositionalEncoding': PositionalEncoding
    })

    # Load previous preprocessed data -- Da igual la normal que la INVERSE, son los mismos datos
    with open("data/backtest_data_model_PLUS_MULTIVARIATE.pkl", "rb") as f:
        data = pickle.load(f)

    X_LSTM = data["dataset_X"]
    Y_LSTM = data["dataset_Y"]

    #  Separate train/test
    X_train, X_test, y_train, y_test = train_test_split(X_LSTM, Y_LSTM, test_size=0.1, shuffle=False)


    # calculate_classifier_data(model_nn, X_train, X_test, y_train, y_test)
    calculate_nn_data(model_nn, X_train, X_test, y_train, y_test)



    features = pd.read_pickle('data/nn_features')
    # features = pd.read_pickle('classifier_features')

    features_test = pd.read_pickle('data/nn_test_features')
    # features_test = pd.read_pickle('classifier_test_features')



    #features = create_lag_columns(features)
    #features_test = create_lag_columns(features_test)
    # -- Use OPTUNA to get best features with xgboost
    # best_set_features, best_trial = get_features_optuna(features_data)

    # -- Plot histogram in order to check trades distribution over time
    matplotlib.use('TkAgg')
    plt.figure(figsize=(15, 4))
    plt.hist(features.index, bins=25)

    #X = features_data.drop(columns = ['signal'])


    mask_train = (features['signal'] != -1)
    mask_test = (features_test['signal'] != -1)

    features = features.loc[mask_train].reset_index(drop=True)
    features_test = features_test.loc[mask_test].reset_index(drop=True) # TODO: VER SI REALMENTE ESTÁ BIEN HACER ESTO

    # X = features[best_feat] # ESTO PARA CLASSIFIER

    X = features['features']
    X = [mat[ALL_feature_names].to_numpy() for mat in X]
    X = np.stack(X)

    y = features['signal']

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.14, shuffle=False)

    # X_t = features_test[best_feat]
    X_t = features_test['features']
    X_t = [mat[ALL_feature_names].to_numpy() for mat in X_t]
    X_t = np.stack(X_t)

    y_t = features_test['signal']


    # train_and_eval_classifier(X_tr, X_val, y_tr, y_val, X_t, y_t)

    train_and_eval_CNN_LSTM(X_tr, X_val, y_tr, y_val, X_t, y_t)

