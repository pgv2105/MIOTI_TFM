import joblib
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pickle

from .get_features import compute_rsi, compute_rel_changes, compute_ema_slope, compute_macd, compute_obv, \
    compute_sma_slope, compute_rel_max_min, compute_roc, compute_tr_and_atr, compute_bbands, compute_adx, \
    compute_candle_count, compute_volatility, compute_ichimoku, compute_stoch, compute_psar, compute_candle_body_stats, \
    compute_dir_vol_skew, compute_linreg_slope, compute_ema_distance, compute_consecutive_counts

from services_preprocessing.services import create_sequences, evaluate_dataset

sequence_length = 5

SPREAD = 0.00002
SLIPPAGE = 0.00001
COMMISSION = 0.00004
# --------------------------------------------------------------------------------------------------- 'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARaf_0.02_0.2',
#

ALL_FEATURES_STOCKS = ['Variation', 'Volatility', 'Max_Change%', 'Min_Change%', 'Volume','RSI', 'EMA_12','EMA_26',
                        'MACD','MACD_Signal','OBV','SMA_10','ROC','Rel_Max_Min','TR','ATR','Upper_BB','Middle_BB','Lower_BB']

ALL_FEATURES_FOREX = ['Variation', 'Volatility', 'Max_Change%', 'Min_Change%', 'RSI', 'EMA_12', 'EMA_26', 'MACD',
                  'MACD_Signal', 'SMA_10', 'ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ROC', 'Rel_Max_Min', 'TR', 'ATR', 'Upper_BB', 'Middle_BB',
                  'Lower_BB','ADX','Bullish_count','Bearish_count', 'PSAR','PSAR_trend', 'STOCHk_7_3_3', 'STOCHd_7_3_3','log_return','Volatility2']

ALL_FEATURES_FOREX_MULTIVARIATE = ['Max_Change%', 'Min_Change%', 'RSI', 'EMA_12', 'EMA_26', 'MACD',
                      'MACD_Signal', 'SMA_10', 'ISA_9', 'ISB_26', 'ROC', 'TR', 'ATR', 'Upper_BB', 'Middle_BB',
                      'Lower_BB','ADX','Bullish_count','Bearish_count', 'PSAR','PSAR_trend', 'Body_Ratio', 'Body_Sign', 'Consec_Up', 'Consec_Down','EMA_Dist', 'LinReg_Slope_14', 'Dir_Vol_Skew',
                       'RSI_5_21_Diff', 'MACD_Hist','EMA20_50_Dist' ]


# ----------------------------------------------------------------------------------------------------

STD_FEATURES_STOCKS = ['Variation', 'RSI', 'EMA_12','EMA_26','MACD','MACD_Signal','OBV','SMA_10','ROC','Rel_Max_Min','TR',
                      'ATR','Upper_BB','Middle_BB','Lower_BB']

STD_FEATURES_FOREX = ['Variation', 'RSI', 'EMA_12','EMA_26','MACD','MACD_Signal','SMA_10','ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26','ROC','Rel_Max_Min','TR',
                      'ATR','Upper_BB','Middle_BB','Lower_BB','ADX','Bullish_count','Bearish_count', 'PSAR','PSAR_trend', 'STOCHk_7_3_3', 'STOCHd_7_3_3',
                      'log_return','Volatility2']

# -----------------------------------------------------------------------------------------------------

LOG_FEATURES_STOCKS = ['Volatility', 'Max_Change%', 'Min_Change%', 'Volume']

LOG_FEATURES_FOREX = ['Volatility', 'Max_Change%', 'Min_Change%']

# ------------------------------------------------------------------------------------------------------



'''
    For preprocessing, data is separated in three different groups, this separation is performed in order to use all 
    our data available. Then each group will contain data from different 30 mins intervals. For example data in each 
    group will be is timestamps similar as:
    
    - Group A: 10:00, 10:30, 11:00, 11:30 ....
    - Group B: 10:10, 10:40, 11:10, 11:40 ....
    - Group C: 10:20, 10:50, 11:20, 11:50 ....
    
'''
# Function to split a DataFrame into n cyclic groups
def split_into_groups(df, n_groups=3):
    return [df.iloc[i::n_groups].copy() for i in range(n_groups)]

# Function to reassemble groups by interleaving the rows
def restack_groups(groups):
    # Convert each group to a NumPy array and trim to the minimum length among them
    arrays = [group.values for group in groups]
    min_len = min(len(arr) for arr in arrays)
    trimmed = [arr[:min_len] for arr in arrays]

    # Create interleaved indices from all groups:
    # For each group, take the first min_len index values and convert them to a NumPy array.
    idx_arrays = [group.index[:min_len].to_numpy() for group in groups]
    # Stack the index arrays along a new axis (resulting shape: [min_len, n_groups])
    stacked_idx = np.stack(idx_arrays, axis=1)
    # Reshape the stacked indices to a 1D array, effectively interleaving them.
    final_index = stacked_idx.reshape(-1)

    # Stack the trimmed arrays: resulting in an array with shape (min_len, n_groups, n_columns)
    stacked = np.stack(trimmed, axis=1)
    n_rows, n_groups, n_cols = stacked.shape

    # Reshape to interleave the rows from each group
    reshaped = stacked.reshape(-1, n_cols)

    # Build and return the final DataFrame using the interleaved index and the original columns
    return pd.DataFrame(reshaped, index=final_index, columns=groups[0].columns)


def preprocess_LSTM_data(input_data, output_data):

    # === STEP 1: Split into 3 groups and compute features separately ===

    # Split both input_data and output_data into 3 groups
    groups_input = split_into_groups(input_data, n_groups=3)
    groups_output = split_into_groups(output_data, n_groups=3)

    # Sort indexes and apply feature computation on each group
    for i in range(3):
        groups_input[i].sort_index(inplace=True)
        groups_output[i].sort_index(inplace=True)

        groups_input[i], groups_output[i] = compute_features_multivariate(groups_input[i], groups_output[i])

    # Reassemble the DataFrames by interleaving the rows
    input_data = restack_groups(groups_input)
    output_data = restack_groups(groups_output)

    # Normalize and transform both DataFrames using the same scalers
    input_data, output_data = normalize_and_transform(input_data, output_data)

    # === STEP 2: Split again to create time window sequences ===

    # Re-split the normalized DataFrame into 3 groups
    groups_input = split_into_groups(input_data, n_groups=3)
    groups_output = split_into_groups(output_data, n_groups=3)

    # Apply the sequence creation function to each group
    for i in range(3):
        groups_input[i], groups_output[i] = create_sequences(groups_input[i], groups_output[i], sequence_length)

    # Trim all groups to the same minimum length
    min_len = min(len(g) for g in groups_input)
    groups_input = [g[:min_len] for g in groups_input]
    # groups_output = [g[:min_len] for g in groups_output]

    # Stack the sequences from each group (along axis 1) and reshape to the final form
    final_input = np.stack(groups_input, axis=1).reshape(-1, sequence_length, len(ALL_FEATURES_FOREX_MULTIVARIATE))
    #final_output = np.stack(groups_output, axis=1).reshape(-1, 1, 1)
    final_output = restack_groups(groups_output)

    # Save the processed dataset
    data = {"dataset_X": final_input, "dataset_Y": final_output}
    with open("training_data_model_PLUS_MULTIVARIATE.pkl", "wb") as f:
        pickle.dump(data, f)

    # Finally, return the final data
    return final_input, final_output




def compute_features(input_data, output_data):


    # Set variation in (%) between each stock
    input_data["Variation"] = (input_data['Close'] - input_data['Open']) / input_data['Open'] * 100
    # Set volatility
    input_data['Volatility'] = (input_data['High'] - input_data['Low']) / input_data['Open'] * 100

    input_data.set_index('Start', inplace = True)
    output_data.set_index('Start', inplace = True)

    #  -  STOCKS/FOREX
    input_data = compute_rel_changes(input_data)  # Relative maximum and minimum changes (%)
    input_data = compute_rsi(input_data)
    input_data = compute_ema_slope(input_data)
    input_data = compute_macd(input_data)
    input_data = compute_ichimoku(input_data)
    input_data = compute_stoch(input_data)
    input_data = compute_psar(input_data)
    # input_data = compute_obv(input_data)  # On-Balance Volume (OBV)
    input_data = compute_sma_slope(input_data)  # Simple Mobile Average (SMA)
    input_data = compute_rel_max_min(input_data)  # Relation between max and min in session
    input_data = compute_roc(input_data)  # Rate of Change  (ROC)
    input_data = compute_tr_and_atr(input_data)  # True Range (TR) and Average True Range (ATR)
    input_data = compute_bbands(input_data)  # Bollinger bands
    #  - FOREX
    input_data = compute_adx(input_data)
    input_data = compute_candle_count(input_data)  # Calculate bearish and bullish candles
    input_data = compute_volatility(input_data)


    # ---------------------------- GET SPECIFIC OUTPUT FEATURE ------------------------------------ #

    # -- GET NEW FEATURES FOR OUTPUT DATA
    # output_data["High_%"] = (output_data["High"] - input_data["Close"]) / (input_data["Close"]) * 100
    output_data["Low_%"] = (output_data["Low"] - input_data["Close"]) / (input_data["Close"]) * 100

    threshold = - 0.051 # 0.051
    output_data['Binary_Output'] = output_data['Low_%'].apply(lambda value: 1 if value < threshold else 0)
    #output_data['Binary_Output'] = output_data['High_%'].apply(lambda value: 0 if value < threshold else 1)


    featuresOUT = ['Binary_Output']
    # featuresOUT = ['Open', 'Close', 'High','Low', 'Prices']
    output_data = output_data[featuresOUT]
    input_data = input_data[ALL_FEATURES_FOREX]


    return input_data, output_data



def compute_features_multivariate(input_data, output_data):


    # Set variation in (%) between each stock
    #input_data["Variation"] = (input_data['Close'] - input_data['Open']) / input_data['Open'] * 100
    # Set volatility
    #input_data['Volatility'] = (input_data['High'] - input_data['Low']) / input_data['Open'] * 100

    input_data.set_index('Start', inplace = True)
    output_data.set_index('Start', inplace = True)

    #  -  STOCKS/FOREX
    input_data = compute_rel_changes(input_data)  # Relative maximum and minimum changes (%)
    input_data = compute_rsi(input_data)
    input_data = compute_ema_slope(input_data)
    input_data = compute_macd(input_data)
    input_data = compute_ichimoku(input_data)
    #input_data = compute_stoch(input_data)
    input_data = compute_psar(input_data)
    # input_data = compute_obv(input_data)  # On-Balance Volume (OBV)
    input_data = compute_sma_slope(input_data)  # Simple Mobile Average (SMA)
    #input_data = compute_rel_max_min(input_data)  # Relation between max and min in session
    input_data = compute_roc(input_data)  # Rate of Change  (ROC)
    input_data = compute_tr_and_atr(input_data)  # True Range (TR) and Average True Range (ATR)
    input_data = compute_bbands(input_data)  # Bollinger bands
    #  - FOREX
    input_data = compute_adx(input_data)
    input_data = compute_candle_count(input_data)  # Calculate bearish and bullish candles
    #input_data = compute_volatility(input_data)
    # NEW FEATURES -- MULTIVARIATE
    input_data = compute_candle_body_stats(input_data)
    input_data = compute_consecutive_counts(input_data)
    input_data = compute_ema_distance(input_data)
    input_data = compute_linreg_slope(input_data)
    input_data = compute_dir_vol_skew(input_data)

    # EXTRA INDICATORS
    input_data["RSI_5"]  = compute_rsi(input_data, window=5)["RSI"]
    input_data["RSI_21"] = compute_rsi(input_data, window=21)["RSI"]
    input_data["RSI_5_21_Diff"] = input_data["RSI_5"] - input_data["RSI_21"]

    # 2. MACD recalibrado para intradía: EMA(5) vs EMA(13)
    input_data["EMA_5"]  = input_data["Close"].ewm(span=5,  adjust=False).mean()
    input_data["EMA_13"] = input_data["Close"].ewm(span=13, adjust=False).mean()
    input_data["MACD_5_13"]    = input_data["EMA_5"] - input_data["EMA_13"]
    input_data["Signal_5"]     = input_data["MACD_5_13"].ewm(span=5, adjust=False).mean()
    input_data["MACD_Hist"]    = input_data["MACD_5_13"] - input_data["Signal_5"]

    # 3. EMA largo vs corto en ATR (para tendencia mayor)
    #    EMA20 (10h) vs EMA50 (25h) en M30 → corresponde a 10h y 25h
    input_data["EMA_20"]  = input_data["Close"].ewm(span=20, adjust=False).mean()
    input_data["EMA_50"]  = input_data["Close"].ewm(span=50, adjust=False).mean()
    input_data["EMA20_50_Dist"] = (
                                          input_data["EMA_20"] - input_data["EMA_50"]
                                  ) / input_data["ATR"].replace(0, np.nan)

    # ---------------------------- GET SPECIFIC OUTPUT FEATURE ------------------------------------ #

    # -- GET NEW FEATURES FOR OUTPUT DATA
    output_data["High_%"] = (output_data["High"] - input_data["Close"]) / input_data["Close"] * 100
    output_data["Low_%"]  = (output_data["Low"]  - input_data["Close"]) / input_data["Close"] * 100

    threshold = 0.075

    # 1 if High_% ≥ +threshold  or  Low_% ≤ −threshold
    output_data["Binary_Output"] = (
            (output_data["High_%"] >=  threshold) |
            (output_data["Low_%"]  <= -threshold)
    ).astype(int)

    '''
    labels = build_first_hit_label(output_data,
                                   up_th=0.00058,      # ajusta a tu volatilidad media
                                   down_th=-0.00058)

    output_data = pd.concat([output_data, labels], axis=1)

    featuresOUT = ['FirstHit_Label']
    '''

    featuresOUT = ['Open', 'Close', 'High','Low', 'Prices','Binary_Output']
    #featuresOUT = ['Binary_Output']
    output_data = output_data[featuresOUT]
    input_data = input_data[ALL_FEATURES_FOREX_MULTIVARIATE]


    return input_data, output_data


def normalize_and_transform(input_data,output_data):

    # DATA MUST BE SEPARATED IN TRAINING AND TEST IN ORDER TO AVOID DATA LEAKAGE (use same separation as in models.py)
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, shuffle=False)

    length = int(X_train.shape[0]*0.8)
    #for train_index, val_index in tscv.split(X_train):
    X_tr, X_val = X_train[:length], X_train[length:]
    y_tr, y_val = y_train[:length], y_train[length:]

    # Scale all training data and save scalers
    is_train = True
    X_train, y_train = normalize_data_multivariate(X_tr, y_tr, is_train)

    # Scale test data with previous saved scalers to ensure data integrity
    X_combined = pd.concat([X_val, X_test], axis=0)
    y_combined = pd.concat([y_val, y_test], axis=0)
    is_train = False
    X_comb, y_comb = normalize_data_multivariate(X_combined, y_combined, is_train)

    input_data = pd.concat([X_train, X_comb], axis=0)
    output_data = pd.concat([y_train, y_comb], axis=0)


    return input_data, output_data


def normalize_data(input_data, output_data, is_train):

    features = ALL_FEATURES_FOREX

    input_data = input_data[features]

    # Clear nan rows that appears after applying logarithmic transformation
    nan_rows = input_data.isnull().any(axis=1)
    input_data = input_data[~nan_rows]
    nan_rows = nan_rows.to_numpy()
    output_data = output_data[~nan_rows]

    #cols_to_log = ['Volatility', 'Max_Change%', 'EMA_26', 'ISA_9', 'ISB_26', 'ITS_9','IKS_26', 'SMA_10','TR','ATR','log_return']
    cols_to_log = ['Max_Change%', 'EMA_26', 'ISA_9', 'ISB_26', 'ITS_9','IKS_26', 'SMA_10','TR','ATR']

    input_data[cols_to_log] = np.log1p(input_data[cols_to_log])  # log(x+1)

    # Important differentiate if it is train or test data
    if is_train:
         # -- If TRAIN DATA: create and save scalers

        # Apply RobustScaler in order to reduce outliers impact
        robust_scaler = RobustScaler()
        input_data = robust_scaler.fit_transform(input_data)
        joblib.dump(robust_scaler, 'robust_scaler_PLUS_INVERSE.pkl')

        # Apply MinMaxScaler and normalize between 0 and 1 (important for LSTM)
        minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        input_data = minmax_scaler.fit_transform(input_data)
        joblib.dump(minmax_scaler, 'minmax_scaler_PLUS_INVERSE.pkl')

    else:
        # -- IF TEST DATA: apply previously saved scalers
        robust_scaler_loaded = joblib.load('robust_scaler_PLUS_INVERSE.pkl')
        minmax_scaler_loaded = joblib.load('minmax_scaler_PLUS_INVERSE.pkl')

        input_data = robust_scaler_loaded.transform(input_data)
        input_data = minmax_scaler_loaded.transform(input_data)


    return pd.DataFrame(input_data), pd.DataFrame(output_data)


def normalize_data_multivariate(input_data, output_data, is_train):

    features = ALL_FEATURES_FOREX_MULTIVARIATE

    input_data = input_data[features]

    # Clear nan rows that appears after applying logarithmic transformation
    nan_rows = input_data.isnull().any(axis=1)
    input_data = input_data[~nan_rows]
    nan_rows = nan_rows.to_numpy()
    output_data = output_data[~nan_rows]

    cols_to_log = ['Max_Change%', 'EMA_26', 'ISA_9', 'ISB_26', 'SMA_10','TR','ATR']

    input_data[cols_to_log] = np.log1p(input_data[cols_to_log])  # log(x+1)

    # Important differentiate if it is train or test data
    if is_train:
        # -- If TRAIN DATA: create and save scalers

        # Apply RobustScaler in order to reduce outliers impact
        robust_scaler = RobustScaler()
        input_data = robust_scaler.fit_transform(input_data)
        joblib.dump(robust_scaler, 'robust_scaler_PLUS_INVERSE.pkl')

        # Apply MinMaxScaler and normalize between 0 and 1 (important for LSTM)
        minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        input_data = minmax_scaler.fit_transform(input_data)
        joblib.dump(minmax_scaler, 'minmax_scaler_PLUS_INVERSE.pkl')

    else:
        # -- IF TEST DATA: apply previously saved scalers
        robust_scaler_loaded = joblib.load('robust_scaler_PLUS_INVERSE.pkl')
        minmax_scaler_loaded = joblib.load('minmax_scaler_PLUS_INVERSE.pkl')

        input_data = robust_scaler_loaded.transform(input_data)
        input_data = minmax_scaler_loaded.transform(input_data)


    return pd.DataFrame(input_data), pd.DataFrame(output_data)


def build_first_hit_label(
        output_df: pd.DataFrame,
        *,
        up_th: float = 0.00051,      # +0.10 % above the Close
        down_th: float = -0.00051    # –0.10 % below the Close
) -> pd.DataFrame:
    """
    Returns Hit_Up, Hit_Down and FirstHit_Label
    (0 ↓ bearish hit first, 1 no breakout, 2 ↑ bullish hit first).

    Parameters
    ----------
    output_df : DataFrame with columns **'Close'** and **'Prices'**.
                *Prices* must be a list/array of the next 120 future prices.
    up_th     : Relative bullish threshold (e.g. 0.001 = +0.1 %).
    down_th   : Relative bearish threshold (e.g. –0.001 = –0.1 %).
    """
    n = len(output_df)
    hit_up   = np.zeros(n, dtype=int)
    hit_down = np.zeros(n, dtype=int)
    label    = np.zeros(n, dtype=int)         # 1 = no breakout by default

    open_arr  = output_df["Open"].values
    prices_col = output_df["Prices"].values  # object array (lists)

    for i in range(n):
        close_now = open_arr[i]
        future_p  = np.asarray(prices_col[i], dtype=float)  # 120 prices ahead
        if future_p.size == 0:            # safeguard in case of empty lists
            continue

        up_price   = close_now * (1 + up_th)
        down_price = close_now * (1 + down_th)

        # indices of the first occurrence
        up_idx   = np.argmax(future_p >= up_price)   if np.any(future_p >= up_price)   else None
        down_idx = np.argmax(future_p <= down_price) if np.any(future_p <= down_price) else None

        if up_idx is not None and (down_idx is None or up_idx < down_idx):
            hit_up[i] = 1
            label[i]  = 1        # bullish hit first (long)
        elif down_idx is not None and (up_idx is None or down_idx < up_idx):
            hit_down[i] = 1
            label[i]    = 2      # bearish hit first (short)
        # else: neither threshold was hit → label remains 1

    return pd.DataFrame(
        {
            "Hit_Up": hit_up,
            "Hit_Down": hit_down,
            "FirstHit_Label": label,
        },
        index=output_df.index,
    )
