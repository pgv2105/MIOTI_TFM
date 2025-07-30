
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib


from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from .get_features import compute_rsi, compute_rel_changes, compute_ema_slope, compute_macd, compute_obv, \
    compute_sma_slope, compute_rel_max_min, compute_roc, compute_tr_and_atr, compute_bbands, compute_adx, \
    compute_candle_count, compute_volatility



plt.ion()  # Plot mode interactive
# ---------------------------------------------------------------------------------------------------

ALL_FEATURES_STOCKS = ['Variation', 'Volatility', 'Max_Change%', 'Min_Change%', 'Volume','RSI', 'EMA_12','EMA_26',
                        'MACD','MACD_Signal','OBV','SMA_10','ROC','Rel_Max_Min','TR','ATR','Upper_BB','Middle_BB','Lower_BB']

ALL_FEATURES_FOREX = ['Variation', 'Volatility', 'Max_Change%', 'Min_Change%', 'RSI', 'EMA_12', 'EMA_26', 'MACD',
                  'MACD_Signal', 'SMA_10', 'ROC', 'Rel_Max_Min', 'TR', 'ATR', 'Upper_BB', 'Middle_BB',
                  'Lower_BB','ADX','Bullish_count','Bearish_count','log_return','Volatility2']

# ----------------------------------------------------------------------------------------------------

STD_FEATURES_STOCKS = ['Variation', 'RSI', 'EMA_12','EMA_26','MACD','MACD_Signal','OBV','SMA_10','ROC','Rel_Max_Min','TR',
                      'ATR','Upper_BB','Middle_BB','Lower_BB']

STD_FEATURES_FOREX = ['Variation', 'RSI', 'EMA_12','EMA_26','MACD','MACD_Signal','SMA_10','ROC','Rel_Max_Min','TR',
                      'ATR','Upper_BB','Middle_BB','Lower_BB','ADX','Bullish_count','Bearish_count','log_return','Volatility2']

# -----------------------------------------------------------------------------------------------------

LOG_FEATURES_STOCKS = ['Volatility', 'Max_Change%', 'Min_Change%', 'Volume']

LOG_FEATURES_FOREX = ['Volatility', 'Max_Change%', 'Min_Change%']

# ------------------------------------------------------------------------------------------------------

def get_and_align_features():

    # -- LOAD DATA FROM CSV (Previous loaded from INTERACTIVE BROKERS)
    input_data = pd.read_csv('data/forex/INPUT_EURUSD_intervals_30_60_3YEARS.csv')
    output_data = pd.read_csv('data/forex/OUTPUT_EURUSD_intervals_30_60_3YEARS.csv')

    return input_data, output_data



def preprocess_features(input_data, output_data):


    # ---------------------------- GET SPECIFIC INPUT FEATURES ------------------------------------ #

    # Set variation in (%) between each stock
    input_data["Variation"] = (input_data['Close'] - input_data['Open']) / input_data['Open'] * 100
    # Set volatility
    input_data['Volatility'] = (input_data['High'] - input_data['Low']) / input_data['Open'] * 100


    #  -- GET SPECIFIC FEATURES AND TECHNICAL INDICATORS
    #  -  STOCKS/FOREX
    input_data = compute_rel_changes(input_data) # Relative maximum and minimum changes (%)
    input_data = compute_rsi(input_data)
    input_data = compute_ema_slope(input_data)
    input_data = compute_macd(input_data)
    # input_data = compute_obv(input_data)  # On-Balance Volume (OBV)
    input_data = compute_sma_slope(input_data)  # Simple Mobile Average (SMA)
    input_data = compute_rel_max_min(input_data)  # Relation between max and min in session
    input_data = compute_roc(input_data)  # Rate of Change  (ROC)
    input_data = compute_tr_and_atr(input_data) # True Range (TR) and Average True Range (ATR)
    input_data = compute_bbands(input_data)  # Bollinger bands
    #  - FOREX
    input_data = compute_adx(input_data)
    input_data = compute_candle_count(input_data)  # Calculate bearish and bullish candles
    input_data = compute_volatility(input_data)



    features = ALL_FEATURES_FOREX
    X = input_data[features]

    # -- APPLY DATA TRANSFORMATION FOR INPUT FEATURES

    # Initialize the scaler
    scaler = StandardScaler()

    # Apply standard scaler only to variation feature
    features = STD_FEATURES_FOREX
    X[features] = scaler.fit_transform(X[features])

    # We are going to initially use logarithmic transform in order to slightly reduce the left and right bias
    # for Max, Min and volatility. For volume, we use log transform in order to handle higher volumes
    features = LOG_FEATURES_FOREX
    X.loc[X['Min_Change%'] <= -1, 'Min_Change%'] = -0.99 # Avoid negative logarithms
    X[features] = np.log1p(X[features])

    # Clear nan rows that appears after applying logarithmic transformation
    nan_rows = X.isnull().any(axis=1)
    X = X[~nan_rows]

    # Then we standard normAlize data so we have all features in the same scale
    features = LOG_FEATURES_FOREX
    X[features] = scaler.fit_transform(X[features])

    # We delete outliers if necessary
    X['Variation'] = X['Variation'].clip(lower=X['Variation'].quantile(0.01),
        upper=X['Variation'].quantile(0.99))

    # X['Volume'] = X['Volume'].squeeze().clip(lower=X['Volume'].squeeze().quantile(0.01),
    #    upper=X['Volume'].squeeze().quantile(0.99)
    #)




    # ---------------------------- GET SPECIFIC OUTPUT FEATURE ------------------------------------ #

    # -- GET NEW FEATURES FOR OUTPUT DATA
    output_data["High_%"] = (output_data["High"] - input_data["Close"]) / input_data["Close"] * 100
    threshold = 0.04
    output_data['Binary_Output'] = output_data['High_%'].apply(lambda value: 0 if value < threshold else 1)

    featuresOUT = ['Binary_Output']
    Y = output_data[featuresOUT]

    Y = Y[~nan_rows]


    # STORE PREPROCESSED FEATURES SO WE DON'T HAVE TO PERFORM SAME OPERATIONS LATER
    X.to_csv('features_INPUT_LSTM_intervals_30_60_3YEARS.csv', index=False)
    Y.to_csv('features_OUTPUT_LSTM_intervals_30_60_3YEARS.csv', index=False)

    return X, Y


def evaluate_dataset(X, Y):

    # Check distribution features
    X.hist(bins=30, figsize=(10, 8))
    plt.tight_layout()
    plt.show()

    # Check class balance
    print(Y.value_counts(normalize=True))

    # Check correlation between features
    correlation_matrix = X.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()

    # Check feature-output correlation
    for col in X.columns:
        plt.scatter(X[col], Y)
        plt.title(f'Relación entre {col} y Output')
        plt.xlabel(col)
        plt.ylabel('Output')
        plt.show()


# Create window lengths as input for the LSTM model
def create_sequences(data, target, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])  # Aggregate X intervals to the window
        y.append(target[i + sequence_length - 1:i + sequence_length])  # Set output to the new window
    return np.array(X), pd.concat(y, axis=0)


def show_results(y_test, y_pred):
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    matplotlib.use('TkAgg')
    # Display the confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='Blues')

    plt.show()

    '''
    
         # ---------------------------
    # Graficar la evolución del entrenamiento
    # ---------------------------
    # Gráfica de la pérdida (loss)
    plt.figure(figsize=(14, 4))
    plt.plot(history.history['loss'], label='Pérdida entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida validación')
    plt.title('Evolución de la pérdida')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Gráfica de la exactitud (accuracy)
    plt.figure(figsize=(14, 4))
    plt.plot(history.history['accuracy'], label='Exactitud entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Exactitud validación')
    plt.title('Evolución de la exactitud')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Gráfica de Precision
    plt.figure(figsize=(14, 4))
    plt.plot(history.history['precision'], label='Precision entrenamiento')
    plt.plot(history.history['val_precision'], label='Precision validación')
    plt.title('Evolución de la precision')
    plt.xlabel('Época')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

    # Gráfica de Recall
    plt.figure(figsize=(14, 4))
    plt.plot(history.history['recall'], label='Recall entrenamiento')
    plt.plot(history.history['val_recall'], label='Recall validación')
    plt.title('Evolución del recall')
    plt.xlabel('Época')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()

    
    '''

