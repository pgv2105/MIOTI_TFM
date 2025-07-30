import numpy as np
import pandas_ta as ta
import pandas as pd
from pandas import DataFrame


def compute_rsi(input_data, window=7):
    delta = input_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss

    input_data['RSI'] = 100 - (100 / (1 + rs))
    return input_data


def compute_rel_changes(input_data):
    input_data['Max_Change%'] = (input_data['High'] - input_data['Open']) / input_data['Open'] * 100
    input_data['Min_Change%'] = (input_data['Low'] - input_data['Open']) / input_data['Open'] * 100
    return input_data


def compute_ema_slope(input_data):
    input_data['EMA_12'] = input_data['Close'].ewm(span=5, adjust=False).mean()
    input_data['EMA_26'] = input_data['Close'].ewm(span=10, adjust=False).mean()
    return input_data

def compute_macd(input_data):
    macd = ta.macd(input_data['Close'], fast=6, slow=13, signal=5)
    input_data['MACD'] = macd['MACD_6_13_5']
    input_data['MACD_Signal'] = macd['MACDs_6_13_5']
    return input_data


def compute_obv(input_data):
    input_data['OBV'] = ta.obv(input_data['Close'], input_data['Volume'])
    return input_data


def compute_sma_slope(input_data):
    input_data['SMA_10'] = ta.sma(input_data['Close'], timeperiod=5)  # SMA de 10 períodos
    return input_data

def compute_roc(input_data):
    input_data['ROC'] = ta.roc(input_data['Close'], timeperiod=5)
    return input_data


def compute_rel_max_min(input_data):
    input_data['Rel_Max_Min'] = (input_data['Close'] - input_data['Low']) / (input_data['High'] - input_data['Low'])
    return input_data


def compute_tr_and_atr(input_data):
    input_data['TR'] = ta.true_range(input_data['High'], input_data['Low'], input_data['Close'])
    input_data['ATR'] = ta.atr(input_data['High'], input_data['Low'], input_data['Close'], length=7)
    return input_data


def compute_bbands(input_data):
    bb = ta.bbands(input_data['Close'], length=10, std=2)

    input_data['Upper_BB'] = bb['BBU_10_2.0']
    input_data['Middle_BB'] = bb['BBM_10_2.0']
    input_data['Lower_BB'] = bb['BBL_10_2.0']
    return input_data


def compute_adx(input_data, period=7):
    input_data['ADX'] = ta.adx(input_data['High'], input_data['Low'], input_data['Close'], timeperiod=period)['ADX_14']
    return input_data


# Calculate bullish/bearish candles
def compute_candle_count(df, window=10):
    df['Bullish_count'] = df['Close'].rolling(window).apply(lambda x: pd.Series(x).diff().gt(0).sum(), raw=True)
    df['Bearish_count'] = df['Close'].rolling(window).apply(lambda x: pd.Series(x).diff().lt(0).sum(), raw=True)
    return df

def compute_volatility(df, window=6):
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility2'] = df['log_return'].rolling(window=window).std()
    return df




def compute_ichimoku(input_data):
    df = pd.DataFrame()

    df['high'] = input_data['High']
    df['low'] = input_data['Low']
    df['close'] = input_data['Close']

    df.reset_index(drop=True, inplace=True)

    ichimoku = df.ta.ichimoku(conversion_line_period=7,
                            base_line_period=14,
                            lagging_span_period=28,
                            displacement=14)

    input_data['ISA_9'] = ichimoku[0]['ISA_9'].values
    input_data['ISB_26'] = ichimoku[0]['ISB_26'].values
    input_data['ITS_9'] = ichimoku[0]['ITS_9'].values
    input_data['IKS_26'] = ichimoku[0]['IKS_26'].values
    input_data['ICS_26'] = ichimoku[0]['ICS_26'].values

    return input_data

def compute_stoch(input_data):

    # Crea una copia con las columnas necesarias
    df = input_data[['High', 'Low', 'Close']].copy()

    # Calcula el Estocástico con dropna desactivado (por si acaso)
    stoch = df.ta.stoch(k=7, d=3, smooth_k=3, dropna=False)

    # Reindexa para asegurar que ambos DataFrames tengan la misma forma e índice
    stoch = stoch.reindex(input_data.index)

    input_data['STOCHk_7_3_3'] = stoch['STOCHk_7_3_3'].values
    input_data['STOCHd_7_3_3'] = stoch['STOCHd_7_3_3'].values

    return input_data

def compute_psar(input_data):
    # Crea una copia con las columnas necesarias
    df = input_data[['High', 'Low', 'Close']].copy()

    psar = df.ta.psar(step=0.03, max=0.15)

    psar = psar.reindex(input_data.index)

    # Unificas el PSAR:
    input_data['PSAR'] = psar['PSARl_0.02_0.2'].combine_first(psar['PSARs_0.02_0.2'])

    # Creas una columna que indique la tendencia:
    input_data['PSAR_trend'] = np.where(psar['PSARl_0.02_0.2'].notna(), 1,
                                np.where(psar['PSARs_0.02_0.2'].notna(), -1, np.nan))

    return input_data


def compute_candle_body_stats(df: DataFrame) -> DataFrame:
    """Add Body_Ratio (|body|/range) and Body_Sign (‑1,0,+1)."""
    body = df["Close"] - df["Open"]
    rng  = (df["High"] - df["Low"]).replace(0, np.nan)
    df["Body_Ratio"] = (body.abs() / rng).fillna(0)
    df["Body_Sign"]  = np.sign(body)
    return df


def compute_consecutive_counts(df: DataFrame, lookback: int = 4) -> DataFrame:
    """Number of upper/lower candles consecutive until t‑1 (máx = lookback)."""
    up   = (df["Close"] > df["Open"])
    down = (df["Close"] < df["Open"])
    df["Consec_Up"]   = up.groupby((up == 0).cumsum()).cumcount().clip(0, lookback)
    df["Consec_Down"] = down.groupby((down == 0).cumsum()).cumcount().clip(0, lookback)
    return df


def compute_ema_distance(df: DataFrame, length: int = 14) -> DataFrame:
    """Distance to EMA normalized for ATR"""
    ema = df["Close"].ewm(span=length, adjust=False).mean()
    atr = df["ATR"] if "ATR" in df.columns else (df["High"] - df["Low"]).rolling(10).mean()
    df["EMA_Dist"] = (df["Close"] - ema) / atr.replace(0, np.nan)
    return df


def compute_linreg_slope(df: DataFrame, window: int = 11) -> DataFrame:
    """Slope of regression (pips per candle) over window."""
    idx = np.arange(window)
    roll = df["Close"].rolling(window)
    slope = (roll.apply(lambda y: np.polyfit(idx, y, 1)[0], raw=False))
    df[f"LinReg_Slope_14"] = slope
    return df


def compute_dir_vol_skew(df: pd.DataFrame, *, window: int = 6) -> pd.DataFrame:
    """
    Añade al DataFrame la columna `Dir_Vol_Skew_{window}` que mide
    el sesgo direccional de la volatilidad en la ventana de longitud `window`.

    Parámetros
    ----------
    df : pd.DataFrame
        Debe contener al menos la columna "Close", con precios en float64.
    window : int
        Número de barras (periodos) sobre las que se calcula el rolling.
        Si cada barra es de 30 min, window=6 cubre 3 h de datos.

    Retorna
    -------
    df : pd.DataFrame
        Con la nueva columna `Dir_Vol_Skew_{window}` agregada.
    """
    # 1) Calculamos retornos porcentuales
    ret = df["Close"].pct_change()

    # 2) Desviaciones de retornos positivos y negativos en rolling window
    #    Se usa min_periods=2 para que no haya toda una ventana NaN
    pos_std = (
        ret.where(ret > 0)
        .rolling(window=window, min_periods=2)
        .std()
    )
    neg_std = (
        ret.where(ret < 0)
        .rolling(window=window, min_periods=2)
        .std()
    )

    # 3) Sesgo direccional = sigma_pos - sigma_neg
    skew = (pos_std - neg_std).fillna(0.0)

    df[f"Dir_Vol_Skew"] = skew
    return df
