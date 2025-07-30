import json
import numpy as np
from collections import deque
import pandas as pd

from backtesting.indicators.KAMA import KAMA
from backtesting.indicators.dmi_diffs import IncrementalDMI_Diff
from backtesting.indicators.ema_diff import IncrementalEMA_Diff
from backtesting.indicators.fisherTransform import FisherTransform
from backtesting.indicators.kalman_trend import KalmanTrend
from backtesting.indicators.macd import IncrementalMACD
from backtesting.indicators.priceDerivatives import PriceDerivatives
from backtesting.indicators.rsi import IncrementalRSI
from backtesting.indicators.savitzkyGolay import SavitzkyGolayTrend
from backtesting.indicators.stochasticRSI import StochasticRSI
from backtesting.indicators.superSmoother import EhlersSuperSmoother
from backtesting.indicators.superTrendATR import SuperTrendATR
from backtesting.indicators.welford import IncrementalVolatility
from backtesting.indicators.zeroLagEMA import ZeroLagEMA
from backtesting.strategy.services_strategy import simulate_forex_closure_ibkr, simulate_forex_short_closure_ibkr, \
    group_trades_by_final_exec, get_trade_exposure, get_trade_exposure_LEVERAGE
from backtesting.strategy.trading_strats import simulate_trade_partial, simulate_short_trade_partial, \
    simulate_short_trade_partial_training, simulate_trade_partial_training

# Strategy parameters
INITIAL_CAPITAL = 70000  # Example initial capital in USD
LEVERAGE = 10            # Leverage of 1:10
SPREAD = 0.00003  # spread de 0.3 pips en EUR/USD (típico IB)
COMM_P_SIDE = 0.00002 # Minimum commission IBKR
SLIPPAGE = 0.00001

global none_flag


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


def safe_value(val):
    global none_flag
    if val is None:
        none_flag = True
        return np.nan
    return val




# Build indicators
def initialize_indicators():
    kf_long  = KalmanTrend(delta=1e-4, R=0.005)
    kf_short = KalmanTrend(delta=1e-4, R=0.0008)

    # RSI (corto y largo)
    rsi_fast = IncrementalRSI(period=5)     # 25 min
    rsi_slow = IncrementalRSI(period=7)    # 70 min

    # Volatility
    vol_fast = IncrementalVolatility(period=3)   # 30 min
    vol_slow = IncrementalVolatility(period=9)  # 90 min

    # EMA diff
    ema_fast_slow = IncrementalEMA_Diff(2, 6)   # 20 min vs 60 min

    # MACD
    macd = IncrementalMACD(2, 6, 3)             # fast, slow, signal

    # DMI
    dmi = IncrementalDMI_Diff(period=7)         # 70 min

    # Savitzky-Golay trend
    sg = SavitzkyGolayTrend(window_length=5, polyorder=2, delta=1.0)

    # Ehlers Super-Smoother
    ss_fast = EhlersSuperSmoother(filt_length=3)   # 25 min
    ss_slow = EhlersSuperSmoother(filt_length=7)  # 70 min

    # Zero-Lag EMA
    zl_ema = ZeroLagEMA(period=2)                  # 20 min

    # KAMA
    kama = KAMA(period=3)                          # 30 min

    # Stochastic RSI
    srsi = StochasticRSI(rsi_period=3, stoch_period=6)

    # Fisher Transform
    fisher = FisherTransform(period=4)             # 40 min

    # SuperTrend ATR
    sup_tr = SuperTrendATR(period=4, multiplier=3)

    # Derivadas de precio
    deriv = PriceDerivatives()

    return {**locals()}


# Buffers of features
def initialize_buffers():
    lbs = 15
    buffers = {}
    # Buffers
    for name in feature_names:
        buffers[name+'_buf'] = deque(maxlen=lbs)

    return buffers


def get_nn_train_data_through_strategy(model_nn, X_train, y_train):
    global none_flag

    test_predictions = model_nn.predict(X_train).flatten()



    with open('../data/predictions/fullDataPredictionsLongShort.json', 'w') as file:
        json.dump(test_predictions.tolist(), file)



    with open('../data/predictions/fullDataPredictionsLongShort.json', 'r') as f:
        predictionsList = json.load(f)


    open_trades = []
    trades_to_close = []
    features = []


    d = initialize_indicators()
    b = initialize_buffers()
    prev_close = None


    #Prepare timestamp to set session hour
    y_train['timestamp'] = pd.to_datetime(y_train.index, utc=True)
    y_train['timestamp_local'] = y_train['timestamp'].dt.tz_convert('Europe/Madrid')

    y_train['session'] = y_train['timestamp_local'].apply(set_session)

    exposure = 125000

    for i in range(len(y_train)):

        if i == 0: continue

        none_flag = False

        # 1) UPDATE AND CLOSE TRADES ALREADY OPENED
        updated_trades = []

        for trade in open_trades:

            intervals_passed = i - trade['entry_index']


            price_long, still_open_long, reason_long = simulate_trade_partial_training(trade['prices'], trade['open_price_mid'], intervals_passed)
            price_short, still_open_short, reason_short = simulate_short_trade_partial_training(trade['prices'], trade['open_price_mid'], intervals_passed)

            if not still_open_long and not trade['flag_finish_long']:
                trade['flag_finish_long'] = True
                trade['long_final_price'] = price_long
                trade['reason_long'] = reason_long
                if not trade['flag_finish_short']: trade['first_trade_finished'] = 'long'

            if not still_open_short and not trade['flag_finish_short']:
                trade['flag_finish_short'] = True
                trade['short_final_price'] = price_short
                trade['reason_short'] = reason_short
                if not trade['flag_finish_long']: trade['first_trade_finished'] = 'short'

            if (not trade['flag_finish_long'] or not trade['flag_finish_short']) and intervals_passed < 6:
                updated_trades.append(trade)
            else:

                # Store data long operation
                if intervals_passed == 6:
                    if not trade['flag_finish_long']:
                        trade['long_final_price'] = price_long
                        trade['reason_long'] = reason_long
                        trade['flag_finish_long'] = True

                    if not trade['flag_finish_short']:
                        trade['short_final_price'] = price_short
                        trade['reason_short'] = reason_short
                        trade['flag_finish_short'] = True

            if trade['flag_finish_short'] and trade['flag_finish_long']:
                trades_to_close.append(trade)


        # Iterate over trades that must be closed
        for trade in trades_to_close:

            out_signal = -1
            # In this case one of the stop losses / target factor was activated
            if trade['reason_long'] == 'target_factor' and trade['reason_short'] == 'target_factor':
                if trade['first_trade_finished'] == 'long': out_signal = 0
                if trade['first_trade_finished'] == 'short': out_signal = 1

            elif trade['reason_long'] == 'target_factor':
                out_signal = 0 # LONG signal
            elif trade['reason_short'] == 'target_factor':
                out_signal = 1

            if trade['reason_long'] == 'open':
                # These are those trades that exceed an hour
                pnl_long = simulate_forex_closure_ibkr([exposure], [trade['open_price_original']], trade['long_final_price'], SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)
                # For trades larger than an hour get minimal gains
                if pnl_long > 36: out_signal = 0

            if trade['reason_short'] == 'open':
                # These are those trades that exceed an hour
                pnl_short = simulate_forex_short_closure_ibkr([exposure], [trade['open_price_original']], trade['short_final_price'], SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)
                # For trades larger than an hour get minimal gains
                if pnl_short > 36: out_signal = 1


            features.append({
                'entry_index': trade['entry_index'],
                'time_entry': trade['time_entry'],
                'trade_signal': trade['trade_signal_pred'],
                'features':  trade['seq_features'],
                #'session': trade['session'],
                'signal': out_signal
            })





        # 3) CONTINUE IF WE HAVE ALREADY OPENED A TRADE
        open_trades = updated_trades
        trades_to_close = []

        # 4) EVALUATE AND OPEN NEW TRADE
        open_price        = y_train.iloc[i]['Open']
        prices           = y_train.iloc[i]['Prices']
        time_entry       = y_train.iloc[i].name
        session          = y_train.iloc[i]['session']

        previous_minutes_prices = y_train.iloc[i-1]['Prices']
        previous_minutes_prices = previous_minutes_prices[20:21:2]

        prediction = predictionsList[i]


        for p in previous_minutes_prices:
            # ───────── 1) Tendencia general con filtros Kalman ─────────
            t_kf_l, s_kf_l = d['kf_long'].update(p)    # long-horizon Kalman
            t_kf_s, s_kf_s = d['kf_short'].update(p)   # short-horizon Kalman

            # ───────── Indicadores “fast” y “slow” ─────────
            r_fast  = d['rsi_fast'].update(p)
            r_slow  = d['rsi_slow'].update(p)

            v_fast  = d['vol_fast'].update(p)
            v_slow  = d['vol_slow'].update(p)

            ema_fs  = d['ema_fast_slow'].update(p)
            mcd     = d['macd'].update(p)

            t_sg, s_sg = d['sg'].update(p)

            ss_fast_p  = d['ss_fast'].update(p)
            ss_slow_p  = d['ss_slow'].update(p)

            # Residuos
            res_kf_l = p - t_kf_l
            res_kf_s = p - t_kf_s
            res_sg   = p - t_sg if t_sg is not None else None

            # ───────── Indicadores adicionales ─────────
            zl      = d['zl_ema'].update(p)
            kama_v  = d['kama'].update(p)
            srsi_v  = d['srsi'].update(p)
            fish_v  = d['fisher'].update(p)
            st_v, st_dir = d['sup_tr'].update(high=p, low=p, close=p)
            delta, accel = d['deriv'].update(p)

            # ───────── Rellenar buffers ─────────
            b['kf_slope_fast_buf'].append(s_kf_s)
            b['kf_slope_slow_buf'].append(s_kf_l)
            b['kf_trend_fast_buf'].append(t_kf_s)
            b['kf_trend_slow_buf'].append(t_kf_l)

            b['kf_res_fast_buf'].append(res_kf_s)
            b['kf_res_slow_buf'].append(res_kf_l)

            b['rsi_fast_buf'].append(r_fast)
            b['rsi_slow_buf'].append(r_slow)

            b['vol_fast_buf'].append(v_fast)
            b['vol_slow_buf'].append(v_slow)

            b['macd_buf'].append(mcd)
            b['ema_fast_slow_buf'].append(ema_fs)

            b['sg_trend_buf'].append(t_sg)
            b['sg_slope_buf'].append(s_sg)
            b['sg_res_buf'].append(res_sg)

            b['ss_fast_price_buf'].append(ss_fast_p)
            b['ss_slow_price_buf'].append(ss_slow_p)


            b['zl_ema_buf'].append(zl)
            b['kama_buf'].append(kama_v)
            b['srsi_buf'].append(srsi_v)
            b['fisher_buf'].append(fish_v)
            b['sup_tr_buf'].append(st_v)

            b['delta_buf'].append(delta)
            b['accel_buf'].append(accel)

            prev_close = p


        # ───────── Build features sequence ─────────
        seq_features = np.column_stack([
            b['kf_slope_fast_buf'], b['kf_slope_slow_buf'],
            b['kf_trend_fast_buf'], b['kf_trend_slow_buf'],
            b['kf_res_fast_buf'],   b['kf_res_slow_buf'],
            b['rsi_fast_buf'],      b['rsi_slow_buf'],
            b['vol_fast_buf'],      b['vol_slow_buf'],
            b['macd_buf'],          b['ema_fast_slow_buf'],
            b['sg_trend_buf'],      b['sg_slope_buf'],      b['sg_res_buf'],
            b['ss_fast_price_buf'], b['ss_slow_price_buf'],
            b['zl_ema_buf'],        b['kama_buf'],          b['srsi_buf'],
            b['fisher_buf'],        b['sup_tr_buf'],
            b['delta_buf'],         b['accel_buf']
        ])



        seq_df = pd.DataFrame(seq_features, columns=feature_names)

        # Use constant exposure for these operations

        new_trade = {
            'entry_index': i,
            'open_price_mid': open_price,
            'open_price_original': open_price,
            'prices': prices,
            'time_entry': time_entry,
            'flag_finish_long': False,
            'flag_finish_short': False,
            'trade_signal_pred': prediction,
            'seq_features' : seq_df,
            'session' : session
        }


        if prediction > 0.1 and not seq_df.isnull().values.any() :
            open_trades.append(new_trade)


    return features



def set_session(hora_local):
    h = hora_local.hour
    if 2 <= h < 10:
        return 'Asia'
    elif 10 <= h < 15:
        return 'London'
    elif 15 <= h < 23:
        return 'NewYork'
    else:
        return 'Other'