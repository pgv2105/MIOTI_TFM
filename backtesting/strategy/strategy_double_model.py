import copy
import json
import logging
import random
from collections import deque
import joblib

import numpy as np
import pandas as pd
from numba import njit
import datetime

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
from backtesting.strategy.trading_strats import simulate_trade_partial, simulate_short_trade_partial

# Strategy parameters
INITIAL_CAPITAL = 70000  # Example initial capital in USD
LEVERAGE = 10            # Leverage of 1:10
SPREAD = 0.00003  # spread de 0.3 pips en EUR/USD (típico IB)
COMM_P_SIDE = 0.00002 # Minimum commission IBKR
SLIPPAGE = 0.00001

feature_names = [
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

def backtest_strategy_monkeys(model_operation, model_direction, X_val, y_val, type_simulation):




    balance = INITIAL_CAPITAL
    balance_history = [balance]

    open_trades = []
    executed_trades = []


    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    O   = y_val['Open'  ].to_numpy()


    n_total = y_val.shape[0]
    n_zeros = n_total // 2           # 50%
    n_ones = n_total // 4            # 25%
    n_twos = n_total - n_zeros - n_ones  # El resto (25%)

    # Crear los valores
    predictions = [0] * n_zeros + [1] * n_ones + [2] * n_twos


    for i in range(len(y_val)):

        if i == 0: continue


        jump_trade = False

        # 1) UPDATE AND CLOSE TRADES ALREADY OPENED
        updated_trades = []
        trades_to_close = []

        for trade in open_trades:
            intervals_passed = i - trade['entry_index']

            type_trade = trade['type']

            if type_trade == 'long':
                price, still_open, idx_final_price = simulate_trade_partial(trade['prices'], trade['open_price_mid'], intervals_passed)
                pnl = simulate_forex_closure_ibkr([trade['exposure']], [trade['open_price_original']], price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)
            else:  # 'short'
                price, still_open, idx_final_price = simulate_short_trade_partial(trade['prices'], trade['open_price_mid'], intervals_passed)
                pnl = simulate_forex_short_closure_ibkr([trade['exposure']], [trade['open_price_original']], price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)


            trade['current_price'] = price
            trade['current_pnl']   = pnl


            if still_open and intervals_passed < 6:
                updated_trades.append(trade)
            else:

                # We keep open the operation with a random choice
                #keep_open = random.choices([0, 1], weights=[1, 2])[0]
                keep_open = 0 if i % 3 == 0 else 1
                if keep_open:
                    trade['entry_index'] = i
                    trade['open_price_mid'] = trade['prices'][idx_final_price]
                    new_segment_data = 20 * (6 - intervals_passed)
                    trade['prices'] = trade['prices'][idx_final_price:] + y_val.iloc[i]['Prices'][new_segment_data:new_segment_data + idx_final_price] # trade['entry_index'] + interval_passed
                    updated_trades.append(trade)
                    jump_trade = True
                    continue


                # Close real trades which are not reopened
                if not trade['is_paper_trade'] :
                    trade['executed_price'] = price
                    trade['interval_passed'] = intervals_passed
                    trades_to_close.append(trade)


                # Skip next trade only if we have just opened a new position
                #jump_trade = jump_trade or jump_flag_aux



        # 2) CLOSE TRADES AND UPDATE BALANCE
        if len(trades_to_close) > 0:
            # Group trades by close price, we can close all this trades at the same time
            grouped = group_trades_by_final_exec(trades_to_close)
            for (price, ttype), group in grouped.items():
                exposures = [trade['exposure'] for trade in group]
                open_prices = [trade['open_price_original'] for trade in group]

                if ttype == 'long':
                    pnl = simulate_forex_closure_ibkr(exposures, open_prices, price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)
                else:
                    pnl = simulate_forex_short_closure_ibkr(exposures, open_prices, price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)

                balance += pnl

                mean_pnl_group = pnl/len(group)
                for t in group:
                    executed_trades.append({
                        'entry_index': t['entry_index'],
                        'orig_entry_index': t['orig_entry_index'],
                        'trade_type': t['type'],
                        'executed_price': price,
                        'duration': t['interval_passed'] * 10,
                        'pnl_baseline': mean_pnl_group,
                        'pnl_dynamic': balance,
                        'win_baseline': 1 if pnl > 0 else 0,
                        'win_dynamic': 0,
                        'entry_time': t['time_entry'],
                        'exit_time': y_val.iloc[i].name
                    })

                    logging.info(f"[TRADE - {t['orig_entry_index']}- ] Entry: {t['time_entry']} @ {t['open_price_original']}, Exit: {y_val.iloc[i].name} @ {price}, "
                                 f"PnL Baseline: {mean_pnl_group:.2f}, Final Balance: {balance:.2f}")

        # 3) CONTINUE IF WE HAVE ALREADY OPENED A TRADE
        open_trades = updated_trades
        open_price        = O[i]


        # 4) EVALUATE AND OPEN NEW TRADE
        prices           = y_val.iloc[i]['Prices']
        time_entry       = y_val.iloc[i].name


        if jump_trade:
            balance_history.append(balance)
            continue



        # -- 2) -- Model signal & clear history
        trade_value = predictions[i]

        # -- 3) -- Get operation type long/short
        final_signal = False
        entry_price = 0
        type_trade = 'Null'

        exposure = 0

        if trade_value != 0:

            type_trade = 'long' if trade_value == 1 else 'short'

            exposure = get_trade_exposure(open_price, open_trades, balance, type_trade, LEVERAGE)
            final_signal  = True

        new_trade = {
            'entry_index': i,
            'orig_entry_index': i,
            'open_price_mid': open_price,
            'open_price_original': open_price,
            'prices': prices,
            'current_price': entry_price,
            'exposure': exposure,
            'current_pnl': 0,
            'type': type_trade,
            'time_entry': time_entry,
            'is_paper_trade': False #is_paper
        }


        if final_signal and exposure != 0 and len(open_trades) < 6 :
            open_trades.append(new_trade)

        # 4) Record balance
        balance_history.append(balance)

    return balance_history, executed_trades


def backtest_strategy_double_model(model_operation, model_direction, X_val, y_val, type_simulation):

    scaler = joblib.load('data/models/standard_scaler_model_directionality_1.pkl')

    test_predictions = model_operation.predict(X_val).flatten()

    predictionsList = test_predictions.tolist()



    balance = INITIAL_CAPITAL
    balance_history = [balance]

    open_trades = []
    executed_trades = []

    d = initialize_indicators()
    b = initialize_buffers()


    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    P   = y_val['Prices'].to_numpy()
    O   = y_val['Open'  ].to_numpy()

    for i in range(len(y_val)):

        if i == 0: continue


        jump_trade = False

        # 1) UPDATE AND CLOSE TRADES ALREADY OPENED
        updated_trades = []
        trades_to_close = []

        for trade in open_trades:
            intervals_passed = i - trade['entry_index']

            type_trade = trade['type']

            if type_trade == 'long':
                price, still_open, idx_final_price = simulate_trade_partial(trade['prices'], trade['open_price_mid'], intervals_passed)
                pnl = simulate_forex_closure_ibkr([trade['exposure']], [trade['open_price_original']], price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)
            else:  # 'short'
                price, still_open, idx_final_price = simulate_short_trade_partial(trade['prices'], trade['open_price_mid'], intervals_passed)
                pnl = simulate_forex_short_closure_ibkr([trade['exposure']], [trade['open_price_original']], price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)


            trade['current_price'] = price
            trade['current_pnl']   = pnl


            if still_open and intervals_passed < 6:
                updated_trades.append(trade)
            else:
                jump_flag_aux = False


                # TODO: ESTO PODRÍA MEJORARSE, ACTUALMENTE COGEMOS PREDICCION ANTERIOR PERO HABRÍA QUE CREAR UNA NUEVA
                #  PARA EL MOMENTO EXACTO (ES DIFÍCIL CON LOS DATOS QUE TENEMOS)
                if (idx_final_price + 1) % 20 == 0:
                    prediction = predictionsList[i]
                else:
                    prediction = predictionsList[i-1]


                if prediction > 0.65:

                    previous_minutes_prices = trade['prices'][(20*(intervals_passed - 1)) : idx_final_price : 2]

                    testing_buffers = copy.deepcopy(b)
                    copied_indicators = copy.deepcopy(d)

                    prediction_dir = get_directionality_prediction(testing_buffers, copied_indicators, previous_minutes_prices, scaler, model_direction)

                    if type_trade ==  prediction_dir:
                        trade['entry_index'] = i
                        trade['open_price_mid'] = trade['prices'][idx_final_price]
                        new_segment_data = 20 * (6 - intervals_passed)
                        trade['prices'] = trade['prices'][idx_final_price:] + P[i][new_segment_data:new_segment_data + idx_final_price] # trade['entry_index'] + interval_passed
                        updated_trades.append(trade)
                        jump_flag_aux = True


                # Close real trades which are not reopened
                if not trade['is_paper_trade'] and not jump_flag_aux:
                    trade['executed_price'] = price
                    trade['interval_passed'] = intervals_passed
                    trades_to_close.append(trade)


                # Skip next trade only if we have just opened a new position
                jump_trade = jump_trade or jump_flag_aux


        # 2) CLOSE TRADES AND UPDATE BALANCE
        if len(trades_to_close) > 0:
            # Group trades by close price, we can close all this trades at the same time
            grouped = group_trades_by_final_exec(trades_to_close)
            for (price, ttype), group in grouped.items():
                exposures = [trade['exposure'] for trade in group]
                open_prices = [trade['open_price_original'] for trade in group]

                if ttype == 'long':
                    pnl = simulate_forex_closure_ibkr(exposures, open_prices, price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)
                else:
                    pnl = simulate_forex_short_closure_ibkr(exposures, open_prices, price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)

                balance += pnl

                mean_pnl_group = pnl/len(group)
                for t in group:
                    executed_trades.append({
                        'entry_index': t['entry_index'],
                        'trade_type': t['type'],
                        'executed_price': price,
                        'duration': t['interval_passed'] * 10,
                        'pnl_baseline': mean_pnl_group,
                        'pnl_dynamic': balance,
                        'win_baseline': 1 if pnl > 0 else 0,
                        'win_dynamic': 0,
                        'entry_time': t['time_entry'],
                        'exit_time': y_val.iloc[i].name
                    })

                    logging.info(f"[TRADE - {t['type']}- ] Entry: {t['time_entry']} @ {t['open_price_original']}, Exit: {y_val.iloc[i].name} @ {price}, "
                                 f"PnL Baseline: {mean_pnl_group:.2f}, Final Balance: {balance:.2f}")

        # 3) CONTINUE IF WE HAVE ALREADY OPENED A TRADE
        open_trades = updated_trades
        open_price        = O[i]






        # 4) EVALUATE AND OPEN NEW TRADE
        prices           = P[i]
        time_entry       = y_val.iloc[i].name
        previous_minutes_prices = P[i-1]
        previous_minutes_prices = previous_minutes_prices[1:21:2]


        add_prices_to_buffer(b, d, previous_minutes_prices)




        if jump_trade:
            balance_history.append(balance)
            continue



        # Construir secuencia de features para el modelo
        seq_features = np.column_stack([
            b['kf_slope_short_buf'], b['kf_slope_long_buf'], b['kf_trend_short_buf'], b['kf_trend_long_buf'],
            b['kf_res_short_buf'], b['kf_res_long_buf'], b['rsi10_buf'], b['rsi24_buf'], b['vol15_buf'], b['vol50_buf'],
            b['macd_buf'], b['ema12_26_buf'], b['sg_trend15_buf'], b['sg_slope15_buf'], b['sg_res15_buf'],
            b['ss10_price_buf'], b['ss50_price_buf'], b['ss10_slope_buf'], b['ss50_slope_buf'],
            b['zl_ema5_buf'], b['kama10_buf'], b['srsi5_buf'], b['fisher10_buf'], b['sup_tr7_buf'],
            b['delta_buf'], b['accel_buf']
        ])
        seq_df = pd.DataFrame(seq_features, columns=feature_names)

        # -- 2) -- Model signal & clear history
        perform_trade = predictionsList[i]

        # -- 3) -- Get operation type long/short
        final_signal = False
        entry_price = 0
        type_trade = 'Null'

        exposure = 0

        if perform_trade > 0.65 and not seq_df.isnull().values.any():

            seq_features = scaler.transform(seq_features)
            feat = seq_features.reshape(1,seq_df.shape[0], seq_df.shape[1])

            type_operation = model_direction.predict(feat).flatten()

            type_trade = 'long' if type_operation < 0.4942 else 'short'

            exposure = get_trade_exposure(open_price, open_trades, balance, type_trade, LEVERAGE)
            final_signal  = True

        new_trade = {
            'entry_index': i,
            'open_price_mid': open_price,
            'open_price_original': open_price,
            'prices': prices,
            'current_price': entry_price,
            'exposure': exposure,
            'current_pnl': 0,
            'type': type_trade,
            'time_entry': time_entry,
            'is_paper_trade': False #is_paper
        }


        if final_signal and exposure != 0 and len(open_trades) < 6 :
            open_trades.append(new_trade)

        # 4) Record balance
        balance_history.append(balance)

    return balance_history, executed_trades


def backtest_strategy_test_no_model(model, x_val, y_val):

    balance = INITIAL_CAPITAL
    balance_history = [balance]

    open_trades = []
    executed_trades = []

    # Initialize Kalman filter to evaluate trends
    kf_long_period = KalmanTrend(delta=1e-4, R=0.01)

    # RSI parameters
    rsi_calc = IncrementalRSI(period=14)


    for i in range(len(y_val)):

        if i == 0: continue


        jump_trade = False

        # 1) UPDATE AND CLOSE TRADES ALREADY OPENED
        updated_trades = []
        trades_to_close = []

        for trade in open_trades:
            intervals_passed = i - trade['entry_index']

            type_trade = trade['type']

            if type_trade == 'long':
                price, still_open, idx_final_price = simulate_trade_partial(trade['prices'], trade['open_price_mid'], intervals_passed)
                pnl = simulate_forex_closure_ibkr([trade['exposure']], [trade['open_price_original']], price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)
            else:  # 'short'
                price, still_open, idx_final_price = simulate_short_trade_partial(trade['prices'], trade['open_price_mid'], intervals_passed)
                pnl = simulate_forex_short_closure_ibkr([trade['exposure']], [trade['open_price_original']], price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)


            trade['current_price'] = price
            trade['current_pnl']   = pnl


            if still_open and intervals_passed < 6:
                updated_trades.append(trade)
            else:
                jump_flag_aux = False

                # Check if we have to open new trade
                trend_long, slope_long = kf_long_period.test_kalman_result(price)
                rsi_value = rsi_calc.test_rsi_result(price)


                if rsi_value is not None and ( # TODO: MIRAR BIEN TENGO QUE TENER EN CUENTA DESPLAZAMIENTO FOR
                        (rsi_value < 60 and slope_long > 0 and type_trade == 'long') or
                        (rsi_value > 75 and type_trade == 'short')):

                    trade['entry_index'] = i
                    trade['open_price_mid'] = trade['prices'][idx_final_price]
                    new_segment_data = 20 * (6 - intervals_passed)
                    trade['prices'] = trade['prices'][idx_final_price:] + y_val.iloc[i]['Prices'][new_segment_data:new_segment_data + idx_final_price] # trade['entry_index'] + interval_passed
                    updated_trades.append(trade)
                    jump_flag_aux = True


                # Close real trades which are not reopened
                if not trade['is_paper_trade'] and not jump_flag_aux:
                    trade['executed_price'] = price
                    trade['interval_passed'] = intervals_passed
                    trades_to_close.append(trade)


                # Skip next trade only if we have just opened a new position
                jump_trade = jump_trade or jump_flag_aux


        # 2) CLOSE TRADES AND UPDATE BALANCE
        if len(trades_to_close) > 0:
            # Group trades by close price, we can close all this trades at the same time
            grouped = group_trades_by_final_exec(trades_to_close)
            for (price, ttype), group in grouped.items():
                exposures = [trade['exposure'] for trade in group]
                open_prices = [trade['open_price_original'] for trade in group]

                if ttype == 'long':
                    pnl = simulate_forex_closure_ibkr(exposures, open_prices, price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)
                else:
                    pnl = simulate_forex_short_closure_ibkr(exposures, open_prices, price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)

                balance += pnl

                mean_pnl_group = pnl/len(group)
                for t in group:
                    executed_trades.append({
                        'entry_index': t['entry_index'],
                        'trade_type': t['type'],
                        'executed_price': price,
                        'duration': t['interval_passed'] * 10,
                        'pnl_baseline': mean_pnl_group,
                        'slope_pips': t['slope_pips'],
                        'pnl_dynamic': balance,
                        'win_baseline': 1 if pnl > 0 else 0,
                        'win_dynamic': 0,
                        'entry_time': t['time_entry'],
                        'exit_time': y_val.iloc[i].name
                    })

        # 3) CONTINUE IF WE HAVE ALREADY OPENED A TRADE
        open_trades = updated_trades
        open_price        = y_val.iloc[i]['Open']

        if jump_trade:
            # Update kalman and rsi filters and go to next operation
            _, _ = kf_long_period.update(open_price)
            _ = rsi_calc.update(open_price)
            balance_history.append(balance)
            continue

        # 4) EVALUATE AND OPEN NEW TRADE
        open_price        = y_val.iloc[i]['Open']
        prices           = y_val.iloc[i]['Prices']
        time_entry       = y_val.iloc[i].name


        # -- 1) -- Get current trend to evaluate if we should perform the trade
        # -- Get tendency trend with KALMAN filter and check if it is positive/medium
        trend_long, slope_long = kf_long_period.update(open_price)

        # RSI tendency
        rsi_value = rsi_calc.update(open_price)


        # -- 3) -- Get operation type long/short
        final_signal = False
        entry_price = 0
        type_trade = 'Null'

        exposure = 0

        if rsi_value is not None and rsi_value < 60 and slope_long > 0: #
            type_trade = 'long'
            exposure = get_trade_exposure_LEVERAGE(open_price, open_trades, balance, type_trade, LEVERAGE)
            final_signal  = True
        elif rsi_value is not None and rsi_value > 75: # slope_long < 0 and
            type_trade = 'short'
            exposure = get_trade_exposure_LEVERAGE(open_price, open_trades, balance, type_trade, LEVERAGE)
            final_signal = True


        new_trade = {
            'entry_index': i,
            'open_price_mid': open_price,
            'open_price_original': open_price,
            'prices': prices,
            'current_price': entry_price,
            'exposure': exposure,
            'current_pnl': 0,
            'type': type_trade,
            'time_entry': time_entry,
            'slope_pips': slope_long * 1e4,
            'is_paper_trade': False #is_paper
        }


        if final_signal and exposure is not 0 and len(open_trades) < 6 :
            open_trades.append(new_trade)

        # 4) Record balance
        balance_history.append(balance)

    return balance_history, executed_trades


# Indicators
def initialize_indicators():
    kf_long  = KalmanTrend(delta=1e-4, R=0.01)
    kf_short = KalmanTrend(delta=1e-4, R=0.001)

    rsi10    = IncrementalRSI(period=100)
    rsi24    = IncrementalRSI(period=240)
    vol15    = IncrementalVolatility(period=150)
    vol50    = IncrementalVolatility(period=500)
    ema12_26 = IncrementalEMA_Diff(120, 260)
    macd     = IncrementalMACD(120, 260, 90)
    dmi14    = IncrementalDMI_Diff(period=140)
    sg15     = SavitzkyGolayTrend(window_length=151, polyorder=2, delta=1.0)
    ss10     = EhlersSuperSmoother(filt_length=61)
    ss50     = EhlersSuperSmoother(filt_length=420)

    zl_ema5  = ZeroLagEMA(period=5)
    kama10   = KAMA(period=10)
    srsi5    = StochasticRSI(rsi_period=5, stoch_period=14)
    fisher10 = FisherTransform(period=10)
    sup_tr7  = SuperTrendATR(period=7, multiplier=3)
    deriv    = PriceDerivatives()

    return {**locals()}


# Buffers de features
def initialize_buffers():
    lbs = 60
    buffers = {}
    # Buffers
    for name in ['kf_slope_short','kf_slope_long','kf_trend_short','kf_trend_long',
                 'kf_res_short','kf_res_long','rsi10','rsi24','vol15','vol50',
                 'macd','ema12_26','sg_trend15','sg_slope15','sg_res15',
                 'ss10_price','ss50_price','ss10_slope','ss50_slope','zl_ema5','kama10','srsi5',
                 'fisher10','sup_tr7','delta','accel']:
        buffers[name+'_buf'] = deque(maxlen=lbs)

    return buffers


def add_prices_to_buffer(b, d, previous_minutes_prices):

    # -- 1) -- Get current trend to evaluate if we should perform the trade
    for p in previous_minutes_prices:

        # Old indicators
        t_kf_l, s_kf_l = d['kf_long'].update(p)
        t_kf_s, s_kf_s = d['kf_short'].update(p)

        r10 = d['rsi10'].update(p)
        r24 = d['rsi24'].update(p)
        v15 = d['vol15'].update(p)
        v50 = d['vol50'].update(p)
        e12_26 = d['ema12_26'].update(p)
        mcd = d['macd'].update(p)
        t_sg15, s_sg15 = d['sg15'].update(p)
        ss10_p = d['ss10'].update(p)
        ss50_p = d['ss50'].update(p)

        res_kf_l = p - t_kf_l
        res_kf_s = p - t_kf_s
        res_sg15 = p - t_sg15 if t_sg15 is not None else None


        # New indicators
        zl5 = d['zl_ema5'].update(p)
        kama_v = d['kama10'].update(p)
        srsi_v = d['srsi5'].update(p)
        fish_v = d['fisher10'].update(p)
        st_v, st_dir = d['sup_tr7'].update(high=p, low=p, close=p)
        delta, accel = d['deriv'].update(p)


        # Fill buffers
        b['kf_slope_short_buf'].append(s_kf_s)
        b['kf_slope_long_buf'].append(s_kf_l)
        b['kf_trend_short_buf'].append(t_kf_s)
        b['kf_trend_long_buf'].append(t_kf_l)

        b['kf_res_short_buf'].append(res_kf_s)
        b['kf_res_long_buf'].append(res_kf_l)
        b['rsi10_buf'].append(r10)
        b['rsi24_buf'].append(r24)
        b['vol15_buf'].append(v15)
        b['vol50_buf'].append(v50)
        b['macd_buf'].append(mcd)
        b['ema12_26_buf'].append(e12_26)
        b['sg_trend15_buf'].append(t_sg15)
        b['sg_slope15_buf'].append(s_sg15)
        b['sg_res15_buf'].append(res_sg15)
        b['ss10_price_buf'].append(ss10_p)
        b['ss50_price_buf'].append(ss50_p)
        b['ss10_slope_buf'].append(ss10_p - d['ss10'].prev1 if hasattr(d['ss10'], 'prev1') else 0)
        b['ss50_slope_buf'].append(ss50_p - d['ss50'].prev1 if hasattr(d['ss50'], 'prev1') else 0)


        # New buffers
        b['zl_ema5_buf'].append(zl5)
        b['kama10_buf'].append(kama_v)
        b['srsi5_buf'].append(srsi_v)
        b['fisher10_buf'].append(fish_v)
        b['sup_tr7_buf'].append(st_v)
        b['delta_buf'].append(delta)
        b['accel_buf'].append(accel)




def get_directionality_prediction(b, d, previous_minutes_prices, scaler, model_direction):

    # -- 1) -- Get current trend to evaluate if we should perform the trade
    for p in previous_minutes_prices:

        # Old indicators
        t_kf_l, s_kf_l = d['kf_long'].update(p)
        t_kf_s, s_kf_s = d['kf_short'].update(p)

        r10 = d['rsi10'].update(p)
        r24 = d['rsi24'].update(p)
        v15 = d['vol15'].update(p)
        v50 = d['vol50'].update(p)
        e12_26 = d['ema12_26'].update(p)
        mcd = d['macd'].update(p)
        t_sg15, s_sg15 = d['sg15'].update(p)
        ss10_p = d['ss10'].update(p)
        ss50_p = d['ss50'].update(p)

        res_kf_l = p - t_kf_l
        res_kf_s = p - t_kf_s
        res_sg15 = p - t_sg15 if t_sg15 is not None else None


        # New indicators
        zl5 = d['zl_ema5'].update(p)
        kama_v = d['kama10'].update(p)
        srsi_v = d['srsi5'].update(p)
        fish_v = d['fisher10'].update(p)
        st_v, st_dir = d['sup_tr7'].update(high=p, low=p, close=p)
        delta, accel = d['deriv'].update(p)


        # Fill buffers
        b['kf_slope_short_buf'].append(s_kf_s)
        b['kf_slope_long_buf'].append(s_kf_l)
        b['kf_trend_short_buf'].append(t_kf_s)
        b['kf_trend_long_buf'].append(t_kf_l)

        b['kf_res_short_buf'].append(res_kf_s)
        b['kf_res_long_buf'].append(res_kf_l)
        b['rsi10_buf'].append(r10)
        b['rsi24_buf'].append(r24)
        b['vol15_buf'].append(v15)
        b['vol50_buf'].append(v50)
        b['macd_buf'].append(mcd)
        b['ema12_26_buf'].append(e12_26)
        b['sg_trend15_buf'].append(t_sg15)
        b['sg_slope15_buf'].append(s_sg15)
        b['sg_res15_buf'].append(res_sg15)
        b['ss10_price_buf'].append(ss10_p)
        b['ss50_price_buf'].append(ss50_p)
        b['ss10_slope_buf'].append(ss10_p - d['ss10'].prev1 if hasattr(d['ss10'], 'prev1') else 0)
        b['ss50_slope_buf'].append(ss50_p - d['ss50'].prev1 if hasattr(d['ss50'], 'prev1') else 0)


        # New buffers
        b['zl_ema5_buf'].append(zl5)
        b['kama10_buf'].append(kama_v)
        b['srsi5_buf'].append(srsi_v)
        b['fisher10_buf'].append(fish_v)
        b['sup_tr7_buf'].append(st_v)
        b['delta_buf'].append(delta)
        b['accel_buf'].append(accel)

    seq_features = np.column_stack([
        b['kf_slope_short_buf'], b['kf_slope_long_buf'], b['kf_trend_short_buf'], b['kf_trend_long_buf'],
        b['kf_res_short_buf'], b['kf_res_long_buf'], b['rsi10_buf'], b['rsi24_buf'], b['vol15_buf'], b['vol50_buf'],
        b['macd_buf'], b['ema12_26_buf'], b['sg_trend15_buf'], b['sg_slope15_buf'], b['sg_res15_buf'],
        b['ss10_price_buf'], b['ss50_price_buf'], b['ss10_slope_buf'], b['ss50_slope_buf'],
        b['zl_ema5_buf'], b['kama10_buf'], b['srsi5_buf'], b['fisher10_buf'], b['sup_tr7_buf'],
        b['delta_buf'], b['accel_buf']
    ])
    seq_df = pd.DataFrame(seq_features, columns=feature_names)

    seq_features = scaler.transform(seq_features)
    feat = seq_features.reshape(1,seq_df.shape[0], seq_df.shape[1])

    type_operation = model_direction.predict(feat).flatten()

    type_trade = 'long' if type_operation < 0.4942 else 'short'

    return type_trade