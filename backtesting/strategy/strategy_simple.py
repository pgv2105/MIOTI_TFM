from collections import deque
from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import json
from typing import List

import numpy as np
import pandas as pd

from backtesting.indicators.atr import ATR
from backtesting.indicators.kalman_trend import KalmanTrend
from backtesting.indicators.rsi import IncrementalRSI
from backtesting.strategy.services_strategy import process_confirmation, update_drawdown_freeze, classify_volatility, \
    get_factors_for_market, get_dynamic_params, simulate_forex_closure_ibkr, simulate_forex_short_closure_ibkr, \
    group_trades_by_final_exec
from backtesting.strategy.trading_strats import simulate_trade_partial, simulate_short_trade_partial

# Strategy parameters
INITIAL_CAPITAL = 70000  # Example initial capital in euros
LEVERAGE = 10             # Leverage of 1:10
SPREAD = 0.00003  # spread de 0.2 pips en EUR/USD (típico IB)
COMM_P_SIDE = 0.00002
SLIPPAGE = 0.00001

MAX_OPEN_TRADES = 6
TRADE_DURATION = 6  # Trade duration in intervals (6 intervals = 1 hour, if each interval is 10 minutes)



def backtest_strategy(model, X_val, y_val, type_simulation):


    #test_predictions = model.predict(X_val).flatten()
    #test_predictions_inverse = model.predict(X_val).flatten()

    if type_simulation == 'test':
        # Open and read the JSON file
        #with open('data/predictions/testPredictionsINVERSE.json', 'w') as file:
         #   json.dump(test_predictions_inverse.tolist(), file)

        with open('data/predictions/testPredictions.json', 'r') as f:
            predictionsList = json.load(f)

        with open('data/predictions/testPredictionsINVERSE.json', 'r') as f:
            predictionsListInverse = json.load(f)

    else:
        # Open and read the JSON file
        #with open('data/predictions/valPredictionsINVERSE.json', 'w') as file:
         #   json.dump(test_predictions_inverse.tolist(), file)

        with open('data/predictions/valPredictions.json', 'r') as f:
            predictionsList = json.load(f)

        with open('data/predictions/valPredictionsINVERSE.json', 'r') as f:
            predictionsListInverse = json.load(f)

    balance = INITIAL_CAPITAL
    balance_history = [balance]


    open_trades = []
    executed_trades = []
    recent_closed_trades = []

    # Parameters
    RECENT_TRADES_LIMIT   = 3
    CONFIRM_LOOKBACK      = 30
    SIGNAL_THRESHOLD      = 0.44

    # Freeze state vars
    freeze_flag          = False
    intervals_no_signal  = 0
    peak_balance         = balance
    freeze_min_balance   = balance
    balance_freezed      = balance
    drawdown_freeze      = False
    confirm_freeze       = False
    confirm_pnls: List[float] = []
    slope = 0
    trend_type = 'long'

    # Initialize Kalman filter to evaluate trends
    kf_long_period = KalmanTrend(delta=1e-4, R=0.01)

    # RSI parameters
    RSI_WINDOW = 14
    OVERSOLD  = 35
    OVERBOUGHT = 70
    rsi_calc = IncrementalRSI(period=14)


    for i in range(len(y_val)):

        if i == 0: continue

        TRADE_SIZE = balance / 6
        EXPOSURE   = TRADE_SIZE * LEVERAGE

        MAX_RECENT_LOSS       = - TRADE_SIZE / 82 # Initial = -100
        DRAWDOWN_THRESHOLD    = - TRADE_SIZE / 10  # Initial = -1000
        RECOVERY_AMOUNT       = DRAWDOWN_THRESHOLD / 3.3 # Initial ~ 300

        jump_trade = False

        # 1) Update existing trades
        updated_trades = []
        for trade in open_trades:
            intervals_passed = i - trade['entry_index']

            type_trade = trade['type']

            if type_trade == 'long':
                price, still_open, idx_final_price = simulate_trade_partial(trade['prices'], trade['openPrice_real'], intervals_passed)
                pnl = simulate_forex_closure_ibkr([trade['exposure_eur']], [trade['openPrice_exec']], price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)

            else:  # 'short'
                price, still_open, idx_final_price = simulate_short_trade_partial(trade['prices'], trade['openPrice_real'], intervals_passed)
                pnl = simulate_forex_closure_ibkr([trade['exposure_eur']], [trade['openPrice_exec']], price, SPREAD, SLIPPAGE, COMM_P_SIDE, commission_min = 2)



            trade['current_price'] = price
            trade['current_pnl']   = pnl


            if still_open and intervals_passed < 6:
                updated_trades.append(trade)
            else:
                jump_flag_aux = False
                # Check if we have to open new trade
                #if intervals_passed == 6: # not jump_flag :
                trend_long, slope_long = kf_long_period.test_kalman_result(price)
                rsi_value = rsi_calc.test_rsi_result(price)

                # TODO: ESTO PODRÍA MEJORARSE, ACTUALMENTE COGEMOS PREDICCION ANTERIOR PERO HABRÍA QUE CREAR UNA NUEVA
                #  PARA EL MOMENTO EXACTO (ES DIFÍCIL CON LOS DATOS QUE TENEMOS)
                if (idx_final_price + 1) % 20 == 0:
                    prediction = predictionsList[i]
                    inverse_prediction = predictionsListInverse[i]
                else:
                    prediction = predictionsList[i-1]
                    inverse_prediction = predictionsListInverse[i-1]

                signal = prediction > 0.42
                inverse_signal = inverse_prediction > 0.45

                if rsi_value is not None and ( # TODO: MIRAR BIEN TENGO QUE TENER EN CUENTA DESPLAZAMIENTO FOR
                        (signal and rsi_value < 60 and slope_long > 0 and type_trade == 'long') or
                        (inverse_signal and rsi_value > 75 and type_trade == 'short')):

                    trade['entry_index'] = i
                    trade['openPrice_real'] = trade['prices'][idx_final_price]
                    new_segment_data = 20 * (6 - intervals_passed)
                    trade['prices'] = trade['prices'][idx_final_price:] + y_val.iloc[i]['Prices'][new_segment_data:new_segment_data + idx_final_price] # trade['entry_index'] + interval_passed
                    updated_trades.append(trade)
                    jump_flag_aux = True


                # Close real trades which are not reopened
                if not trade['is_paper_trade'] and not jump_flag_aux:
                    balance += trade['current_pnl']
                    exit_reason = 'Stop loss' if trade['current_pnl'] < -50 else 'Controlled loss' # TODO: DARLE UNA VUELTA
                    executed_trades.append({
                        'entry_index': trade['entry_index'],
                        'trade_type': type_trade,
                        'executed_price': price,
                        'duration': intervals_passed * 10,
                        'pnl_baseline': trade['current_pnl'],
                        'slope_pips': trade['slope_pips'],
                        'pnl_dynamic': balance,
                        'win_baseline': 1 if trade['current_pnl'] > 0 else 0,
                        'win_dynamic': 0,
                        'exit_reason': exit_reason,
                        'entry_time': trade['time_entry'],
                        'exit_time': y_val.iloc[i].name
                    })

                # Skip next trade only if we have just opened a new position
                jump_trade = jump_trade or jump_flag_aux

                # -- EVALUATE LOSSES AND FREEZE FLAGS --

                # Maintain recent closed trades buffer
                recent_closed_trades.append(trade['current_pnl'])
                if len(recent_closed_trades) > RECENT_TRADES_LIMIT:
                    recent_closed_trades.pop(0)

                # Drawdown & recovery logic
                (drawdown_freeze,
                 peak_balance,
                 freeze_min_balance,
                 balance_freezed,
                 confirm_freeze) = update_drawdown_freeze(
                    balance,
                    trade['current_pnl'],
                    peak_balance,
                    freeze_min_balance,
                    balance_freezed,
                    drawdown_freeze,
                    confirm_freeze,
                    DRAWDOWN_THRESHOLD,
                    RECOVERY_AMOUNT
                )

                if len(confirm_pnls) > CONFIRM_LOOKBACK:
                    confirm_pnls.pop(0)

                # If in confirmation phase, process it
                if confirm_freeze:
                    confirm_pnls.append(trade['current_pnl'])
                    (drawdown_freeze,
                     confirm_freeze,
                     confirm_pnls,
                     freeze_min_balance,
                     peak_balance) = process_confirmation(
                        confirm_pnls, confirm_freeze, drawdown_freeze, freeze_min_balance,
                            balance_freezed, balance, peak_balance, CONFIRM_LOOKBACK)


        open_trades = updated_trades
        open_price        = y_val.iloc[i]['Open']

        if jump_trade:
            # Update kalman and rsi filters and go to next operation
            _, _ = kf_long_period.update(open_price)
            _ = rsi_calc.update(open_price)
            balance_history.append(balance)
            continue

        open_price        = y_val.iloc[i]['Open']
        prices           = y_val.iloc[i]['Prices']
        time_entry       = y_val.iloc[i].name


        # 2) -- Get current trend to evaluate if we should perform the trade
        # Get tendency trend with KALMAN filter and check if it is positive/medium
        trend_long, slope_long = kf_long_period.update(open_price)

        # RSI tendency
        rsi_value = rsi_calc.update(open_price)



        # 3) Model signal & clear history
        prediction = predictionsList[i]
        inverse_prediction = predictionsListInverse[i]
        signal = prediction > 0.42
        inverse_signal = inverse_prediction > 0.45

        # 4) -- Get operation type long/short
        final_signal = False
        entry_price = 0
        type_trade = 'Null'

        EXPOSURE_EUR = 0

        if rsi_value is not None and signal and rsi_value < 60 and slope_long > 0: #
            type_trade = 'long'
            entry_price = open_price + SPREAD/2 + SLIPPAGE + COMM_P_SIDE
            EXPOSURE_EUR = EXPOSURE / entry_price
            final_signal  = True
        elif rsi_value is not None and inverse_signal and rsi_value > 75: # slope_long < 0 and
            type_trade = 'short'
            entry_price = open_price - SPREAD/2 - SLIPPAGE - COMM_P_SIDE
            EXPOSURE_EUR = EXPOSURE / entry_price
            final_signal = True


        # Update intervals with no signal to take into account freeze flags
        intervals_no_signal = 0 if final_signal else intervals_no_signal + 1


        if intervals_no_signal >= 3:
            recent_closed_trades.clear()

        freeze_flag = (sum(recent_closed_trades) <= MAX_RECENT_LOSS)

        is_paper = freeze_flag # or drawdown_freeze or confirm_freeze

        new_trade = {
            'entry_index': i,
            'openPrice_real': open_price,
            'openPrice_exec': open_price,
            'prices': prices,
            'current_price': entry_price,
            'exposure_eur': EXPOSURE,
            'current_pnl': 0,
            'type': type_trade,
            'time_entry': time_entry,
            'slope_pips': slope_long * 1e4,
            'is_paper_trade': False #is_paper
        }


        if final_signal and len(open_trades) < MAX_OPEN_TRADES:
            open_trades.append(new_trade)

        # 4) Record balance
        balance_history.append(balance)

    return balance_history, executed_trades



def backtest_no_model_strategy(model, X_val, y_val):

    balance = INITIAL_CAPITAL
    balance_history = [balance]


    open_trades = []
    executed_trades = []
    recent_closed_trades = []

    # Parameters
    RECENT_TRADES_LIMIT   = 3
    CONFIRM_LOOKBACK      = 30

    # Freeze state vars
    freeze_flag          = False
    intervals_no_signal  = 0
    peak_balance         = balance
    freeze_min_balance   = balance
    balance_freezed      = balance
    drawdown_freeze      = False
    confirm_freeze       = False
    confirm_pnls: List[float] = []

    # Initialize Kalman filter to evaluate trends
    kf_long_period = KalmanTrend(delta=1e-4, R=0.01)

    # RSI parameters
    rsi_calc = IncrementalRSI(period=14)


    for i in range(len(y_val)):

        if i == 0: continue

        TRADE_SIZE = balance / 6
        EXPOSURE   = TRADE_SIZE * LEVERAGE

        MAX_RECENT_LOSS       = - TRADE_SIZE / 82 # Initial = -100
        DRAWDOWN_THRESHOLD    = - TRADE_SIZE / 10  # Initial = -1000
        RECOVERY_AMOUNT       = DRAWDOWN_THRESHOLD / 3.3 # Initial ~ 300

        jump_trade = False

        # 1) Update existing trades
        updated_trades = []
        for trade in open_trades:
            intervals_passed = i - trade['entry_index']

            type_trade = trade['type']

            if type_trade == 'long':
                price, still_open, idx_final_price = simulate_trade_partial(trade['prices'], trade['openPrice_real'], intervals_passed)
                final_price  = price  - SPREAD/2 - SLIPPAGE - COMM_P_SIDE
                pnl        = (final_price - trade['openPrice_exec']) * EXPOSURE # * trade['exposure_eur']
            else:  # 'short'
                price, still_open, idx_final_price = simulate_short_trade_partial(trade['prices'], trade['openPrice_real'], intervals_passed)
                final_price  = price  + SPREAD/2 + SLIPPAGE + COMM_P_SIDE
                pnl        = (trade['openPrice_exec'] - final_price) * EXPOSURE # * trade['exposure_eur']


            trade['current_price'] = price
            trade['current_pnl']   = pnl

            if still_open and intervals_passed < 6:
                updated_trades.append(trade)
            else:
                jump_flag_aux = False
                # Check if we have to open new trade
                #if intervals_passed == 6: # not jump_flag :
                trend_long, slope_long = kf_long_period.test_kalman_result(price)
                rsi_value = rsi_calc.test_rsi_result(price)

                if rsi_value is not None and (
                        (rsi_value < 60 and slope_long > 0 and type_trade == 'long') or
                        (rsi_value > 75 and type_trade == 'short')):

                    trade['entry_index'] = i
                    trade['openPrice_real'] = trade['prices'][idx_final_price]
                    new_segment_data = 20 * (6 - intervals_passed)
                    trade['prices'] = trade['prices'][idx_final_price:] + y_val.iloc[i]['Prices'][new_segment_data:new_segment_data + idx_final_price] # trade['entry_index'] + interval_passed
                    updated_trades.append(trade)
                    jump_flag_aux = True


                # Close real trades which are not reopened
                if not trade['is_paper_trade'] and not jump_flag_aux:
                    balance += trade['current_pnl']
                    exit_reason = 'Stop loss' if trade['current_pnl'] < -50 else 'Controlled loss' # TODO: DARLE UNA VUELTA
                    executed_trades.append({
                        'entry_index': trade['entry_index'],
                        'trade_type': type_trade,
                        'executed_price': final_price,
                        'duration': intervals_passed * 10,
                        'pnl_baseline': trade['current_pnl'],
                        'slope_pips': trade['slope_pips'],
                        'pnl_dynamic': balance,
                        'win_baseline': 1 if trade['current_pnl'] > 0 else 0,
                        'win_dynamic': 0,
                        'exit_reason': exit_reason,
                        'entry_time': trade['time_entry'],
                        'exit_time': y_val.iloc[i].name
                    })

                # Skip next trade only if we have just opened a new position
                jump_trade = jump_trade or jump_flag_aux

                # -- EVALUATE LOSSES AND FREEZE FLAGS --

                # Maintain recent closed trades buffer
                recent_closed_trades.append(trade['current_pnl'])
                if len(recent_closed_trades) > RECENT_TRADES_LIMIT:
                    recent_closed_trades.pop(0)

                # Drawdown & recovery logic
                (drawdown_freeze,
                 peak_balance,
                 freeze_min_balance,
                 balance_freezed,
                 confirm_freeze) = update_drawdown_freeze(
                    balance,
                    trade['current_pnl'],
                    peak_balance,
                    freeze_min_balance,
                    balance_freezed,
                    drawdown_freeze,
                    confirm_freeze,
                    DRAWDOWN_THRESHOLD,
                    RECOVERY_AMOUNT
                )

                if len(confirm_pnls) > CONFIRM_LOOKBACK:
                    confirm_pnls.pop(0)

                # If in confirmation phase, process it
                if confirm_freeze:
                    confirm_pnls.append(trade['current_pnl'])
                    (drawdown_freeze,
                     confirm_freeze,
                     confirm_pnls,
                     freeze_min_balance,
                     peak_balance) = process_confirmation(
                        confirm_pnls, confirm_freeze, drawdown_freeze, freeze_min_balance,
                        balance_freezed, balance, peak_balance, CONFIRM_LOOKBACK)


        open_trades = updated_trades
        open_price        = y_val.iloc[i]['Open']

        if jump_trade:
            # Update kalman and rsi filters and go to next operation
            _, _ = kf_long_period.update(open_price)
            _ = rsi_calc.update(open_price)
            balance_history.append(balance)
            continue

        open_price        = y_val.iloc[i]['Open']
        prices           = y_val.iloc[i]['Prices']
        time_entry       = y_val.iloc[i].name


        # 2) -- Get current trend to evaluate if we should perform the trade
        # Get tendency trend with KALMAN filter and check if it is positive/medium
        trend_long, slope_long = kf_long_period.update(open_price)

        # RSI tendency
        rsi_value = rsi_calc.update(open_price)


        # 4) -- Get operation type long/short
        final_signal = False
        entry_price = 0
        type_trade = 'Null'

        EXPOSURE_EUR = 0

        if rsi_value is not None and rsi_value < 60 and slope_long > 0  : #
            type_trade = 'long'
            entry_price = open_price + SPREAD/2 + SLIPPAGE + COMM_P_SIDE
            EXPOSURE_EUR = EXPOSURE / entry_price
            final_signal  = True
        elif rsi_value is not None and rsi_value > 75: # slope_long < 0 and
            type_trade = 'short'
            entry_price = open_price - SPREAD/2 - SLIPPAGE - COMM_P_SIDE
            EXPOSURE_EUR = EXPOSURE / entry_price
            final_signal = True


        # Update intervals with no signal to take into account freeze flags
        intervals_no_signal = 0 if final_signal else intervals_no_signal + 1


        if intervals_no_signal >= 3:
            recent_closed_trades.clear()

        freeze_flag = (sum(recent_closed_trades) <= MAX_RECENT_LOSS)

        is_paper = freeze_flag # or drawdown_freeze or confirm_freeze


        new_trade = {
            'entry_index': i,
            'openPrice_real': open_price,
            'openPrice_exec': entry_price,
            'prices': prices,
            'current_price': entry_price,
            'exposure_eur': EXPOSURE_EUR,
            'current_pnl': 0,
            'type': type_trade,
            'time_entry': time_entry,
            'slope_pips': slope_long * 1e4,
            'is_paper_trade': False #is_paper
        }


        if final_signal and len(open_trades) < MAX_OPEN_TRADES:
            open_trades.append(new_trade)

        # 4) Record balance
        balance_history.append(balance)

    return balance_history, executed_trades


def buy_and_hold(y_test):

    # For demonstration, we assume that y_test (from previous examples) is available
    # and that the closing price is the second column.
    # You can replace this with your actual array of closing prices.
    # Here, we use y_test[:, 1] as the closing prices array.
    closing_prices = y_test['Open'].reset_index()
    closing_prices = closing_prices.drop(columns= ['index'])
    # --- Buy and Hold Strategy Calculation ---
    # Buy at the first closing price
    buy_hold_shares = INITIAL_CAPITAL / closing_prices['Open'].iloc[0]
    # Portfolio value evolves as the closing prices change
    buy_hold_portfolio = buy_hold_shares * closing_prices['Open']

    return buy_hold_portfolio.tolist()
