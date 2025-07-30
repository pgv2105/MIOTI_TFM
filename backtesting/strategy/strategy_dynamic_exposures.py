import json
import logging

from backtesting.indicators.kalman_trend import KalmanTrend
from backtesting.indicators.rsi import IncrementalRSI
from backtesting.strategy.services_strategy import simulate_forex_closure_ibkr, simulate_forex_short_closure_ibkr, \
    group_trades_by_final_exec, get_trade_exposure, get_trade_exposure_LEVERAGE
from backtesting.strategy.trading_strats import simulate_trade_partial, simulate_short_trade_partial

# Strategy parameters
INITIAL_CAPITAL = 70000  # Example initial capital in USD
LEVERAGE = 10            # Leverage of 1:10
SPREAD = 0.00003  # spread de 0.3 pips en EUR/USD (típico IB)
COMM_P_SIDE = 0.00002 # Minimum commission IBKR
SLIPPAGE = 0.00001


def backtest_strategy_dynamic_exposure(model, model_inv, X_val, y_val, type_simulation):


    test_predictions = model.predict(X_val).flatten()
    test_predictions_inverse = model_inv.predict(X_val).flatten()

    predictionsList = test_predictions.tolist()
    predictionsListInverse = test_predictions_inverse.tolist()

    if type_simulation == 'test':
        # Open and read the JSON file
        #with open('data/predictions/testPredictionsINVERSE.json', 'w') as file:
        #   json.dump(test_predictions_inverse.tolist(), file)

        with open('data/predictions/testPredictions.json', 'r') as f:
            predictionsList = json.load(f)

        with open('data/predictions/testPredictionsINVERSE.json', 'r') as f:
            predictionsListInverse = json.load(f)

    elif type_simulation == 'val':
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

    # Initialize Kalman filter to evaluate trends
    kf_long_period = KalmanTrend(delta=1e-4, R=0.01)

    # RSI parameters
    rsi_calc = IncrementalRSI(period=14)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

                # Check current sum in case the model is not working
                current_overall_pnl = sum(t['current_pnl'] for t in open_trades)
                # flag_fail_trades = current_overall_pnl < -275 # TODO: MIRAR BIEN PILLAR EL FINAL TODOS

                if rsi_value is not None and ( # TODO: MIRAR BIEN TENGO QUE TENER EN CUENTA DESPLAZAMIENTO FOR
                        (signal and rsi_value < 60 and slope_long > 0 and type_trade == 'long') or
                        (inverse_signal and rsi_value > 75 and type_trade == 'short')):

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

                    logging.info(f"[TRADE - {t['type']}- ] Entry: {t['time_entry']} @ {t['open_price_original']}, Exit: {y_val.iloc[i].name} @ {price}, "
                                 f"PnL Baseline: {mean_pnl_group:.2f}, Final Balance: {balance:.2f}")

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



        # -- 2) -- Model signal & clear history
        prediction = predictionsList[i]
        inverse_prediction = predictionsListInverse[i]
        signal = prediction > 0.42
        inverse_signal = inverse_prediction > 0.45

        # -- 3) -- Get operation type long/short
        final_signal = False
        entry_price = 0
        type_trade = 'Null'

        exposure = 0

        if rsi_value is not None and signal and rsi_value < 60 and slope_long > 0: #
            type_trade = 'long'
            exposure = get_trade_exposure(open_price, open_trades, balance, type_trade, LEVERAGE)
            # get_trade_exposure_LEVERAGE
            final_signal  = True
        elif rsi_value is not None and inverse_signal and rsi_value > 75: # slope_long < 0 and
            type_trade = 'short'
            exposure = get_trade_exposure(open_price, open_trades, balance, type_trade, LEVERAGE)
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

        # Check current sum in case the model is not working
        current_overall_pnl = sum(t['current_pnl'] for t in open_trades)
        # flag_fail_trades = current_overall_pnl < -275

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