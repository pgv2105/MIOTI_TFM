import h5py
import json
import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
from keras.src.saving import load_model
from sklearn.model_selection import train_test_split

from backtesting.analyze.analyze_results import results_analysis
from backtesting.analyze.market_analysis import study_market
from backtesting.strategy.services_strategy import detect_market_trend
from backtesting.strategy.strategy_double_model import backtest_strategy_double_model, backtest_strategy_monkeys
from backtesting.strategy.strategy_dynamic_exposures import backtest_strategy_dynamic_exposure, backtest_strategy_test_no_model
from backtesting.strategy.strategy_simple import backtest_strategy, buy_and_hold, backtest_no_model_strategy
from models.HYBRIDS.model_HYBRIDS import custom_loss_fp, PositionalEncoding



if __name__ == '__main__':

    loss_function = custom_loss_fp(1.1, 0.0)
    loss_function.__name__ = "loss_fn"  # Se debe registrar con el nombre "loss_fn"

    # MODEL PARA LONG Y SHORT ACTUAL
    model_OPERATIONS = load_model('data/models/model_PLUS_EXTENDED.keras', custom_objects={ # model_LONG_SHORT
        'loss_fn': loss_function,
        'PositionalEncoding': PositionalEncoding
    })

    model_DIRECTIONALITY = load_model('data/models/model_DIRECTIONALITY_0_4942.keras')
    #model_DIRECTIONALITY = load_model('data/models/model_DIRECTIONALITY_0_51_EXTENDED.keras')
    # Load previous preprocessed data -- Da igual la normal que la INVERSE, son los mismos datos
    with open("model_directionality/data/backtest_data_model_PLUS_MULTIVARIATE.pkl", "rb") as f:
        data = pickle.load(f)

    X_LSTM = data["dataset_X"]
    Y_LSTM = data["dataset_Y"]

    #  Separate train/test
    X_train, X_test, y_train, y_test = train_test_split(X_LSTM, Y_LSTM, test_size=0.3, shuffle=False)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    type_simulation = 'test'



    if type_simulation == 'test':
        x = X_test
        y = y_test
    else:
        x = X_val
        y = y_val

    #x = X_LSTM
    #y = Y_LSTM

    buyAndHold = buy_and_hold(y)
    balanceHistory, executedTrades = backtest_strategy_monkeys(model_OPERATIONS, model_DIRECTIONALITY, x, y, type_simulation)
    balanceHistoryNoModel, executedTradesNoModel = backtest_strategy_test_no_model(model_OPERATIONS, x, y)

    results_analysis(executedTrades, balanceHistory, buyAndHold)

    #study_market(y, executedTrades)

    matplotlib.use('TkAgg')

    plt.figure(figsize=(12, 6))

    # Estilo de línea y grosor
    plt.plot(balanceHistory, label='Balance con modelo', linewidth=2)
    plt.plot(balanceHistoryNoModel, label='Balance sin modelo', linestyle='--', linewidth=2)
    plt.plot(buyAndHold, label='Buy and Hold', linestyle=':', linewidth=2)

    # Ejes y formato
    plt.xlabel('Intervalos de Tiempo (Cierres)', fontsize=12)
    plt.ylabel('Valor de la Cartera (€)', fontsize=12)
    plt.title('Evolución del PnL: Estrategia de Trading con IA vs Benchmark', fontsize=14)


    # Estética
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()

    # Eliminar bordes superior y derecho
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()

    # Plot cumulative balance evolution over time
    plt.figure(figsize=(10, 5))
    plt.plot(balanceHistory, label='Strategy Cumulative Balance')
    plt.plot(balanceHistoryNoModel, label='Strategy Cumulative Balance No Model')
    plt.plot(buyAndHold, label='Buy and Hold Portfolio Value')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Time Steps (Closing Price Intervals)')
    plt.ylabel('Portfolio Value (€)')
    plt.title('Comparison: Trading Strategy vs. Buy and Hold')
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------  PRINT TRADES STATISTICS  -------------
    # -- MODEL
    total_trades = len(executedTrades)
    winning_trades = sum(1 for t in executedTrades if t['pnl_baseline'] > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    average_pnl = sum(t['pnl_baseline'] for t in executedTrades) / total_trades if total_trades > 0 else 0
    avg_win = sum(t['pnl_baseline'] for t in executedTrades if t['pnl_baseline'] > 0) / winning_trades if winning_trades > 0 else 0
    avg_loss = sum(t['pnl_baseline'] for t in executedTrades if t['pnl_baseline'] < 0) / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0
    risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 'N/A'

    print("----------- RESULTS USING MODEL --------------")
    print("Total trades:", total_trades)
    print("Win rate:", win_rate)
    print("Average pnl:", average_pnl)
    print("Average win:", avg_win)
    print("Average loss:", avg_loss)
    print("Risk reward ratio:", risk_reward_ratio)


    # -- NO MODEL
    total_trades = len(executedTradesNoModel)
    winning_trades = sum(1 for t in executedTradesNoModel if t['pnl_baseline'] > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    average_pnl = sum(t['pnl_baseline'] for t in executedTradesNoModel) / total_trades if total_trades > 0 else 0
    avg_win = sum(t['pnl_baseline'] for t in executedTradesNoModel if t['pnl_baseline'] > 0) / winning_trades if winning_trades > 0 else 0
    avg_loss = sum(t['pnl_baseline'] for t in executedTradesNoModel if t['pnl_baseline'] < 0) / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0
    risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 'N/A'

    print("----------- RESULTS WITHOUT MODEL --------------")
    print("Total trades:", total_trades)
    print("Win rate:", win_rate)
    print("Average pnl:", average_pnl)
    print("Average win:", avg_win)
    print("Average loss:", avg_loss)
    print("Risk reward ratio:", risk_reward_ratio)






    finish= True



