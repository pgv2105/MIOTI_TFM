import json
from statistics import median
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import pandas_ta as ta

from backtesting.indicators.kalman_trend import KalmanTrend


def get_trade_exposure(open_price, open_trades, balance,type_trade, LEVERAGE):

    if len(open_trades) == 0: return (balance / 6) * LEVERAGE
    # get last open trade
    last_trade = max(open_trades, key=lambda t: t['entry_index'])

    if ((type_trade == 'long' and last_trade['open_price_mid'] > open_price and last_trade['type'] == type_trade) or
            (type_trade == 'short' and last_trade['open_price_mid'] < open_price and last_trade['type'] == type_trade)):

        exposure = last_trade['exposure'] * 1.5 / LEVERAGE
    else:
        exposure = balance / 6

    if sum(trade['exposure'] for trade in open_trades) + exposure*LEVERAGE > balance * LEVERAGE:
        rest_amount = (balance * LEVERAGE - sum(trade['exposure'] for trade in open_trades)) / LEVERAGE
        exposure = rest_amount if rest_amount > 10000 else 0 # Set last money that we have if it exceed minimum transaction

    return exposure * LEVERAGE

def get_trade_exposure_LEVERAGE(open_price, open_trades, balance,type_trade, LEVERAGE):

    if len(open_trades) == 0: return (balance / 6) * LEVERAGE

    exposure = balance / 6

    # get last open trade
    last_trade = max(open_trades, key=lambda t: t['entry_index'])

    if ((type_trade == 'long' and last_trade['open_price_original'] > open_price and last_trade['type'] == type_trade) or
            (type_trade == 'short' and last_trade['open_price_original'] < open_price and last_trade['type'] == type_trade)):

        LEVERAGE = LEVERAGE + 6
    else:
        LEVERAGE = LEVERAGE


    return exposure * LEVERAGE

def detect_market_trend(prices: pd.Series, fast_length=100, slow_length=300, check_interval=800):
    """
    Evalúa el market trend basándose en medias móviles y decide modelo (long o short).
    Devuelve una lista con el modelo a usar en cada timestep: 'long' o 'short'
    """
    df = pd.DataFrame({'Open': prices})
    df['SMA_fast'] = ta.sma(df['Open'], length=fast_length)
    df['SMA_slow'] = ta.sma(df['Open'], length=slow_length)

    model_usage = []
    current_model = 'long'  # default

    for i in range(len(df)):
        if (i == 300 or i % check_interval == 0) and i >= slow_length:
            if df.iloc[i]['SMA_fast'] > df.iloc[i]['SMA_slow']:
                current_model = 'long'
            else:
                current_model = 'short'
        model_usage.append(current_model)

    return model_usage


def trimmed_mean(data: List[float], proportion: float = 0.1) -> float:
    """Compute the mean after trimming the highest and lowest `proportion` of values."""
    if not data:
        return 0.0
    n = len(data)
    cut = int(n * proportion)
    trimmed = sorted(data)[cut : n - cut]
    return sum(trimmed) / len(trimmed) if trimmed else 0.0

def update_drawdown_freeze(
        balance: float,
        trade_pnl: float,
        peak_balance: float,
        freeze_min_balance: float,
        balance_freezed: float,
        drawdown_freeze: bool,
        confirm_freeze: bool,
        DRAWDOWN_THRESHOLD: float,
        RECOVERY_AMOUNT: float
) -> Tuple[bool, float, float, float, bool]:
    """
    Updates the drawdown freeze state and related markers.
    Returns:
      drawdown_freeze,
      updated peak_balance,
      updated freeze_min_balance,
      updated balance_freezed,
      confirm_freeze_signal (True if RECOVERY_AMOUNT reached)
    """

    # 1) Update peak if not frozen
    if not drawdown_freeze:
        peak_balance = max(peak_balance, balance)

    # 2) Enter freeze if drawdown exceeds threshold
    current_drawdown = balance - peak_balance
    if not drawdown_freeze and current_drawdown < DRAWDOWN_THRESHOLD:
        drawdown_freeze    = True
        freeze_min_balance = balance
        balance_freezed    = balance

    # 3) While frozen, update the simulated frozen balance
    elif drawdown_freeze:
        balance_freezed    += trade_pnl
        freeze_min_balance = min(freeze_min_balance, balance_freezed)

        # 4) If we've recovered enough on the frozen balance, signal confirmation
        if balance_freezed >= freeze_min_balance + RECOVERY_AMOUNT:
            confirm_freeze = True

    return drawdown_freeze, peak_balance, freeze_min_balance, balance_freezed, confirm_freeze

def process_confirmation(
        confirm_pnls: List[float],
        confirm_freeze: bool,
        drawdown_freeze: bool,
        freeze_min_balance: float,
        balance_freezed: float,
        balance: float,
        peak_balance: float,
        CONFIRM_LOOKBACK: int
) -> Tuple[bool, bool, List[float], float, float]:
    """
    Handles the confirmation phase:
      - Accumulates the last CONFIRM_LOOKBACK PnLs in confirm_pnls
      - If trimmed mean > 0, exits freeze
      - If window filled without success, resets freeze markers
    Returns:
      updated drawdown_freeze,
      updated confirm_freeze,
      cleaned confirm_pnls,
      freeze_min_balance
    """

    if confirm_freeze:

        # When we have enough for a full window
        if len(confirm_pnls) == CONFIRM_LOOKBACK:
            avg_trim = trimmed_mean(confirm_pnls, proportion=0.1)
            if avg_trim > 0:
                # Successful confirmation: exit all freeze states
                drawdown_freeze = False
                confirm_freeze  = False
                peak_balance = balance  # readjust peak
            else:
                # Failed confirmation: stay frozen, reset freeze start point
                drawdown_freeze = True
                confirm_freeze  = False
                freeze_min_balance = min(freeze_min_balance, balance_freezed)

            confirm_pnls.clear()

    return drawdown_freeze, confirm_freeze, confirm_pnls, freeze_min_balance, peak_balance




def classify_volatility(current_atr, history, fixed_thresholds=None):
    """
    Classify volatility state as 'low', 'medium', or 'high'.
    Uses rolling quantiles if fixed_thresholds is None.
    """
    if fixed_thresholds:
        q1, q2 = fixed_thresholds
    else:
        # Need enough history to compute quantiles
        if len(history) < history.maxlen:
            return 'medium'
        q1, q2 = np.percentile(history, [45, 73])
    if current_atr <= q1:
        return 'low'
    if current_atr <= q2:
        return 'medium'
    return 'high'


def get_factors_for_market(vol_state, current_atr):

    factor_loss = 0
    target = 0

    if vol_state == 'low':
        factor_loss = 0.035
        target = 0.04
    elif vol_state == 'medium':
        factor_loss = 0.04
        target = 0.05
    elif vol_state == 'high':
        factor_loss = 0.045
        target = 0.07

    return factor_loss, target


def get_dynamic_params(vol_state: str, slope_pips: float) -> dict:
    """
    Return dynamic parameters for each regime:
      - pred_thr: minimum model score to enter
      - slope_thr: minimum expected move (pips) to enter
      - sl_mult: stop-loss as a multiple of ATR
      - tp_mult: take-profit as a multiple of ATR
    """
    # Example lookup; calibra estos valores en tu backtest
    lookup = {
        ('low',  'up'):     {'pred_thr': 0.45, 'slope_thr': 0.25, 'sl_mult': 0.03, 'tp_mult': 0.045},
        ('low',  'flat'):   {'pred_thr': 0.45, 'slope_thr': 0.25, 'sl_mult': 0.03, 'tp_mult': 0.45},
        ('low',  'down'):   {'pred_thr': 0.60, 'slope_thr': 0.0, 'sl_mult': 0.025, 'tp_mult': 0.35},
        ('medium','up'):    {'pred_thr': 0.43, 'slope_thr': 0.8, 'sl_mult': 0.045, 'tp_mult': 0.055},
        ('medium','flat'):  {'pred_thr': 0.50, 'slope_thr': 1.2, 'sl_mult': 0.035, 'tp_mult': 0.05},
        ('medium','down'):  {'pred_thr': 0.60, 'slope_thr': 1.2, 'sl_mult': 0.03, 'tp_mult': 0.04},
        ('high', 'up'):     {'pred_thr': 0.43, 'slope_thr': 1.0, 'sl_mult': 0.05, 'tp_mult': 0.065},
        ('high', 'flat'):   {'pred_thr': 0.50, 'slope_thr': 1.5, 'sl_mult': 0.045, 'tp_mult': 0.055},
        ('high', 'down'):   {'pred_thr': 0.65, 'slope_thr': 1.5, 'sl_mult': 0.05, 'tp_mult': 0.045},
    }
    # map slope_pips to categorical 'up','flat','down'
    if slope_pips > 0.15:
        slope_state = 'up'
    elif slope_pips > 0:
        slope_state = 'flat'
    else:
        slope_state = 'down'

    return lookup.get((vol_state, slope_state), {})

def simulate_forex_closure_ibkr(usd_entries, entry_prices, exit_price, spread, slippage, commission_pips, commission_min = 2):

    total_eur = 0
    total_usd_invested = 0
    total_commission = 0

    for usd_amount, entry_price in zip(usd_entries, entry_prices):
        # Adjust entry price to spread and slippage
        executed_entry_price = entry_price + spread/2 + slippage

        # EUR bought overall
        eur_bought = usd_amount / executed_entry_price
        total_eur += eur_bought
        total_usd_invested += usd_amount

        # Calculate comission per operation
        commission = usd_amount * commission_pips * 2
        commission = max(commission, commission_min)
        total_commission += commission

    # Closed all trades at same time
    executed_exit_price = exit_price - spread/2 - slippage
    usd_returned = total_eur * executed_exit_price

    # Calculate final pnl
    net_profit = usd_returned - total_usd_invested - total_commission

    return net_profit

def simulate_forex_short_closure_ibkr(usd_entries,
                                      entry_prices,
                                      exit_price,
                                      spread,
                                      slippage,
                                      commission_pips,
                                      commission_min = 2):
    """
    Simula cierre de varias operaciones en corto en EURUSD usando IBKR.

    Parámetros:
    - usd_entries: lista de importes en USD de la exposición inicial deseada.
    - entry_prices: lista de precios de entrada (mid) para cada operación.
    - exit_price: precio mid único al que cierras todas.
    - spread: spread total en pips (o en precio).
    - slippage: slippage en pips (o en precio).
    - commission_pips: comisión expresada como fracción del volumen (por ejemplo 0.00002 para 0.2 pips).
    - commission_min: comisión mínima en USD por operación.

    Retorna:
    - net_profit: beneficio neto en USD.
    """

    total_eur_shorted = 0.0
    total_usd_received = 0.0
    total_commission = 0.0

    # Apertura de cortos: vendemos EURUSD
    for usd_amount, entry_price in zip(usd_entries, entry_prices):
        # Precio de ejecución de venta (bid)
        executed_entry_price = entry_price - spread/2 - slippage

        # EUR “vendidos” en la apertura del corto
        eur_shorted = usd_amount / executed_entry_price
        total_eur_shorted += eur_shorted
        total_usd_received += usd_amount

        # Comisión doble (entrada + salida)
        commission = usd_amount * commission_pips * 2
        commission = max(commission, commission_min)
        total_commission += commission

    # Cierre de cortos: recompramos EURUSD
    executed_exit_price = exit_price + spread/2 + slippage
    usd_needed_to_close = total_eur_shorted * executed_exit_price

    # P&L neto: USD recibidos al abrir menos USD pagados al cerrar y comisiones
    net_profit = total_usd_received - usd_needed_to_close - total_commission

    return net_profit


from collections import defaultdict

def group_trades_by_final_exec(trades):
    """
    Groups a list of trades by their 'finalPrice_exec'.

    Parameters:
    - trades: list of dicts, each containing the 'finalPrice_exec' key.

    Returns:
    - dict where each key is a final execution price and the value is
      a list of trades sharing that price.
    """
    grouped = defaultdict(list)
    for trade in trades:
        key = (trade.get('executed_price'), trade.get('type'))
        grouped[key].append(trade)
    return dict(grouped)