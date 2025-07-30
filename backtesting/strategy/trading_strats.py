from numba import njit

STOP_LOSS_FACTOR = 0.1    # Example: limit 2% loss (SL)
SPREAD = 0.00002  # spread de 0.2 pips en EUR/USD (típico IB)
SLIPPAGE = 0.00005
PREDICTED_TARGET = 0.05
STOP_LOSS_FACTOR_ADJUSTED = 0.014



def simulate_trade_partial(prices, openPrice, intervals_passed):
    """
    Simulate operation for LONG position in given 10 minutes interval (each of them with 20 steps of 30 secs).
    Returns current trade state (current_price, if it is still open and final price index)
    """
    target_factor = 0.062 #0.62
    current_factor = 0.047 #0.05
    tp_factor = 0.037
    switched = False
    switched_tp = False

    stopLoss = openPrice * (1 - current_factor / 100)
    maxPrice = openPrice
    target_price = openPrice * (1 + target_factor / 100)

    for i, bidPrice in enumerate(prices[:intervals_passed*20]):

        # Check stop-loss
        if bidPrice <= stopLoss:
            executedPrice = stopLoss
            return executedPrice, False, i  # Close trade

        # Activate if it reaches our objective
        if not switched and bidPrice >= target_price:
            switched = True
            current_factor = STOP_LOSS_FACTOR_ADJUSTED
            stopLoss = bidPrice * (1 - current_factor / 100)

        # Update trailing
        if switched and bidPrice > maxPrice:
            maxPrice = bidPrice
            stopLoss = maxPrice * (1 - current_factor / 100)

    # Current price of the last interval simulated (trade still open)
    return bidPrice, True, i

def simulate_short_trade_partial(prices, openPrice, intervals_passed):
    """
    Simulate operation for SHORT position in given 10 minutes interval (each of them with 20 steps of 30 secs).
    Returns current trade state (current_price, if it is still open and final price index)
    """
    current_factor = 0.047 #0.05# 0.055
    target_factor = 0.062 #0.064 # 0.07
    tp_factor = 0.037
    switched = False
    switched_tp = False

    stopLoss = openPrice * (1 + current_factor / 100)
    minPrice = openPrice
    target_price = openPrice * (1 - target_factor / 100)

    for i, bidPrice in enumerate(prices[:intervals_passed*20]):

        # Check stop-loss
        if bidPrice >= stopLoss:
            executedPrice = stopLoss
            return executedPrice, False, i  # Close trade

        # Activate if it reaches our objective
        if not switched and bidPrice <= target_price:
            switched = True
            current_factor = STOP_LOSS_FACTOR_ADJUSTED
            stopLoss = bidPrice * (1 + current_factor / 100)

        # Update trailing
        if switched and bidPrice < minPrice:
            minPrice = bidPrice
            stopLoss = minPrice * (1 + current_factor / 100)

    # Current price of the last interval simulated (trade still open)
    return bidPrice, True, i




def simulate_trade_partial_training(prices, openPrice, intervals_passed):
    """
    Simulate operation for LONG position in given 10 minutes interval (each of them with 20 steps of 30 secs).
    Returns current trade state (current_price, if it is still open and final price index)
    """
    target_factor = 0.075 #0.067 #0.62
    current_factor = 0.055 #0.047 #0.05
    switched = False

    stopLoss = openPrice * (1 - current_factor / 100)
    maxPrice = openPrice
    target_price = openPrice * (1 + target_factor / 100)

    for i, bidPrice in enumerate(prices[:intervals_passed*20]):

        # Check stop-loss
        if bidPrice <= stopLoss:
            executedPrice = stopLoss
            reason = 'stop_loss'
            if switched: reason = 'target_factor'
            return executedPrice, False, reason  # Close trade

        # Activate if it reaches our objective
        if not switched and bidPrice >= target_price:
            switched = True
            current_factor = STOP_LOSS_FACTOR_ADJUSTED
            stopLoss = bidPrice * (1 - current_factor / 100)

        # Update trailing
        if switched and bidPrice > maxPrice:
            maxPrice = bidPrice
            stopLoss = maxPrice * (1 - current_factor / 100)

    # Current price of the last interval simulated (trade still open)
    return bidPrice, True, 'open'


def simulate_short_trade_partial_training(prices, openPrice, intervals_passed):
    """
    Simulate operation for SHORT position in given 10 minutes interval (each of them with 20 steps of 30 secs).
    Returns current trade state (current_price, if it is still open and final price index)
    """
    current_factor = 0.055# 0.06
    target_factor = 0.075 # 0.065
    switched = False

    stopLoss = openPrice * (1 + current_factor / 100)
    minPrice = openPrice
    target_price = openPrice * (1 - target_factor / 100)

    for i, bidPrice in enumerate(prices[:intervals_passed*20]):

        # Check stop-loss
        if bidPrice >= stopLoss:
            executedPrice = stopLoss
            reason = 'stop_loss'
            if switched: reason = 'target_factor'
            return executedPrice, False, reason  # Close trade

        # Activate if it reaches our objective
        if not switched and bidPrice <= target_price:
            switched = True
            current_factor = STOP_LOSS_FACTOR_ADJUSTED
            stopLoss = bidPrice * (1 + current_factor / 100)

        # Update trailing
        if switched and bidPrice < minPrice:
            minPrice = bidPrice
            stopLoss = minPrice * (1 + current_factor / 100)

    # Current price of the last interval simulated (trade still open)
    return bidPrice, True, 'open'
