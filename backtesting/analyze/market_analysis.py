import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import ttest_rel, wilcoxon
import matplotlib
from ta.trend import ADXIndicator

from backtesting.indicators.kalman_trend import KalmanTrend


def study_market(df_market, df_trades):
    """
    Enrich trades with market context features.
    """
    df_features = add_context_features(df_market)

    df_trades = pd.DataFrame(df_trades)
    df_enriched = enrich_trades_with_context(df_trades, df_features)


    # 1. Bucketize regimes:
    df_buckets = bucketize_features(df_enriched)

    # 2.a) Compare current strategy with market timeline
    check_correlations(df_enriched)
    compare_market_strategy(df_trades, df_features)

    # 2. Compute metrics by regime:
    regime_metrics_df = compute_regime_analysis(df_buckets)
    print(regime_metrics_df)

    # 3. Plot heatmap for EV_delta:
    plot_heatmap(regime_metrics_df, metric='EV_delta')
    return df_enriched


# 1) Add context features (as defined previously)
def add_context_features(df):
    df = df.copy()

    # Realized volatility over 1 hour (6 periods)
    df['rv_1h'] = (
        np.log(df['Close'].astype(float) /
               df['Close'].astype(float).shift(1))
        .rolling(6)
        .std()
    )

    df['high_aux']  = df['Prices'].apply(lambda px: max(px[:20]))
    df['low_aux']   = df['Prices'].apply(lambda px: min(px[:20]))
    df['close_aux'] = df['Prices'].apply(lambda px: px[20])

    adx_period = 14
    adx_ind = ADXIndicator(high=df['high_aux'],
                           low=df['low_aux'],
                           close=df['close_aux'],
                           window=adx_period)
    df['ADX14'] = adx_ind.adx()

    # True Range (TR) and Average True Range (ATR) over 1 hour
    df['prev_close'] = df['Open'].astype(float)
    df['tr'] = df.apply(
        lambda x: max(
            max(x['Prices'][:20]) - min(x['Prices'][:20]),
            abs(max(x['Prices'][:20]) - x['prev_close']),
            abs(min(x['Prices'][:20]) - x['prev_close'])
        ),
        axis=1
    )
    df['atr_1h'] = df['tr'].ewm(span=6, adjust=False).mean()

    kf = KalmanTrend(delta=1e-4, R=0.001)
    _, df['kf_slope'] = zip(*(kf.update(p) for p in df['Close']))


    # Relative Strength Index (RSI) with 14-period window
    delta = df['Close'].astype(float).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    df['rsi'] = 100 - 100 / (
            1 + gain.rolling(14).mean() / loss.rolling(14).mean()
    )

    # Add EMA slope to evaluate price tendency
    span = 20
    df['ema'] = df['Close'].ewm(span=span, adjust=False).mean()
    df['ema_slope'] = (df['ema'] - df['ema'].shift(span)) / span

    # Returns
    df['price_return'] = df['Open'].pct_change().fillna(0)
    # ATR volatility with 14 periods
    df['volatility'] = df['price_return'].rolling(14).std().fillna(method='bfill')

    # Clean up intermediate columns
    return df.drop(columns=['prev_close', 'tr'], errors='ignore')


# 2) Function to enrich trades with context from market features
def enrich_trades_with_context(df_trades, df_features):
    df_features = df_features.sort_index()
    df = df_trades.copy()
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])

    # Merge market features at entry time
    df = pd.merge_asof(
        df.sort_values('entry_time'),
        df_features,
        left_on='entry_time',
        right_index=True,
        direction='backward'
    )

    # Merge market features at exit time
    #df = pd.merge_asof(
    #    df.sort_values('exit_time'),
    #    df_features,
    #    left_on='exit_time',
    #    right_index=True,
    #    direction='backward',
    #    suffixes=('', '_exit')
    #)

    return df


# 3) Paired trade analysis function
def paired_trade_analysis(df):
    # Prepare data
    df = df.sort_values('entry_time').reset_index(drop=True)
    df['delta_pnl'] = df['pnl_dynamic'] - df['pnl_baseline']
    df['eq_baseline'] = df['pnl_baseline'].cumsum()
    df['eq_dynamic'] = df['pnl_dynamic'].cumsum()
    df['eq_delta'] = df['delta_pnl'].cumsum()

    def compute_metrics(pnl):
        """
        Compute performance metrics: expected value, profit factor,
        Sharpe ratio, and maximum drawdown.
        """
        ev = pnl.mean()
        pf = pnl[pnl > 0].sum() / -pnl[pnl < 0].sum()
        sharpe = pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl))
        max_dd = (pnl.cumsum().cummax() - pnl.cumsum()).max()
        return {'EV': ev, 'PF': pf, 'Sharpe': sharpe, 'MaxDD': -max_dd}

    metrics_baseline = compute_metrics(df['pnl_baseline'])
    metrics_dynamic = compute_metrics(df['pnl_dynamic'])
    metrics_delta = compute_metrics(df['delta_pnl'])

    metrics_table = pd.DataFrame({
        'Baseline': metrics_baseline,
        'Dynamic': metrics_dynamic,
        'Delta': metrics_delta
    })
    display(metrics_table.style.set_caption('Comparison Metrics'))

    # Plot equity curves
    plt.figure()
    plt.plot(df['eq_baseline'], label='Baseline')
    plt.plot(df['eq_dynamic'], label='Dynamic')
    plt.legend()
    plt.title('Equity Curves')
    plt.xlabel('Trade #')
    plt.ylabel('Equity')
    plt.show()

    # Plot ΔPnL distribution
    plt.figure()
    plt.hist(df['delta_pnl'], bins=30)
    plt.title('ΔPnL Distribution')
    plt.xlabel('ΔPnL')
    plt.ylabel('Frequency')
    plt.show()

    # Statistical tests
    t_stat, p_val_t = ttest_rel(df['pnl_dynamic'], df['pnl_baseline'])
    w_stat, p_val_w = wilcoxon(df['pnl_dynamic'], df['pnl_baseline'])
    stats = pd.Series({
        't-stat': t_stat,
        'p-val (t)': p_val_t,
        'Wilcoxon': w_stat,
        'p-val (w)': p_val_w
    })
    display(
        stats.to_frame('Value')
        .style.set_caption('Statistical Tests')
    )

    return df


# 4) Bucket analysis function
def bucket_analysis(df, feature, n_buckets=4):
    df = df.copy()
    bucket_col = f'{feature}_bucket'
    df[bucket_col] = pd.qcut(df[feature], n_buckets, labels=False)

    agg = df.groupby(bucket_col).agg({
        'pnl_baseline': ['mean', 'count'],
        'pnl_dynamic': ['mean'],
        'delta_pnl': ['mean', 'std']
    })

    return agg


def bucketize_features(df, vol_col='rv_1h', slope_col='ema_slope'):
    """
    Bucketize volatility and slope into three regimes: low, medium, high.

    Args:
        df (pd.DataFrame): DataFrame containing the features.
        vol_col (str): Column name for volatility.
        slope_col (str): Column name for slope.

    Returns:
        pd.DataFrame: DataFrame with added 'vol_bucket' and 'slope_bucket'.
    """
    df = df.copy()
    df['vol_bucket'] = pd.qcut(
        df[vol_col], q=3, labels=['vol_low', 'vol_medium', 'vol_high']
    )
    df['slope_bucket'] = pd.qcut(
        df[slope_col], q=3, labels=['tend_low', 'tend_medium', 'tend_high']
    )
    return df


def regime_metrics(sub_df):
    """
    Compute performance metrics for a given regime subset.

    Args:
        sub_df (pd.DataFrame): Subset of df_final for one regime.

    Returns:
        pd.Series: Metrics including EV, win rate, PF, and max drawdown.
    """
    pnl_base = sub_df['pnl_baseline']
    pnl_dyn = sub_df['pnl_dynamic']
    delta_pnl = pnl_dyn - pnl_base

    # Expectancy (mean PnL)
    ev_base = pnl_base.mean()
    ev_dyn = pnl_dyn.mean()
    ev_delta = delta_pnl.mean()

    # Win rates
    win_base = sub_df['win_baseline'].mean()
    win_dyn = sub_df['win_dynamic'].mean()

    # Profit Factor
    gross_win_base = pnl_base[pnl_base > 0].sum()
    gross_loss_base = -pnl_base[pnl_base < 0].sum()
    pf_base = gross_win_base / gross_loss_base if gross_loss_base != 0 else np.nan

    gross_win_dyn = pnl_dyn[pnl_dyn > 0].sum()
    gross_loss_dyn = -pnl_dyn[pnl_dyn < 0].sum()
    pf_dyn = gross_win_dyn / gross_loss_dyn if gross_loss_dyn != 0 else np.nan

    # Max drawdown
    def max_drawdown(pnl_series):
        eq = pnl_series.cumsum()
        drawdown = eq.cummax() - eq
        return drawdown.max()

    mdd_base = max_drawdown(pnl_base)
    mdd_dyn = max_drawdown(pnl_dyn)

    return pd.Series({
        'EV_base': ev_base,
        'EV_dyn': ev_dyn,
        'EV_delta': ev_delta,
        'Win_base': win_base,
        'Win_dyn': win_dyn,
        'PF_base': pf_base,
        'PF_dyn': pf_dyn,
        'MDD_base': mdd_base,
        'MDD_dyn': mdd_dyn
    })


def compute_regime_analysis(df):
    """
    Compute performance metrics for all combinations of volatility and slope regimes.

    Args:
        df (pd.DataFrame): Enriched DataFrame with 'vol_bucket' and 'slope_bucket'.

    Returns:
        pd.DataFrame: Multi-index DataFrame of metrics for each (vol_bucket, slope_bucket).
    """
    regime_df = (
        df
        .groupby(['vol_bucket', 'slope_bucket'], observed=True)
        .apply(regime_metrics)
    )
    return regime_df


def plot_heatmap(regime_df, metric='EV_delta'):
    """
    Plot a heatmap of a selected metric over volatility vs. slope regimes.

    Args:
        regime_df (pd.DataFrame): Output from compute_regime_analysis.
        metric (str): Column name of the metric to plot.
    """

    matplotlib.use('TkAgg')
    # Pivot table: rows=vol_bucket, cols=slope_bucket
    pivot = regime_df[metric].unstack(level='slope_bucket')
    labels_x = pivot.columns.tolist()
    labels_y = pivot.index.tolist()
    data = pivot.values.astype(float)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(data, origin='lower', aspect='auto', interpolation='nearest')
    plt.xticks(np.arange(len(labels_x)), labels_x, rotation=45)
    plt.yticks(np.arange(len(labels_y)), labels_y)
    plt.title(f'Heatmap of {metric} by Regime')
    plt.xlabel('Trend Regime')
    plt.ylabel('Volatility Regime')
    plt.colorbar(im, label=metric)
    plt.tight_layout()
    plt.show()



def compare_market_strategy(df_trades, df_features):
    period_stats = period_summary(df_trades, df_features, period_size=1000)
    print(period_stats)

    matplotlib.use('TkAgg')

    # You can then visualize correlations:
    # e.g. plot win_rate vs. avg_atr, avg_slope

    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].scatter(period_stats['avg_atr'], period_stats['win_rate'])
    ax[0].set_xlabel('Avg ATR (1h)'); ax[0].set_ylabel('Win Rate')
    ax[0].set_title('Win Rate vs. ATR per Period')

    ax[1].scatter(period_stats['avg_slope'], period_stats['avg_pnl'])
    ax[1].set_xlabel('Avg Slope'); ax[1].set_ylabel('Avg PnL (€)')
    ax[1].set_title('Avg PnL vs. Slope per Period')

    plt.tight_layout()
    plt.show()


def period_summary(df_trades, df_feat, period_size=1000):
    """
    Slice the backtest into fixed-size periods and compute, per period:
      - market context: avg ATR, avg slope, vol_bucket distribution, slope_bucket distribution
      - strategy performance: win_rate, avg_pnl, profit_factor, max_drawdown
    Args:
        df_trades (pd.DataFrame): must contain
            ['entry_idx', 'entry_time', 'pnl_dynamic', 'exit_price', 'entry_price']
        df_feat   (pd.DataFrame): indexed by entry_time, must contain
            ['atr_1h', 'kf_slope'] and optionally 'vol_bucket', 'slope_bucket'
        period_size (int): number of trades per group
    Returns:
        pd.DataFrame: one row per period with aggregated stats
    """
    periods = []
    n = len(df_feat)
    for start in range(0, n, period_size):
        end = min(start + period_size, n)

        mask_index = (start < df_trades['entry_index']) & (df_trades['entry_index'] < end)
        trades_slice = df_trades[mask_index].copy()

        # MARKET CONTEXT
        # map each trade back to its market features at entry
        feats = df_feat.reindex(trades_slice['entry_time'])
        avg_atr   = feats['atr_1h'].mean()
        avg_slope = feats['kf_slope'].mean()

        # (re)compute vol_bucket and slope_bucket if needed
        feats['vol_bucket'] = pd.qcut(feats['atr_1h'], 3, labels=['low','med','high'])
        feats['slope_bucket'] = pd.qcut(feats['kf_slope'], 3, labels=['down','flat','up'])
        vol_dist   = feats['vol_bucket'].value_counts(normalize=True).to_dict()
        slope_dist = feats['slope_bucket'].value_counts(normalize=True).to_dict()

        # STRATEGY PERFORMANCE
        pnl = trades_slice['pnl_baseline']
        win_rate = (pnl > 0).mean()
        avg_pnl   = pnl.mean()
        gross_win = pnl[pnl>0].sum()
        gross_loss= -pnl[pnl<0].sum()
        pf = gross_win / gross_loss if gross_loss>0 else np.nan

        # max drawdown on the equity curve of this slice
        eq = pnl.cumsum()
        mdd = (eq.cummax() - eq).max()

        periods.append({
            'period_start': start,
            'period_end':   end-1,
            'n_trades':     len(trades_slice),
            'avg_atr':      avg_atr,
            'avg_slope':    avg_slope,
            **{f'vol_{k}':v for k,v in vol_dist.items()},
            **{f'slope_{k}':v for k,v in slope_dist.items()},
            'win_rate':     win_rate,
            'avg_pnl':      avg_pnl,
            'profit_factor':pf,
            'max_drawdown': mdd
        })

    return pd.DataFrame(periods)


def check_correlations(df):
    df['equity_return'] = df['pnl_dynamic'].pct_change().fillna(0)


# 3. Correlaciones simples
    # ------------------------------------------------------------------
    corr_matrix = df[['equity_return', 'price_return', 'volatility', 'rsi', 'ADX14','kf_slope','atr_1h']].corr()
    print("Matriz de correlación:")
    print(corr_matrix['equity_return'].sort_values())
