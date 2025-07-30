import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

def results_analysis(executed_trades, balance_history, buyAndHold):

    matplotlib.use('TkAgg')

    # Suponemos que tienes executed_trades y balance_history
    df = pd.DataFrame(executed_trades)

    # Convertimos datos necesarios
    df['cumulative_pnl'] = df['pnl_baseline'].cumsum()
    df['trade_index'] = range(len(df))
    df['is_win'] = df['pnl_baseline'] > 0

    # 1️⃣ Evolución del balance
    plt.figure(figsize=(12, 4))
    plt.plot(balance_history, label='Balance')
    plt.plot(buyAndHold, label='Price EURUSD')
    plt.title("Balance Over Time")
    plt.xlabel("Time step (10-min intervals)")
    plt.ylabel("Balance (€)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

    # 2️⃣ Distribución del PnL
    plt.figure(figsize=(10, 4))
    plt.hist(df['pnl_baseline'], bins=30, edgecolor='black', color='skyblue')
    plt.axvline(df['pnl_baseline'].mean(), color='red', linestyle='--', label=f'Mean: {df["pnl_baseline"].mean():.2f} €')
    plt.title("Distribution of Trade PnL")
    plt.xlabel("PnL (€)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    df_trades = pd.DataFrame(executed_trades)
    # 1. Filtrar solo los trades con pérdidas
    loss_trades = df_trades[df_trades['win_baseline'] == 0]

    # 2. Convertir la duración a numérico (por si acaso)
    loss_trades['duration'] = pd.to_numeric(loss_trades['duration'], errors='coerce')

    # 3. Crear el histograma
    plt.figure(figsize=(10, 5))
    sns.histplot(loss_trades['duration'], bins=15, color='tomato', edgecolor='black')
    plt.title("Distribution of Losing Trade Durations")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Number of Losing Trades")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    '''
    # 3️⃣ Real vs Simulated Trades
    plt.figure(figsize=(10, 4))
    df.groupby('is_paper_trade')['pnl'].mean().plot(kind='bar', color=['green', 'orange'])
    plt.xticks([0, 1], ['Real', 'Simulated'])
    plt.title("Average PnL: Real vs Simulated Trades")
    plt.ylabel("Avg. PnL (€)")
    plt.grid(axis='y', linestyle='--')
    plt.show()
    
    '''

    # 4️⃣ Distribución de duración de trades
    plt.figure(figsize=(10, 4))
    plt.hist(df['duration'], bins=12, color='purple', edgecolor='black')
    plt.title("Trade Duration Distribution")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Number of trades")
    plt.grid(True)
    plt.show()

    # 5️⃣ Evolución del win rate acumulado
    df['win_rate_running'] = df['is_win'].expanding().mean()
    plt.figure(figsize=(10, 4))
    plt.plot(df['trade_index'], df['win_rate_running'], label='Win rate (cumulative)', color='darkblue')
    plt.axhline(0.5, color='red', linestyle='--', label='50% baseline')
    plt.title("Cumulative Win Rate Over Time")
    plt.xlabel("Trade index")
    plt.ylabel("Win rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 6️⃣ Cumulative PnL curve
    plt.figure(figsize=(10, 4))
    plt.plot(df['trade_index'], df['cumulative_pnl'], label='Cumulative PnL', color='green')
    plt.title("Cumulative PnL")
    plt.xlabel("Trade index")
    plt.ylabel("Cumulative PnL (€)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 7) Asegúrate de que entry_time es datetime
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    # Extrae la hora del día
    df['hour'] = df['entry_time'].dt.hour
    plt.figure(figsize=(8,4))
    plt.hist(df['hour'], bins=range(0,25), edgecolor='black')
    plt.xticks(range(0,24))
    plt.xlabel('Hora del día')
    plt.ylabel('Número de operaciones abiertas')
    plt.title('Distribución de horas de entrada')
    plt.show()
