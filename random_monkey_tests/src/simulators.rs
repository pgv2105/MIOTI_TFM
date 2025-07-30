// src/simulators.rs
// -----------------------------------------------------------------------------
// Simuladores de trade portados de Python.
//   • simulate_trade_partial           (long intra‑barra, trailing stop/TP)
//   • simulate_short_trade_partial
//   • simulate_forex_closure_ibkr      (cierre de múltiples longs EURUSD)
//   • simulate_forex_short_closure_ibkr
// -----------------------------------------------------------------------------

/// Simula un trade LONG durante `intervals_passed` intervalos (cada intervalo = 20 bids).
/// Devuelve (precio_actual/ejecución, sigue_abierto, idx_bid_final).
pub fn simulate_trade_partial(
    prices: &[f64],
    open_price: f64,
    intervals_passed: usize,
) -> (f64, bool, usize) {
    let target_factor = 0.062;   // 0.62 %
    let mut current_factor = 0.047; // 0.05 %
    let mut switched = false;

    let mut stop_loss = open_price * (1.0 - current_factor / 100.0);
    let mut max_price = open_price;
    let target_price = open_price * (1.0 + target_factor / 100.0);

    let mut last_idx = 0;
    for (i, &bid) in prices.iter().take(intervals_passed * 20).enumerate() {
        last_idx = i;
        // Stop‑loss
        if bid <= stop_loss {
            return (stop_loss, false, i);
        }
        // Activa trailing si llega a target
        if !switched && bid >= target_price {
            switched = true;
            current_factor = 0.014; // STOP_LOSS_FACTOR_ADJUSTED
            stop_loss = bid * (1.0 - current_factor / 100.0);
        }
        // Trailing update
        if switched && bid > max_price {
            max_price = bid;
            stop_loss = max_price * (1.0 - current_factor / 100.0);
        }
    }
    (prices[last_idx], true, last_idx)
}

/// Versión SHORT del simulador.
pub fn simulate_short_trade_partial(
    prices: &[f64],
    open_price: f64,
    intervals_passed: usize,
) -> (f64, bool, usize) {
    let mut current_factor = 0.047; // 0.06 %
    let target_factor = 0.062;       // 0.065 %
    let mut switched = false;

    let mut stop_loss = open_price * (1.0 + current_factor / 100.0);
    let mut min_price = open_price;
    let target_price = open_price * (1.0 - target_factor / 100.0);

    let mut last_idx = 0;
    for (i, &bid) in prices.iter().take(intervals_passed * 20).enumerate() {
        last_idx = i;
        // Stop‑loss
        if bid >= stop_loss {
            return (stop_loss, false, i);
        }
        // Activa trailing al llegar a target
        if !switched && bid <= target_price {
            switched = true;
            current_factor = 0.014;
            stop_loss = bid * (1.0 + current_factor / 100.0);
        }
        // Trailing update
        if switched && bid < min_price {
            min_price = bid;
            stop_loss = min_price * (1.0 + current_factor / 100.0);
        }
    }
    (prices[last_idx], true, last_idx)
}

/// Cierre de múltiples posiciones LONG en EURUSD (IBKR) con spread & slippage.
/// Devuelve PnL neto (USD).
pub fn simulate_forex_closure_ibkr(
    usd_entries: &[f64],
    entry_prices: &[f64],
    exit_price: f64,
    spread: f64,
    slippage: f64,
    commission_pips: f64,
    commission_min: f64,
) -> f64 {
    let mut total_eur = 0.0;
    let mut total_usd_invested = 0.0;
    let mut total_commission = 0.0;

    for (&usd, &entry_price) in usd_entries.iter().zip(entry_prices) {
        let executed_entry = entry_price + spread / 2.0 + slippage;
        let eur = usd / executed_entry;
        total_eur += eur;
        total_usd_invested += usd;
        let mut comm = usd * commission_pips * 2.0;
        if comm < commission_min {
            comm = commission_min;
        }
        total_commission += comm;
    }

    let executed_exit = exit_price - spread / 2.0 - slippage;
    let usd_returned = total_eur * executed_exit;
    usd_returned - total_usd_invested - total_commission
}

/// Cierre de múltiples posiciones SHORT en EURUSD (IBKR).
pub fn simulate_forex_short_closure_ibkr(
    usd_entries: &[f64],
    entry_prices: &[f64],
    exit_price: f64,
    spread: f64,
    slippage: f64,
    commission_pips: f64,
    commission_min: f64,
) -> f64 {
    let mut total_eur_shorted = 0.0;
    let mut total_usd_received = 0.0;
    let mut total_commission = 0.0;

    for (&usd, &entry_price) in usd_entries.iter().zip(entry_prices) {
        let executed_entry = entry_price - spread / 2.0 - slippage; // venta
        let eur_shorted = usd / executed_entry;
        total_eur_shorted += eur_shorted;
        total_usd_received += usd;
        let mut comm = usd * commission_pips * 2.0;
        if comm < commission_min {
            comm = commission_min;
        }
        total_commission += comm;
    }

    let executed_exit = exit_price + spread / 2.0 + slippage; // recompra
    let usd_needed = total_eur_shorted * executed_exit;
    total_usd_received - usd_needed - total_commission
}
