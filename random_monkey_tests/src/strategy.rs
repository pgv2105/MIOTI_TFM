// src/strategy.rs

use rand::thread_rng;
use rand::Rng;

use crate::data_loader::DatasetY;
use crate::simulators::{
    simulate_trade_partial, simulate_short_trade_partial,
    simulate_forex_closure_ibkr, simulate_forex_short_closure_ibkr
};

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::BuildHasherDefault;

use ordered_float::OrderedFloat;  

/// Lado de la operación
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TradeSide { Long, Short }

/// Representación de un trade abierto
#[derive(Clone, Debug)]
pub struct Trade {
    pub entry_idx : usize,
    pub orig_entry_idx : usize,
    pub open_mid  : f64,
    pub open_orig : f64,
    pub prices    : Vec<f64>,
    pub exposure  : f64,
    pub cur_price : f64,
    pub cur_pnl   : f64,
    pub side      : TradeSide,
    pub entry_ts  : usize,
    pub paper     : bool,

    // auxiliares para cierre
    pub interval_passed: usize,
    pub executed_price : f64,
}

impl Default for Trade {
    fn default() -> Self {
        Self {
            entry_idx: 0,
            orig_entry_idx: 0,
            open_mid: 0.0,
            open_orig: 0.0,
            prices: Vec::new(),
            exposure: 0.0,
            cur_price: 0.0,
            cur_pnl: 0.0,
            side: TradeSide::Long,
            entry_ts: 0,
            paper: true,
            interval_passed: 0,
            executed_price: 0.0,
        }
    }
}

/// Lógica de exposición con apalancamiento dinámico
pub fn get_trade_exposure(
    open_price: f64,
    open_trades: &[Trade],
    balance: f64,
    side: TradeSide,
    leverage: f64,
) -> f64 {
    // 1) Sin posiciones abiertas → (balance/6)*leverage
    if open_trades.is_empty() {
        return (balance / 6.0) * leverage;
    }

    // 2) Última operación abierta (= mayor entry_idx)
    let last = open_trades
        .iter()
        .max_by_key(|t| t.entry_idx)
        .expect("slice non‑empty; qed");

    // 3) ¿Mismo lado y precio en contra?
    let price_cond = match side {
        TradeSide::Long  => last.open_mid > open_price,
        TradeSide::Short => last.open_mid < open_price,
    };
    let same_side = last.side == side;

    // 4) Nuevo notional según la condición
    let mut new_notional = if same_side && price_cond {
        // Python:  exposure_margin = last.exposure * 1.5 / leverage
        //          return exposure_margin * leverage = last.exposure * 1.5
        last.exposure * 1.5
    } else {
        (balance / 6.0) * leverage
    };

    // 5) Límite total permitido
    let used_notional: f64 = open_trades.iter().map(|t| t.exposure).sum();
    let max_notional      = balance * leverage;

    if used_notional + new_notional > max_notional {
        let rest_notional = max_notional - used_notional;           // lo que queda
        let rest_margin   = rest_notional / leverage;               // …en margen

        new_notional = if rest_margin > 10_000.0 {
            rest_notional                                          // se permite
        } else {
            0.0                                                   // demasiado poco
        };
    }

    new_notional   // ya en notional (exposure * leverage)
}

/// Alias con hasher determinista
type DHashMap<K, V> = HashMap<K, V, BuildHasherDefault<DefaultHasher>>;

/// Agrupa las operaciones por `(precio_final, lado)`
/// devolviendo un `HashMap` cuyo orden de iteración es reproducible.
pub fn group_trades_by_final_exec(
    trades: &[Trade],
) -> DHashMap<(OrderedFloat<f64>, TradeSide), Vec<Trade>> {
    // Mapa con hasher fijo (k0 = k1 = 0)
    let mut map: DHashMap<(OrderedFloat<f64>, TradeSide), Vec<Trade>> =
        DHashMap::default();

    for t in trades {
        let key = (OrderedFloat(t.cur_price), t.side);
        map.entry(key).or_default().push(t.clone());
    }

    map
}

/// Main backtest
pub fn run_strategy(
    data: &DatasetY,
    predictions: &[u8], 
) -> Vec<f64> {
    const INITIAL_CAPITAL: f64 = 70_000.0;
    const LEVERAGE:       f64 = 10.0;
    const SPREAD:         f64 = 0.00003;
    const COMM_P_SIDE:    f64 = 0.00002;
    const SLIPPAGE:       f64 = 0.00001;
    const LOG_INTERVAL:   usize = 10_000;

    let mut balance = INITIAL_CAPITAL;
    let mut equity = Vec::with_capacity(data.open.len());
    equity.push(balance);


    let mut open_trades: Vec<Trade> = Vec::new();
    let mut executed_trades: Vec<Trade> = Vec::new();

    let mut rng = thread_rng();


    for i in 1..data.open.len() {
        let price_open = data.open[i];
        let bids = &data.bids[i];

        // 1) Update & close open trades
        let mut updated: Vec<Trade> = Vec::new();
        let mut to_close: Vec<Trade> = Vec::new();
        let mut jump_trade = false;

        for mut t in open_trades.drain(..) {
            let intervals = i - t.entry_idx;

            // Tick virtual
            let (px, still_open, idx_final) = match t.side {
                TradeSide::Long  =>
                    simulate_trade_partial(&t.prices, t.open_mid, intervals),
                TradeSide::Short =>
                    simulate_short_trade_partial(&t.prices, t.open_mid, intervals),
            };

            t.cur_price = px;

            if still_open && intervals < 6 {
                // Sigue viva
                updated.push(t);
            } else {
                // Posible re‑apertura aleatoria
                let keep_open = rng.gen_ratio(2, 3);   // 66 %
                // let keep_open = i % 3 != 0; 
                if keep_open {
                    t.entry_idx   = i;
                    t.open_mid    = t.prices[idx_final];
                    let new_seg   = 20 * (6 - intervals);
                    let mut new_px= t.prices[idx_final..].to_vec();
                    new_px.extend_from_slice(
                        &bids[new_seg .. new_seg + idx_final]);
                    t.prices      = new_px;
                    updated.push(t);
                    jump_trade = true;
                    continue;               // no se cierra todavía
                }

                // Cierre normal
                if !t.paper {
                    t.cur_pnl = match t.side {
                        TradeSide::Long  => simulate_forex_closure_ibkr(
                            &[t.exposure], &[t.open_orig], px,
                            SPREAD, SLIPPAGE, COMM_P_SIDE, 2.0),
                        TradeSide::Short => simulate_forex_short_closure_ibkr(
                            &[t.exposure], &[t.open_orig], px,
                            SPREAD, SLIPPAGE, COMM_P_SIDE, 2.0),
                    };
                }
                t.executed_price  = px;
                t.interval_passed = intervals;
                to_close.push(t);
            }
        }
        // agrupar cierres
        for ((price_of, side), group) in group_trades_by_final_exec(&to_close) {
            let price = price_of.0;
            let exposures: Vec<f64> =
                group.iter().map(|t| t.exposure).collect();
            let opens: Vec<f64> =
                group.iter().map(|t| t.open_orig).collect();
            //let indexes: Vec<String> = 
              //  group.iter().map(|t| t.orig_entry_idx.to_string()).collect();

            let pnl = match side {
                TradeSide::Long  =>
                    simulate_forex_closure_ibkr(
                        &exposures, &opens, price,
                        SPREAD, SLIPPAGE, COMM_P_SIDE, 2.0),
                TradeSide::Short =>
                    simulate_forex_short_closure_ibkr(
                        &exposures, &opens, price,
                        SPREAD, SLIPPAGE, COMM_P_SIDE, 2.0),
            };
            balance += pnl;
            // los ejecutados pueden recogerse si quieres
            executed_trades.extend(group.into_iter());

            // if i < 18000 {println!("BALANCE: {:.2} -- exec price {:.6}, -- exposure {:?}, -- opens {:?} -- index  [{}]", balance, price, exposures, opens, indexes.join(", "));}
        }
        open_trades = updated;

        if jump_trade {
            equity.push(balance);
            continue;
        }
        /*        if i == 16299 {
            println!("--- Estado en i = 16299 ---");
            for (idx, t) in open_trades.iter().enumerate() {
                println!(
                    "Trade {} -> entry_idx: {}, open_mid: {:.6}, exposure: {:.2}, side: {:?}",
                    idx,
                    t.entry_idx,
                    t.open_mid,
                    t.exposure,
                    t.side
                );
            }
        }
        */

        // ================================================
        // 2) NUEVA ENTRADA (si no hemos re‑abierto arriba)
        // ================================================
        if open_trades.len() < 6 {
            match predictions[i] {
                1 => {   // LONG
                    let exp = get_trade_exposure(
                        price_open, &open_trades, balance,
                        TradeSide::Long, LEVERAGE);
                    if exp > 0.0 {
                        open_trades.push(Trade {
                            entry_idx : i,
                            orig_entry_idx: i,
                            open_mid  : price_open,
                            open_orig : price_open,
                            prices    : bids.clone(),
                            exposure  : exp,
                            side      : TradeSide::Long,
                            entry_ts  : i,
                            paper     : false,
                            ..Default::default()
                        });
                    }
                }
                2 => {   // SHORT
                    let exp = get_trade_exposure(
                        price_open, &open_trades, balance,
                        TradeSide::Short, LEVERAGE);
                    if exp > 0.0 {
                        open_trades.push(Trade {
                            entry_idx : i,
                            open_mid  : price_open,
                            open_orig : price_open,
                            prices    : bids.clone(),
                            exposure  : exp,
                            side      : TradeSide::Short,
                            entry_ts  : i,
                            paper     : false,
                            ..Default::default()
                        });
                    }
                }
                _ => {} // 0 = nada
            }
        }

        if i % LOG_INTERVAL == 0 {
            log::debug!(
                "bar {} open trades {} balance {:.2}",
                i,
                open_trades.len(),
                balance
            );
        }

        equity.push(balance);
    }

    equity
}
