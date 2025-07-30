// examples/run_monkeys.rs

use crabby_monkeys::{
    data_loader::{load_dataset_y, DatasetY},
    strategy::run_strategy
};
use rand::prelude::*;
use rayon::prelude::*;
use serde_json::from_str;
use std::time::Instant;
use anyhow::Result;
use std::fs;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ----------------------------------------------------
// 1) Generador de predicciones aleatorias
// ----------------------------------------------------
fn generate_random_predictions(len: usize) -> Vec<u8> {
    let n_zeros = len / 2;               // 50 %
    let n_ones  = len / 4;               // 25 %
    let n_twos  = len - n_zeros - n_ones; // resto = 25 %

    let mut preds = Vec::with_capacity(len);
    preds.extend(std::iter::repeat(0u8).take(n_zeros));
    preds.extend(std::iter::repeat(1u8).take(n_ones));
    preds.extend(std::iter::repeat(2u8).take(n_twos));

    preds.shuffle(&mut thread_rng());    // mezcla in‑place
    preds
}


fn main() -> Result<()> {
    env_logger::init();

    // --- 1) Carga y umbraliza ---
    let ds: DatasetY = load_dataset_y("data/dataset_Y.parquet")?;
    let total_bars = ds.open.len();

    // --- 2) Backtest real ---
    let real_curve: Vec<f64> = from_str(&fs::read_to_string("data/balance_final.json")?)?;
    let test_bars = real_curve.len();
    let start = total_bars - test_bars;

    let ds_test = DatasetY {
        open: ds.open[start..].to_vec(),
        bids: ds.bids[start..].to_vec(),
    };

    let real_initial = real_curve[0];
    let real_final = *real_curve.last().unwrap();
    let real_return_pct = (real_final - real_initial) / real_initial * 100.0;

    println!("Real final balance: {:.2}", real_final);
    println!("Real return (%): {:.2}%", real_return_pct);

    // --- Max drawdown real ---
    let real_max_dd = {
        let mut peak = f64::NEG_INFINITY;
        real_curve.iter().fold(0.0f64, |dd, &b| {
            peak = peak.max(b);
            dd.max((peak - b) / peak)
        })
    };
    println!("Real max drawdown: {:.4}", real_max_dd);

    // Ejemplo de ejecución aleatoria
    let preds = generate_random_predictions(test_bars);
    let curve = run_strategy(&ds_test, &preds);
    let comp_final = *curve.last().unwrap();
    println!("COMPARATION (ejemplo random monkey): {:.2}", comp_final);

    // --- 3) Monos en paralelo con contador ---
    const N_MONKEYS: usize = 2_000_000;
    const PROGRESS_STEP: usize = 20_000;
    let progress = Arc::new(AtomicUsize::new(0));
    let start_time = Instant::now();

    let stats: Vec<(f64, f64)> = (0..N_MONKEYS).into_par_iter().map_init(
        || Arc::clone(&progress),
        |progress, _| {
            let current = progress.fetch_add(1, Ordering::Relaxed);
            if current % PROGRESS_STEP == 0 {
                println!("→ Monos simulados: {}", current);
            }

            let preds = generate_random_predictions(test_bars);
            let curve = run_strategy(&ds_test, &preds);

            let initial = curve[0];
            let final_balance = *curve.last().unwrap();
            let return_pct = (final_balance - initial) / initial * 100.0;

            let dd = {
                let mut peak = f64::NEG_INFINITY;
                curve.iter().fold(0.0f64, |mx, &b| {
                    peak = peak.max(b);
                    mx.max((peak - b) / peak)
                })
            };

            (return_pct, dd)
        }
    ).collect();

    println!("Monos en {:.2?}", start_time.elapsed());

    // --- 4) Estadísticas de monos ---
    let (returns, dds): (Vec<_>, Vec<_>) = stats.into_iter().unzip();
    let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
    let mut r_sorted = returns.clone();
    r_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_ret = r_sorted[r_sorted.len() / 2];
    let mean_dd = dds.iter().sum::<f64>() / dds.len() as f64;

    println!("Monkey return μ/med: {:.2}% / {:.2}%", mean_ret, med_ret);
    println!("Monkey maxDD μ:      {:.4}", mean_dd);

    // --- 5) Exportación de resultados ---
    fs::write(
        "data/monkey_returns.json",
        serde_json::to_string(&returns)?,
    )?;
    println!("Monkey returns saved to data/monkey_returns.json");

    Ok(())
}
