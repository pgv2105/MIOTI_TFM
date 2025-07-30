use crabby_monkeys::{data_loader::load_dataset_y, strategy::run_strategy};

fn main() -> polars::prelude::PolarsResult<()> {
    env_logger::init();
    let ds = load_dataset_y("data/dataset_Y.parquet")?;
    // señales dummy
    let len = ds.open.len();
    let long_sig = vec![0u8; len];
    let short_sig = vec![0u8; len];
    let curve = run_strategy(&ds, &long_sig, &short_sig);
    println!("Curva len {}  último balance {}", curve.len(), curve.last().unwrap());
    Ok(())
}
