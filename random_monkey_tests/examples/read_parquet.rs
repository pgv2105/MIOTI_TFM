// fast_backtest/examples/read_parquet.rs
// ------------------------------------------------------------------
// Lee un Parquet con Polars y muestra un resumen rápido.
// Corrige problemas de tipos con PlSmallStr y elimina código duplicado.
// ------------------------------------------------------------------

use polars::prelude::*;
use std::collections::HashSet;
use std::env;
use std::path::Path;

fn main() -> PolarsResult<()> {
    // 1️⃣  Ruta al Parquet (primer argumento CLI)
    let path_str = env::args()
        .nth(1)
        .expect("Usage: read_parquet <path_to_dataset_Y.parquet>");
    let path = Path::new(&path_str);
    if !path.exists() {
        panic!("File not found: {}", path.display());
    }

    println!("📂  Leyendo {} …", path.display());

    // 2️⃣  Carga el DataFrame (modo lazy ➜ collect)
    let df = LazyFrame::scan_parquet(path, Default::default())?.collect()?;
    println!("👉  Shape: {:?}", df.shape());

    // 3️⃣  Muestra las primeras 3 filas y 8 primeras columnas
    let col_names = df.get_column_names();
    let first_cols: Vec<&str> = col_names.iter().take(8).map(|s| s.as_ref()).collect();
    let head_partial = df.head(Some(3)).select(first_cols)?;
    println!("\nPrimeras filas (parciales):\n{:#?}", head_partial);

    // 4️⃣  Verifica que existan las 120 columnas bid_*
    let name_set: HashSet<&str> = col_names.iter().map(|s| s.as_ref()).collect();
    let n_bids = (0..120)
        .filter(|i| {
            let name = format!("bid_{}", i);
            name_set.contains(name.as_str())
        })
        .count();
    println!("\n✅  Columnas bid_X detectadas: {}", n_bids);

    Ok(())
}
