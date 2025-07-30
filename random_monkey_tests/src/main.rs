// src/main.rs
// -------------------------------------------------------------
// Punto de entrada del programa. Desde aquí invocamos el loader
// para leer dataset_Y.parquet y mostramos un par de valores.
// -------------------------------------------------------------

mod data_loader;                    
mod simulators;

use polars::prelude::PolarsResult;   // para el tipo Result
use crate::data_loader::load_dataset_y;

fn main() -> PolarsResult<()> {
    // Ruta relativa al Parquet dentro de tu proyecto
    let ds = load_dataset_y("data/dataset_Y.parquet")?;

    println!("Filas totales: {}", ds.open.len());
    println!("Primer open   : {}", ds.open[0]);
    println!("Primer bid[0] : {}", ds.bids[0][0]);

    Ok(())
}
