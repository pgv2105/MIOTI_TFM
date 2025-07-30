// src/data_loader.rs
// -----------------------------------------------------------------------------
// Paso A: Cargar el Parquet a estructuras de datos simples (Vec / Vec<Vec<f64>>)
// -----------------------------------------------------------------------------
// Este módulo expone una única función:
//     pub fn load_dataset_y(path: &str) -> PolarsResult<DatasetY>
//
// Uso típico desde un binario o test:
//     let ds = data_loader::load_dataset_y("data/dataset_Y.parquet")?;
//     println!("open[0] = {}  bids[0][0] = {}", ds.open[0], ds.bids[0][0]);
//
// -----------------------------------------------------------------------------
// Requisitos en Cargo.toml (ya añadidos):
// polars = { version = "0.48", features = ["lazy", "parquet"] }
// -----------------------------------------------------------------------------

use polars::prelude::*;

/// Estructura in‑memory con los campos que el back‑tester necesita.
#[derive(Debug)]
pub struct DatasetY {
    /// Vector con el precio Open de cada barra (f64)
    pub open: Vec<f64>,
    /// Matriz (Vec de Vec) con 120 bids por barra.
    /// bids[i][j] = precio j‑ésimo de la barra i.
    pub bids: Vec<Vec<f64>>, // len = n_bars, cada inner vec len = 120
}

/// Carga `dataset_Y.parquet` y lo transforma en `DatasetY`.
///
/// * `path` – ruta al fichero Parquet.
///
/// Devuelve error de Polars si el fichero no existe o no contiene las
/// columnas esperadas ("Open" y "bid_0".."bid_119").
pub fn load_dataset_y<P: AsRef<std::path::Path>>(path: P) -> PolarsResult<DatasetY> {
    // 1️⃣  Cargar todo el DataFrame en memoria (32‑bit float podría ahorrar RAM
    //      si fuera necesario; partimos con f64 para fidelidad).
    let df = LazyFrame::scan_parquet(path.as_ref(), Default::default())?.collect()?;

    // 2️⃣  Extraer columna "Open" como Vec<f64>
    let open_series = df.column("Open")?.f64()?;
    let open: Vec<f64> = open_series.into_no_null_iter().collect();

    // 3️⃣  Extraer las 120 columnas bid_*
    let mut bids: Vec<Vec<f64>> = Vec::with_capacity(open.len());
    // Pre‑alocar cada inner vec para evitar reallocs repetidas
    bids.resize(open.len(), Vec::with_capacity(120));

    for j in 0..120 {
        let col = df.column(&format!("bid_{}", j))?.f64()?;
        for (i, v) in col.into_no_null_iter().enumerate() {
            // i‑ésima barra, j‑ésimo bid
            bids[i].push(v);
        }
    }

    Ok(DatasetY { open, bids })
}
