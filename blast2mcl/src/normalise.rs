/// Score normalisation — two strategies matching OrthoFinder Python:
///
/// • `normalise_scores`   (default, `v2_scores=false`)
///     Fits a log-linear model  log10(score) = a·log10(Li·Lj) + b  on the top
///     95th-percentile hits per length bin, then scales the matrix by
///     10^{-b} · Li^{-a} · Lj^{-a}.  This is a port of `scnorm.NormaliseScores`.
///
/// • `normalised_bit_score` (`--scores-v2`)
///     Simple diagonal scaling:  B'[i,j] = B[i,j] / sqrt(Li[i] · Lj[j]).
///     Matches `WaterfallMethod.NormalisedBitScore`.

use crate::sparse::CsrMatrix;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Default normalisation (log-linear curve fit).
///
/// Returns the normalised matrix, or an empty matrix of the same shape if
/// there are not enough hits to fit the model (matching Python's fallback).
pub fn normalise_scores(
    mut matrix: CsrMatrix,
    lengths_i: &[f32],
    lengths_j: &[f32],
) -> CsrMatrix {
    if matrix.nnz() == 0 {
        return matrix;
    }

    // Collect (L_product, score) for every non-zero entry.
    let mut ls: Vec<(f32, f32)> = Vec::with_capacity(matrix.nnz());
    for i in 0..matrix.nrows {
        for (j, score) in matrix.row_iter(i) {
            let l = lengths_i[i] * lengths_j[j as usize];
            if l > 0.0 && score > 0.0 {
                ls.push((l, score));
            }
        }
    }

    if ls.len() < 2 {
        eprintln!(
            "WARNING: Too few hits to normalise scores ({} hits). Returning empty matrix.",
            ls.len()
        );
        return CsrMatrix::empty(matrix.nrows, matrix.ncols);
    }

    // Take the top 95th-percentile of scores within each length bin.
    let filtered = top_percentile_by_length(&mut ls, 95.0);

    if filtered.len() <= 1 {
        eprintln!("WARNING: Too few top-percentile hits to fit normalisation model.");
        return CsrMatrix::empty(matrix.nrows, matrix.ncols);
    }

    // Fit: log10(score) = a · log10(L) + b
    let (a, b) = fit_log_linear(&filtered);

    // Apply: B'[i,j] = 10^{-b} · Li[i]^{-a} · B[i,j] · Lj[j]^{-a}
    let row_scale: Vec<f32> = lengths_i
        .iter()
        .map(|&l| if l > 0.0 { 10f32.powf(-b) * l.powf(-a) } else { 0.0 })
        .collect();
    let col_scale: Vec<f32> = lengths_j
        .iter()
        .map(|&l| if l > 0.0 { l.powf(-a) } else { 0.0 })
        .collect();

    matrix.scale_rows_inplace(&row_scale);
    matrix.scale_cols_inplace(&col_scale);
    matrix
}

/// V2 normalisation (simple sqrt scaling).
pub fn normalised_bit_score(
    mut matrix: CsrMatrix,
    lengths_i: &[f32],
    lengths_j: &[f32],
) -> CsrMatrix {
    if matrix.nnz() == 0 {
        return matrix;
    }
    let row_scale: Vec<f32> = lengths_i
        .iter()
        .map(|&l| if l > 0.0 { l.powf(-0.5) } else { 0.0 })
        .collect();
    let col_scale: Vec<f32> = lengths_j
        .iter()
        .map(|&l| if l > 0.0 { l.powf(-0.5) } else { 0.0 })
        .collect();
    matrix.scale_rows_inplace(&row_scale);
    matrix.scale_cols_inplace(&col_scale);
    matrix
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Sort `(L, score)` pairs by L, bin into groups, keep top `percentile`%
/// per bin.  Returns a flat Vec of the surviving (L, score) pairs.
///
/// Bin sizes mirror Python: 1000 if >5000 total, 200 if >1000, else 20.
fn top_percentile_by_length(ls: &mut Vec<(f32, f32)>, percentile: f32) -> Vec<(f32, f32)> {
    let n = ls.len();
    if n < 100 {
        // Not enough data to bin — return everything.
        return ls.clone();
    }

    ls.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let bin_size = if n > 5000 { 1000 } else if n > 1000 { 200 } else { 20 };
    let n_bins = n / bin_size;

    let mut out = Vec::new();
    for b in 0..n_bins {
        let first = b * bin_size;
        let last = ((b + 1) * bin_size).min(n);
        let slice = &ls[first..last];

        let cutoff = percentile_f32(
            &slice.iter().map(|&(_, s)| s).collect::<Vec<_>>(),
            percentile,
        );
        for &(l, s) in slice {
            if s >= cutoff {
                out.push((l, s));
            }
        }
    }
    out
}

/// p-th percentile of a slice of f32 values (linear interpolation).
fn percentile_f32(vals: &[f32], p: f32) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (p / 100.0) * (sorted.len() as f32 - 1.0);
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f32;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// OLS fit of  Y = a·X + b  where X = log10(L), Y = log10(score).
///
/// Returns (a, b).  Uses the closed-form solution (identical to what
/// scipy.optimize.curve_fit gives for a linear model).
fn fit_log_linear(pairs: &[(f32, f32)]) -> (f32, f32) {
    let n = pairs.len() as f32;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;

    for &(l, s) in pairs {
        if l <= 0.0 || s <= 0.0 {
            continue;
        }
        let x = (l as f64).log10();
        let y = (s as f64).log10();
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }

    let n = n as f64;
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        // Degenerate: all lengths identical — fall back to no-op scaling.
        return (0.0, 0.0);
    }
    let a = (n * sxy - sx * sy) / denom;
    let b = (sy - a * sx) / n;
    (a as f32, b as f32)
}

// ---------------------------------------------------------------------------
// FASTA sequence-length reader (needed before normalisation)
// ---------------------------------------------------------------------------

/// Read sequence lengths (in amino acids) from a FASTA file.
///
/// Sequences are assumed to appear in the same order as OrthoFinder assigned
/// numeric IDs, i.e. the k-th `>` header corresponds to sequence index k.
pub fn read_sequence_lengths(fasta_path: &std::path::Path) -> Vec<f32> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(fasta_path)
        .unwrap_or_else(|e| panic!("Cannot open FASTA {:?}: {}", fasta_path, e));
    let mut lengths: Vec<f32> = Vec::new();
    let mut current: u32 = 0;
    let mut in_seq = false;

    for line in BufReader::new(file).lines() {
        let line = line.expect("IO error reading FASTA");
        if line.starts_with('>') {
            if in_seq {
                lengths.push(current as f32);
                current = 0;
            }
            in_seq = true;
        } else if in_seq {
            current += line.trim_end().len() as u32;
        }
    }
    if in_seq {
        lengths.push(current as f32);
    }
    lengths
}
