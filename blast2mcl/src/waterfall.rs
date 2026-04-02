/// Port of OrthoFinder's WaterfallMethod: best hits, RBH, and cognate connection.
///
/// # Algorithm sketch (for species i)
///
/// **Pass 1 — ProcessBlastHits** (per query species `i`)
///   For each target species `j`:
///     1. Read normalised B[i][j].
///     2. Track, for each query gene k, the best (max) score across all j ≠ i.
///     3. Build BH[i][j]: indicator matrix where B[i][j][k,l] ≥ best_score[k] - tol.
///   Save B[i][*] and BH[i][*] to disk.
///
/// **Pass 2 — ConnectCognates** (per query species `i`)
///   1. Load BH[i][j] and BH[j][i] for all j.
///   2. RBH[j] = BH[i][j] AND BH[j][i]ᵀ  (reciprocal best hits).
///   3. Load B[i][j] for all j.
///   4. mostDistant[k] = min over all j where RBH[j][k,·]≠0 of B[i][j][k, rbh_col].
///      Genes without any RBH fall back to bestHit[k] + ε.
///   5. connect[i][j] = { (k,l) : B[i][j][k,l] ≥ mostDistant[k] }  (or k≠l if i==j).
///   Save connect[i][*] to disk.

use crate::sparse::{CooMatrix, CsrMatrix};

// ---------------------------------------------------------------------------
// Best-hit matrices   (GetBH_s)
// ---------------------------------------------------------------------------

/// Compute BH matrices for query species `i_spec`.
///
/// `b`  — slice of length n_species; b[j] = normalised B[i][j].
/// `i_spec` — index of the query species within the slice.
/// `tol` — tolerance for the "≥ best hit" comparison (default 1e-3).
///
/// Returns a Vec of n_species boolean (0/1) CSR matrices matching B[i][j] shape.
pub fn get_best_hits(b: &[CsrMatrix], i_spec: usize, tol: f32) -> Vec<CsrMatrix> {
    let n_seqs_i = b[0].nrows;
    let n_species = b.len();

    // --- Step 1: find best score for each gene k across all j ≠ i_spec ---
    // Initialise to -1 so that even score=0 hits can be detected.
    let mut best = vec![-1.0f32; n_seqs_i];
    for (j, bj) in b.iter().enumerate() {
        if j == i_spec {
            continue;
        }
        for k in 0..n_seqs_i {
            let m = bj.row_max(k);
            if m > best[k] {
                best[k] = m;
            }
        }
    }

    // --- Step 2: build BH[i][j] for every j (including self for paralogs) ---
    let mut h = Vec::with_capacity(n_species);
    for bj in b.iter() {
        let mut coo = CooMatrix::new(n_seqs_i, bj.ncols);
        for k in 0..n_seqs_i {
            let threshold = best[k] - tol; // best[k] == -1 when gene has no hits outside species
            for (l, v) in bj.row_iter(k) {
                if v > threshold {
                    coo.insert_one(k as u32, l);
                }
            }
        }
        h.push(coo.into_csr());
    }
    h
}

// ---------------------------------------------------------------------------
// ConnectCognates
// ---------------------------------------------------------------------------

/// Compute `connect[i][*]` matrices for species `i_spec`.
///
/// `bh_from_i` — BH[i][j] for all j  (loaded from disk after Pass 1).
/// `bh_to_i`   — BH[j][i] for all j  (each loaded from the Pass 1 of species j).
/// `b`          — normalised B[i][j] for all j.
/// `i_spec`     — species index within the slices.
/// `v2_scores`  — if true, uses the relative-RBH distance estimator (slower);
///               otherwise uses the simpler `GetMostDistant_s` method.
///
/// Returns Vec<CsrMatrix> of length n_species: the connect[i][j] matrices.
pub fn connect_cognates(
    bh_from_i: &[CsrMatrix],
    bh_to_i: &[CsrMatrix],
    b: &[CsrMatrix],
    i_spec: usize,
    v2_scores: bool,
) -> Vec<CsrMatrix> {
    // Build RBH[j] = BH[i][j] AND BH[j][i]ᵀ
    let rbh: Vec<CsrMatrix> = bh_from_i
        .iter()
        .zip(bh_to_i.iter())
        .map(|(from, to)| from.and_with_transpose(to))
        .collect();

    let most_distant = if v2_scores {
        get_most_distant_v2(&rbh, b, i_spec)
    } else {
        get_most_distant_simple(&rbh, b, i_spec)
    };

    connect_all_better_than_cutoff(b, &most_distant, i_spec)
}

// ---------------------------------------------------------------------------
// GetMostDistant_s  (default, v2_scores=false)
// ---------------------------------------------------------------------------

/// For each gene k in species i, find the score of its most-distant RBH across
/// all other species.  Genes without any RBH fall back to bestHit + ε.
///
/// This is a direct port of `WaterfallMethod.GetMostDistant_s`.
fn get_most_distant_simple(
    rbh: &[CsrMatrix],
    b: &[CsrMatrix],
    i_spec: usize,
) -> Vec<f32> {
    let n_seqs_i = b[0].nrows;
    let mut most_distant = vec![1e9f32; n_seqs_i];
    let mut best_hit = vec![0.0f32; n_seqs_i];

    for (k_spec, (rbh_j, bj)) in rbh.iter().zip(b.iter()).enumerate() {
        if k_spec == i_spec {
            continue;
        }
        // Update best_hit (max score to any gene in any other species).
        for k in 0..n_seqs_i {
            let m = bj.row_max(k);
            if m > best_hit[k] {
                best_hit[k] = m;
            }
        }
        // Update most_distant using RBH entries.
        for (k, l) in rbh_j.nonzero_pairs() {
            let score = bj.get(k, l);
            if score < most_distant[k] {
                most_distant[k] = score;
            }
        }
    }

    // Genes without any RBH: threshold = best hit in another species + epsilon.
    for k in 0..n_seqs_i {
        if most_distant[k] > 1e8 {
            most_distant[k] = best_hit[k] + 1e-6;
        }
    }
    most_distant
}

// ---------------------------------------------------------------------------
// GetMostDistant_s_estimate_from_relative_rbhs  (v2_scores=true)
// ---------------------------------------------------------------------------

/// Estimate the most-distant RBH score for each gene in species i by computing
/// pairwise conversion factors between species and extrapolating.
///
/// Port of `WaterfallMethod.GetMostDistant_s_estimate_from_relative_rbhs`.
fn get_most_distant_v2(
    rbh: &[CsrMatrix],
    b: &[CsrMatrix],
    i_spec: usize,
) -> Vec<f32> {
    let n_seqs_i = b[0].nrows;
    let n_species = b.len();

    // Build RBH_B[k]: score of the unique RBH from i to species k (0 if none).
    // We collect only non-self species (matching `if i != iSpec`).
    let non_self: Vec<usize> = (0..n_species).filter(|&j| j != i_spec).collect();
    let nsp_m1 = non_self.len();

    // Z[sp_idx, gene] = RBH score from i to species sp_idx for gene, else 0.
    let mut z = vec![vec![0.0f32; n_seqs_i]; nsp_m1];
    for (sp_idx, &j) in non_self.iter().enumerate() {
        for k in 0..n_seqs_i {
            // Single RBH: exactly one nonzero in row k of rbh[j].
            let start = rbh[j].row_ptr[k] as usize;
            let end = rbh[j].row_ptr[k + 1] as usize;
            if end - start == 1 {
                let l = rbh[j].col_idx[start] as usize;
                z[sp_idx][k] = b[j].get(k, l);
            }
        }
    }

    // Compute conversion matrix C[i,j] = p90 of z[i,:]/z[j,:] where both > 0.
    let p = 90.0_f32;
    let mut c = vec![vec![1.0f32; nsp_m1]; nsp_m1];
    for isp in 0..nsp_m1 {
        for jsp in 0..nsp_m1 {
            if isp == jsp {
                continue;
            }
            let ratios: Vec<f32> = (0..n_seqs_i)
                .filter(|&k| z[isp][k] > 0.0 && z[jsp][k] > 0.0)
                .map(|k| z[isp][k] / z[jsp][k])
                .collect();
            if ratios.is_empty() {
                c[isp][jsp] = 1.0;
            } else {
                c[isp][jsp] = percentile(&ratios, 100.0 - p);
            }
        }
    }

    // Column-wise minimum of C → conversion from each species' RBH to the
    // most-distant expected RBH.
    let col_min: Vec<f32> = (0..nsp_m1)
        .map(|jsp| {
            (0..nsp_m1)
                .map(|isp| c[isp][jsp])
                .fold(f32::INFINITY, f32::min)
        })
        .collect();

    let mut most_distant = vec![1e9f32; n_seqs_i];
    let mut best_hit = vec![0.0f32; n_seqs_i];

    for (j_idx, &j) in non_self.iter().enumerate() {
        let factor = col_min[j_idx];
        for k in 0..n_seqs_i {
            let m = b[j].row_max(k);
            if m > best_hit[k] {
                best_hit[k] = m;
            }
        }
        for (k, l) in rbh[j].nonzero_pairs() {
            let scaled = factor * b[j].get(k, l);
            if scaled < most_distant[k] {
                most_distant[k] = scaled;
            }
        }
    }

    for k in 0..n_seqs_i {
        if most_distant[k] > 1e8 {
            most_distant[k] = best_hit[k] + 1e-6;
        }
    }
    most_distant
}

// ---------------------------------------------------------------------------
// ConnectAllBetterThanCutoff_s
// ---------------------------------------------------------------------------

/// Build connect[i][j] matrices: gene k connects to gene l in species j iff
/// B[i][j][k,l] ≥ mostDistant[k].  Self-connections (k==l when i==j) excluded.
fn connect_all_better_than_cutoff(
    b: &[CsrMatrix],
    most_distant: &[f32],
    i_spec: usize,
) -> Vec<CsrMatrix> {
    b.iter()
        .enumerate()
        .map(|(j_spec, bj)| {
            let mut coo = CooMatrix::new(bj.nrows, bj.ncols);
            for k in 0..bj.nrows {
                let threshold = most_distant[k];
                for (l, v) in bj.row_iter(k) {
                    let l = l as usize;
                    if i_spec == j_spec && k == l {
                        continue; // no self-connections
                    }
                    if v >= threshold {
                        coo.insert_one(k as u32, l as u32);
                    }
                }
            }
            coo.into_csr()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn percentile(vals: &[f32], p: f32) -> f32 {
    if vals.is_empty() {
        return 1.0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (p / 100.0) * (sorted.len() as f32 - 1.0);
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f32;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}
