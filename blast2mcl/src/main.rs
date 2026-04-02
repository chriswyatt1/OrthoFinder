/// blast2mcl — fast BLAST result processing for OrthoFinder
///
/// Replaces the Python `DoOrthogroups` waterfall/matrix phase:
///   BLAST TSV files  →  MCL native-matrix graph file
///
/// # Three-pass algorithm
///
/// **Pass 1** (parallel, one job per query species `i`):
///   For each target species `j`:
///     - Parse Blast{i}_{j}.txt → raw score matrix B[i][j].
///     - Normalise scores by sequence lengths.
///   Compute best-hit matrices BH[i][*].
///   Write B[i][*] and BH[i][*] to `work_dir` as `.mat` binary files.
///
/// **Pass 2** (parallel after Pass 1):
///   Load BH[i][*] and BH[*][i] from disk.
///   Compute RBH, most-distant thresholds, connect[i][*] matrices.
///   Write connect[i][*] to disk.
///
/// **Pass 3** (parallel after Pass 2):
///   Load connect[i][*], connect[*][i], and B[i][*].
///   Write one segment file per species with MCL graph rows.
///
/// Segments are concatenated into the final output file.

mod blast_parser;
mod graph;
mod normalise;
mod sparse;
mod waterfall;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use rayon::prelude::*;

use crate::normalise::{normalise_scores, normalised_bit_score, read_sequence_lengths};
use crate::sparse::CsrMatrix;
use crate::waterfall::{connect_cognates, get_best_hits};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "blast2mcl",
    about = "Converts OrthoFinder BLAST results to an MCL input graph.\n\
             Replaces the Python waterfall/matrix processing phase."
)]
struct Args {
    /// Directory (or directories, comma-separated) containing Blast{i}_{j}.txt[.gz] files.
    #[arg(long, value_delimiter = ',')]
    blast_dir: Vec<PathBuf>,

    /// Directory (or directories, comma-separated) containing Species{i}.fa FASTA files.
    /// Searched in order; first match for each species wins.
    #[arg(long, value_delimiter = ',')]
    fasta_dir: Vec<PathBuf>,

    /// Comma-separated list of OrthoFinder species IDs to process (e.g. 0,1,2).
    #[arg(long, value_delimiter = ',')]
    species_to_use: Vec<usize>,

    /// Number of sequences per species, comma-separated in the same order as
    /// `--species-to-use`.
    #[arg(long, value_delimiter = ',')]
    n_seqs_per_species: Vec<usize>,

    /// Path for the output MCL graph file.
    #[arg(long, short = 'o')]
    output: PathBuf,

    /// Number of parallel threads (default: all logical CPUs).
    #[arg(long, short = 't')]
    threads: Option<usize>,

    /// BLAST searches were run in both directions (default: true).
    #[arg(long, default_value_t = true)]
    double_blast: bool,

    /// Use v2 score normalisation (simple sqrt scaling instead of log-linear fit).
    #[arg(long)]
    v2_scores: bool,

    /// Use v2 distance estimator (relative-RBH method) in ConnectCognates.
    /// Automatically enabled when --v2-scores is set.
    #[arg(long)]
    v2_distance: bool,

    /// Separator character used in BLAST sequence IDs (default: '_').
    #[arg(long, default_value = "_")]
    sep: char,

    /// Directory for intermediate .mat files (default: <output>.work/).
    #[arg(long)]
    work_dir: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Path for intermediate B[i][j] or BH[i][j] or connect[i][j] matrix files.
fn mat_path(work_dir: &Path, prefix: &str, i: usize, j: usize) -> PathBuf {
    work_dir.join(format!("{prefix}_{i}_{j}.mat"))
}

fn save_matrix(path: &Path, m: &CsrMatrix) {
    fs::write(path, m.to_bytes())
        .unwrap_or_else(|e| panic!("Cannot write {:?}: {}", path, e));
}

fn load_matrix(path: &Path) -> CsrMatrix {
    let data = fs::read(path)
        .unwrap_or_else(|e| panic!("Cannot read {:?}: {}", path, e));
    CsrMatrix::from_bytes(&data)
        .unwrap_or_else(|e| panic!("Cannot deserialise {:?}: {}", path, e))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    // ---- validate ----
    let n_species = args.species_to_use.len();
    assert_eq!(
        args.n_seqs_per_species.len(),
        n_species,
        "--n-seqs-per-species must have the same number of entries as --species-to-use"
    );
    assert!(!args.blast_dir.is_empty(), "--blast-dir must not be empty");

    // ---- thread pool ----
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .expect("Failed to build rayon thread pool");
    }

    let v2_distance = args.v2_scores || args.v2_distance;

    // ---- work directory ----
    let work_dir: PathBuf = args.work_dir.clone().unwrap_or_else(|| {
        let mut p = args.output.clone();
        p.set_extension("work");
        p
    });
    fs::create_dir_all(&work_dir)
        .unwrap_or_else(|e| panic!("Cannot create work dir {:?}: {}", work_dir, e));

    // ---- sequence lengths from FASTA ----
    // species_lengths[i_spec] = Vec of lengths for species species_to_use[i_spec].
    eprintln!("Reading sequence lengths from FASTA files…");
    let species_lengths: Vec<Vec<f32>> = args
        .species_to_use
        .iter()
        .map(|&sp| {
            let fa = args
                .fasta_dir
                .iter()
                .map(|d| d.join(format!("Species{}.fa", sp)))
                .find(|p| p.exists())
                .unwrap_or_else(|| {
                    panic!(
                        "Species{}.fa not found in any of {:?}",
                        sp, args.fasta_dir
                    )
                });
            read_sequence_lengths(&fa)
        })
        .collect();

    // Validate length counts match n_seqs_per_species.
    for (idx, (expected, lengths)) in args
        .n_seqs_per_species
        .iter()
        .zip(species_lengths.iter())
        .enumerate()
    {
        assert_eq!(
            lengths.len(),
            *expected,
            "Species {} FASTA has {} sequences but --n-seqs-per-species says {}",
            args.species_to_use[idx],
            lengths.len(),
            expected
        );
    }

    // Cumulative global sequence offsets.
    let seq_offsets: Vec<usize> = {
        let mut offsets = vec![0usize; n_species];
        for i in 1..n_species {
            offsets[i] = offsets[i - 1] + args.n_seqs_per_species[i - 1];
        }
        offsets
    };
    let total_seqs: usize = seq_offsets[n_species - 1] + args.n_seqs_per_species[n_species - 1];

    // =========================================================================
    // Pass 1 — ProcessBlastHits: parse, normalise, compute BH, save to disk.
    // =========================================================================
    eprintln!("Pass 1: parsing BLAST files and computing best hits…");

    (0..n_species).into_par_iter().for_each(|i_spec| {
        let i_sp = args.species_to_use[i_spec];
        let n_seqs_i = args.n_seqs_per_species[i_spec];
        let len_i = &species_lengths[i_spec];

        // Parse and normalise B[i][j] for every j.
        let b_i: Vec<CsrMatrix> = (0..n_species)
            .map(|j_spec| {
                let j_sp = args.species_to_use[j_spec];
                let n_seqs_j = args.n_seqs_per_species[j_spec];
                let len_j = &species_lengths[j_spec];

                let raw = blast_parser::get_blast6_scores(
                    &args.blast_dir,
                    i_sp,
                    j_sp,
                    n_seqs_i,
                    n_seqs_j,
                    /*exclude_self_hits=*/ true,
                    args.sep,
                    args.double_blast,
                    /*allow_empty=*/ false,
                );

                if args.v2_scores {
                    normalised_bit_score(raw, len_i, len_j)
                } else {
                    normalise_scores(raw, len_i, len_j)
                }
            })
            .collect();

        // Compute BH[i][j] for all j.
        let bh_i = get_best_hits(&b_i, i_spec, 1e-3);

        // Persist B[i][j] and BH[i][j].
        for j_spec in 0..n_species {
            save_matrix(&mat_path(&work_dir, "B", i_spec, j_spec), &b_i[j_spec]);
            save_matrix(
                &mat_path(&work_dir, "BH", i_spec, j_spec),
                &bh_i[j_spec],
            );
        }

        eprintln!("  Pass 1: species {} done.", i_sp);
    });

    // =========================================================================
    // Pass 2 — ConnectCognates: load BH, compute RBH + connect, save to disk.
    // =========================================================================
    eprintln!("Pass 2: computing reciprocal best hits and connections…");

    (0..n_species).into_par_iter().for_each(|i_spec| {
        let i_sp = args.species_to_use[i_spec];

        // Load BH[i][j] for all j.
        let bh_from_i: Vec<CsrMatrix> = (0..n_species)
            .map(|j_spec| load_matrix(&mat_path(&work_dir, "BH", i_spec, j_spec)))
            .collect();

        // Load BH[j][i] for all j (written during Pass 1 for species j).
        let bh_to_i: Vec<CsrMatrix> = (0..n_species)
            .map(|j_spec| load_matrix(&mat_path(&work_dir, "BH", j_spec, i_spec)))
            .collect();

        // Load B[i][j] for all j.
        let b_i: Vec<CsrMatrix> = (0..n_species)
            .map(|j_spec| load_matrix(&mat_path(&work_dir, "B", i_spec, j_spec)))
            .collect();

        let connect_i = connect_cognates(&bh_from_i, &bh_to_i, &b_i, i_spec, v2_distance);

        for j_spec in 0..n_species {
            save_matrix(
                &mat_path(&work_dir, "connect", i_spec, j_spec),
                &connect_i[j_spec],
            );
        }

        eprintln!("  Pass 2: species {} done.", i_sp);
    });

    // =========================================================================
    // Pass 3 — WriteGraph: build segment files, then concatenate.
    // =========================================================================
    eprintln!("Pass 3: writing MCL graph…");

    // Write header to the output file.
    {
        let mut out = BufWriter::new(
            File::create(&args.output)
                .unwrap_or_else(|e| panic!("Cannot create {:?}: {}", args.output, e)),
        );
        writeln!(
            out,
            "(mclheader\nmcltype matrix\ndimensions {n}x{n}\n)\n\n(mclmatrix\nbegin\n",
            n = total_seqs
        )
        .unwrap();
    }

    // Write one segment per species (in parallel).
    let segment_paths: Vec<PathBuf> = (0..n_species)
        .map(|i_spec| work_dir.join(format!("graph_seg_{i_spec}.txt")))
        .collect();

    (0..n_species).into_par_iter().for_each(|i_spec| {
        let i_sp = args.species_to_use[i_spec];
        let n_seqs_i = args.n_seqs_per_species[i_spec];

        // Load connect[i][j] for all j.
        let connect_from_i: Vec<CsrMatrix> = (0..n_species)
            .map(|j_spec| load_matrix(&mat_path(&work_dir, "connect", i_spec, j_spec)))
            .collect();

        // Load connect[j][i] for all j (transpose source).
        let connect_to_i: Vec<CsrMatrix> = (0..n_species)
            .map(|j_spec| load_matrix(&mat_path(&work_dir, "connect", j_spec, i_spec)))
            .collect();

        // Load B[i][j] for all j.
        let b_i: Vec<CsrMatrix> = (0..n_species)
            .map(|j_spec| load_matrix(&mat_path(&work_dir, "B", i_spec, j_spec)))
            .collect();

        let seg_path = &segment_paths[i_spec];
        let mut writer = BufWriter::new(
            File::create(seg_path)
                .unwrap_or_else(|e| panic!("Cannot create segment {:?}: {}", seg_path, e)),
        );

        graph::write_graph_rows(
            &mut writer,
            &connect_from_i,
            &connect_to_i,
            &b_i,
            &seq_offsets,
            i_spec,
            n_seqs_i,
            /*is_last_species=*/ i_spec == n_species - 1,
        )
        .unwrap_or_else(|e| panic!("Error writing graph segment {}: {}", i_spec, e));

        eprintln!("  Pass 3: species {} done.", i_sp);
    });

    // Concatenate segments in order into the output file.
    {
        let mut out = fs::OpenOptions::new()
            .append(true)
            .open(&args.output)
            .unwrap_or_else(|e| panic!("Cannot open {:?} for append: {}", args.output, e));

        for seg_path in &segment_paths {
            let seg_data = fs::read(seg_path)
                .unwrap_or_else(|e| panic!("Cannot read segment {:?}: {}", seg_path, e));
            out.write_all(&seg_data)
                .unwrap_or_else(|e| panic!("Cannot write to output: {}", e));
            fs::remove_file(seg_path).ok();
        }
    }

    // Clean up work directory (leave it if user specified one explicitly).
    if args.work_dir.is_none() {
        fs::remove_dir_all(&work_dir).ok();
    }

    eprintln!("Done. MCL graph written to {:?}.", args.output);
}
