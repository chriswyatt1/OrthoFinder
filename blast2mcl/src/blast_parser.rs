/// Parse OrthoFinder BLAST-6 TSV result files into sparse score matrices.
///
/// File naming convention: `Blast{i_species}_{j_species}.txt[.gz]`
///
/// Each line has 12+ tab-separated fields; field 0 = query ID, field 1 = hit ID,
/// field 11 = bit score.  IDs are formatted as `{species}_{seq_index}`, e.g. `2_45`.
///
/// When `double_blast` is false and i > j, we read `Blast{j}_{i}` and swap columns
/// (qRev mode), matching the Python `GetBLAST6Scores` logic exactly.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;

use crate::sparse::{CooMatrix, CsrMatrix};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Read `Blast{i_sp}_{j_sp}.txt[.gz]` from `blast_dirs` (first directory that
/// contains the file wins) and return the (nSeqs_i × nSeqs_j) score matrix.
///
/// `n_seqs_i` / `n_seqs_j` are the dimensions used for the matrix; they must
/// match the species counts encoded in the FASTA files.
///
/// When `exclude_self_hits` is true (only relevant when i == j) diagonal entries
/// are skipped.
///
/// Returns an empty matrix when `allow_empty` is true and no file is found;
/// otherwise panics.
pub fn get_blast6_scores(
    blast_dirs: &[PathBuf],
    i_species: usize,
    j_species: usize,
    n_seqs_i: usize,
    n_seqs_j: usize,
    exclude_self_hits: bool,
    sep: char,
    double_blast: bool,
    allow_empty: bool,
) -> CsrMatrix {
    let same_species = i_species == j_species;
    let check_self = exclude_self_hits && same_species;

    // When not running double-blast, only one direction was searched.
    let q_rev = !double_blast && i_species > j_species;

    // Column indices: which column is query, which is hit.
    let (i_q, i_h) = if q_rev { (1usize, 0usize) } else { (0usize, 1usize) };
    // File is always Blast{lower}_{higher} when qRev.
    let (file_i, file_j) = if q_rev {
        (j_species, i_species)
    } else {
        (i_species, j_species)
    };

    let path = find_blast_file(blast_dirs, file_i, file_j);

    if path.is_none() {
        if allow_empty {
            return CsrMatrix::empty(n_seqs_i, n_seqs_j);
        }
        panic!(
            "BLAST result file not found: Blast{}_{}.txt[.gz] in any of {:?}",
            file_i, file_j, blast_dirs
        );
    }
    let path = path.unwrap();

    let mut matrix = CooMatrix::new(n_seqs_i, n_seqs_j);

    let reader = open_possibly_gzipped(&path)
        .unwrap_or_else(|e| panic!("Cannot open {:?}: {}", path, e));

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.unwrap_or_else(|e| {
            panic!("IO error reading {:?} at line {}: {}", path, line_no + 1, e)
        });
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.splitn(13, '\t').collect();
        if fields.len() < 12 {
            eprintln!(
                "WARNING: {:?} line {} has fewer than 12 fields, skipping",
                path,
                line_no + 1
            );
            continue;
        }

        let seq1_id = parse_seq_id(fields[i_q], sep, &path, line_no + 1);
        let seq2_id = parse_seq_id(fields[i_h], sep, &path, line_no + 1);
        let score: f32 = fields[11].parse().unwrap_or_else(|_| {
            panic!(
                "ERROR: {:?} line {}: bit-score field is not a number: {:?}",
                path,
                line_no + 1,
                fields[11]
            )
        });

        if check_self && seq1_id == seq2_id {
            continue;
        }

        if seq1_id >= n_seqs_i as u32 || seq2_id >= n_seqs_j as u32 {
            eprintln!(
                "ERROR: Blast{}_{}.txt: sequence index out of bounds at line {}. \
                 Expected < {} rows and < {} cols, got ({}, {}).",
                i_species,
                j_species,
                line_no + 1,
                n_seqs_i,
                n_seqs_j,
                seq1_id,
                seq2_id
            );
            std::process::exit(1);
        }

        matrix.insert_max(seq1_id, seq2_id, score);
    }

    matrix.into_csr()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the integer sequence index from an ID like `"2_45"`.
/// Splits on `sep` with a limit of 3 parts and takes the second field,
/// matching Python's `row.split(sep, 2)[1]`.
fn parse_seq_id(field: &str, sep: char, path: &Path, line_no: usize) -> u32 {
    // Split at most twice so we handle IDs that contain the separator character.
    let mut parts = field.splitn(3, sep);
    let _ = parts.next(); // species part
    let idx_str = parts.next().unwrap_or_else(|| {
        panic!(
            "ERROR: {:?} line {}: sequence ID {:?} has no separator {:?}",
            path, line_no, field, sep
        )
    });
    idx_str.parse::<u32>().unwrap_or_else(|_| {
        panic!(
            "ERROR: {:?} line {}: sequence index {:?} is not an integer",
            path, line_no, idx_str
        )
    })
}

/// Search `blast_dirs` in order; return the first path that exists.
fn find_blast_file(blast_dirs: &[PathBuf], i: usize, j: usize) -> Option<PathBuf> {
    let stem = format!("Blast{}_{}.txt", i, j);
    let gz = format!("Blast{}_{}.txt.gz", i, j);
    for dir in blast_dirs {
        let p = dir.join(&stem);
        if p.exists() {
            return Some(p);
        }
        let p = dir.join(&gz);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Open a file, transparently decompressing gzip if the path ends with `.gz`.
fn open_possibly_gzipped(path: &Path) -> io::Result<Box<dyn BufRead>> {
    let file = File::open(path)?;
    if path.extension().and_then(|e| e.to_str()) == Some("gz") {
        Ok(Box::new(BufReader::new(GzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}
