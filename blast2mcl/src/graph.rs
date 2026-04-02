/// Write one species' rows to an MCL native-matrix graph file segment.
///
/// The MCL format written by OrthoFinder looks like:
///
/// ```text
/// (mclheader
/// mcltype matrix
/// dimensions NxN
/// )
///
/// (mclmatrix
/// begin
///
/// 0    1:0.420 5:0.310 $
/// 1    0:0.420 $
/// ...
/// N-1    ... $
/// )
/// ```
///
/// The header and `begin` line are written once by the caller.
/// This module writes the body rows for a single query species and appends `)`
/// only when `is_last_species` is true, matching Python's `WriteGraph_perSpecies`.
///
/// # Connection logic  (matches `WriteGraph_perSpecies` + `MatricesAnd_s`)
///
/// For each target species j we compute a symmetrised connection mask:
///   connect2[j] = connect[i][j]  OR  connect[j][i]ᵀ
///
/// Then the score written is B[i][j][k, l] wherever connect2[j][k, l] is set.
/// Entries with score == 0 after masking are skipped (they're structural zeros).

use std::io::{self, Write};

use crate::sparse::CsrMatrix;

/// Write graph rows for species `i_spec` into `writer`.
///
/// Parameters:
///   `connect_from_i`  — connect[i][j] for all j.
///   `connect_to_i`    — connect[j][i] for all j (used to symmetrise).
///   `b`               — normalised B[i][j] for all j.
///   `seq_offsets`     — global sequence index offset for each species.
///   `i_spec`          — query species index (for offset lookup).
///   `n_seqs_i`        — number of sequences in species i.
///   `is_last_species` — if true, write the closing `)` line.
pub fn write_graph_rows<W: Write>(
    writer: &mut W,
    connect_from_i: &[CsrMatrix],
    connect_to_i: &[CsrMatrix],
    b: &[CsrMatrix],
    seq_offsets: &[usize],
    i_spec: usize,
    n_seqs_i: usize,
    is_last_species: bool,
) -> io::Result<()> {
    let n_species = b.len();

    // Pre-compute symmetrised masks: connect2[j][k,l] = connect[i][j][k,l]
    // OR connect[j][i][l,k]  (i.e. transpose of connect_to_i[j]).
    let connect2: Vec<CsrMatrix> = (0..n_species)
        .map(|j| connect_from_i[j].bool_or(&connect_to_i[j].transpose()))
        .collect();

    let i_offset = seq_offsets[i_spec];

    for k in 0..n_seqs_i {
        let global_k = i_offset + k;
        write!(writer, "{k_g}    ", k_g = global_k)?;

        for j in 0..n_species {
            let j_offset = seq_offsets[j];
            // Iterate over (col, _) in connect2[j] row k; look up score in B[i][j].
            for (l, _) in connect2[j].row_iter(k) {
                let l = l as usize;
                let score = b[j].get(k, l);
                if score > 0.0 {
                    write!(writer, "{l_g}:{score:.3} ", l_g = j_offset + l)?;
                }
            }
        }

        writeln!(writer, "$")?;
    }

    if is_last_species {
        writeln!(writer, ")")?;
    }

    Ok(())
}
