/// Sparse matrix types used throughout the pipeline.
///
/// CooMatrix — coordinate format, used during construction (keeps max value per cell).
/// CsrMatrix — compressed sparse row, used for all arithmetic and iteration.
///
/// Serialisation format (to/from byte slices, written to disk for inter-pass exchange):
///   magic:   b"OFMAT"  (5 bytes)
///   nrows:   u32 LE
///   ncols:   u32 LE
///   nnz:     u32 LE
///   row_ptr: (nrows+1) * u32 LE
///   col_idx: nnz * u32 LE
///   values:  nnz * f32 LE

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// COO builder
// ---------------------------------------------------------------------------

pub struct CooMatrix {
    pub nrows: usize,
    pub ncols: usize,
    /// Accumulate entries; we keep the maximum value per (row, col) cell.
    entries: HashMap<(u32, u32), f32>,
}

impl CooMatrix {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            entries: HashMap::new(),
        }
    }

    /// Insert a value, keeping the maximum for duplicate (row, col) pairs.
    #[inline]
    pub fn insert_max(&mut self, row: u32, col: u32, val: f32) {
        let e = self.entries.entry((row, col)).or_insert(0.0_f32);
        if val > *e {
            *e = val;
        }
    }

    /// Insert a value, clamping to exactly 1.0 (boolean matrices).
    #[inline]
    pub fn insert_one(&mut self, row: u32, col: u32) {
        self.entries.insert((row, col), 1.0);
    }

    #[allow(dead_code)]
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Convert to CSR, sorting entries row-major then column-major.
    pub fn into_csr(self) -> CsrMatrix {
        let mut sorted: Vec<(u32, u32, f32)> = self
            .entries
            .into_iter()
            .map(|((r, c), v)| (r, c, v))
            .collect();
        sorted.sort_unstable_by_key(|&(r, c, _)| (r, c));

        let nnz = sorted.len();
        let mut row_ptr = vec![0u32; self.nrows + 1];
        let mut col_idx = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for &(r, c, v) in &sorted {
            row_ptr[r as usize + 1] += 1;
            col_idx.push(c);
            values.push(v);
        }
        // prefix-sum
        for i in 1..=self.nrows {
            row_ptr[i] += row_ptr[i - 1];
        }

        CsrMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            row_ptr,
            col_idx,
            values,
        }
    }
}

// ---------------------------------------------------------------------------
// CSR matrix
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    /// Length nrows+1; row_ptr[i]..row_ptr[i+1] gives column slice for row i.
    pub row_ptr: Vec<u32>,
    pub col_idx: Vec<u32>,
    pub values: Vec<f32>,
}

impl CsrMatrix {
    /// Empty matrix (no non-zeros).
    pub fn empty(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_ptr: vec![0u32; nrows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    // -----------------------------------------------------------------------
    // Row access
    // -----------------------------------------------------------------------

    /// Iterate over (col, val) pairs in row `i`.
    #[inline]
    pub fn row_iter(&self, i: usize) -> impl Iterator<Item = (u32, f32)> + '_ {
        let s = self.row_ptr[i] as usize;
        let e = self.row_ptr[i + 1] as usize;
        self.col_idx[s..e]
            .iter()
            .zip(self.values[s..e].iter())
            .map(|(&c, &v)| (c, v))
    }

    /// Maximum value in row `i`, or 0.0 if the row is empty.
    pub fn row_max(&self, i: usize) -> f32 {
        let s = self.row_ptr[i] as usize;
        let e = self.row_ptr[i + 1] as usize;
        self.values[s..e]
            .iter()
            .cloned()
            .fold(0.0_f32, f32::max)
    }

    /// Vector of per-row maxima (length == nrows).
    #[allow(dead_code)]
    pub fn row_maxes(&self) -> Vec<f32> {
        (0..self.nrows).map(|i| self.row_max(i)).collect()
    }

    /// Scalar value at (i, j); 0.0 if absent.  Assumes col_idx within each
    /// row is sorted (guaranteed by into_csr and all constructors here).
    pub fn get(&self, i: usize, j: usize) -> f32 {
        let s = self.row_ptr[i] as usize;
        let e = self.row_ptr[i + 1] as usize;
        match self.col_idx[s..e].binary_search(&(j as u32)) {
            Ok(pos) => self.values[s + pos],
            Err(_) => 0.0,
        }
    }

    // -----------------------------------------------------------------------
    // Transformations
    // -----------------------------------------------------------------------

    /// In-place: multiply each entry (i,j) by `row_scale[i]`.
    pub fn scale_rows_inplace(&mut self, scale: &[f32]) {
        for i in 0..self.nrows {
            let s = self.row_ptr[i] as usize;
            let e = self.row_ptr[i + 1] as usize;
            let si = scale[i];
            for v in &mut self.values[s..e] {
                *v *= si;
            }
        }
    }

    /// In-place: multiply each entry (i,j) by `col_scale[j]`.
    pub fn scale_cols_inplace(&mut self, scale: &[f32]) {
        for idx in 0..self.col_idx.len() {
            self.values[idx] *= scale[self.col_idx[idx] as usize];
        }
    }

    /// Transpose: returns a new CsrMatrix where result[j][i] = self[i][j].
    pub fn transpose(&self) -> CsrMatrix {
        let mut coo = CooMatrix::new(self.ncols, self.nrows);
        for i in 0..self.nrows {
            for (j, v) in self.row_iter(i) {
                coo.insert_max(j, i as u32, v);
            }
        }
        coo.into_csr()
    }

    /// Boolean OR: result[i,j] = 1 if self[i,j]>0 OR other[i,j]>0.
    /// Panics if dimensions differ.
    pub fn bool_or(&self, other: &CsrMatrix) -> CsrMatrix {
        assert_eq!(self.nrows, other.nrows);
        assert_eq!(self.ncols, other.ncols);
        let mut coo = CooMatrix::new(self.nrows, self.ncols);
        for i in 0..self.nrows {
            for (j, _) in self.row_iter(i) {
                coo.insert_one(i as u32, j);
            }
            for (j, _) in other.row_iter(i) {
                coo.insert_one(i as u32, j);
            }
        }
        coo.into_csr()
    }

    /// Element-wise AND with the transpose of `other`:
    /// result[i,j] = 1 if self[i,j]>0 AND other[j,i]>0.
    /// Returns a boolean (0/1) CsrMatrix.
    pub fn and_with_transpose(&self, other: &CsrMatrix) -> CsrMatrix {
        // Build a fast lookup: set of (j, i) entries present in `other`.
        let mut other_set: HashMap<(u32, u32), ()> =
            HashMap::with_capacity(other.nnz());
        for i in 0..other.nrows {
            for (j, _) in other.row_iter(i) {
                other_set.insert((j, i as u32), ());
            }
        }

        let mut coo = CooMatrix::new(self.nrows, self.ncols);
        for i in 0..self.nrows {
            for (j, _) in self.row_iter(i) {
                if other_set.contains_key(&(j, i as u32)) {
                    coo.insert_one(i as u32, j);
                }
            }
        }
        coo.into_csr()
    }

    /// All (row, col) non-zero index pairs.
    pub fn nonzero_pairs(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(self.nnz());
        for i in 0..self.nrows {
            for (j, _) in self.row_iter(i) {
                out.push((i, j as usize));
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Serialisation
    // -----------------------------------------------------------------------

    pub fn to_bytes(&self) -> Vec<u8> {
        let nnz = self.col_idx.len();
        let capacity = 5 + 4 * 3 + (self.nrows + 1) * 4 + nnz * 4 + nnz * 4;
        let mut buf = Vec::with_capacity(capacity);

        buf.extend_from_slice(b"OFMAT");
        buf.extend_from_slice(&(self.nrows as u32).to_le_bytes());
        buf.extend_from_slice(&(self.ncols as u32).to_le_bytes());
        buf.extend_from_slice(&(nnz as u32).to_le_bytes());

        for &v in &self.row_ptr {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.col_idx {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 17 || &data[..5] != b"OFMAT" {
            return Err("Invalid matrix file: bad magic".into());
        }
        let nrows = u32::from_le_bytes(data[5..9].try_into().unwrap()) as usize;
        let ncols = u32::from_le_bytes(data[9..13].try_into().unwrap()) as usize;
        let nnz = u32::from_le_bytes(data[13..17].try_into().unwrap()) as usize;

        let rp_start = 17;
        let rp_end = rp_start + (nrows + 1) * 4;
        let ci_start = rp_end;
        let ci_end = ci_start + nnz * 4;
        let vl_start = ci_end;
        let vl_end = vl_start + nnz * 4;

        if data.len() < vl_end {
            return Err(format!(
                "Matrix file truncated: need {} bytes, have {}",
                vl_end,
                data.len()
            ));
        }

        let row_ptr = data[rp_start..rp_end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let col_idx = data[ci_start..ci_end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let values = data[vl_start..vl_end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        Ok(Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        })
    }
}
