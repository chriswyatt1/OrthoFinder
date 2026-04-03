#!/usr/bin/env python3
"""
test_waterfall.py — Compare blast2mcl (Rust) vs Python waterfall output.

Runs OrthoFinder on the built-in test data up to the orthogroup stage
(which exercises the Rust waterfall), then re-runs the Python waterfall
on the same BLAST results and compares the two MCL graph files.

Usage:
    # Use the default built-in test data (tests/data/proteome/Core/):
    python tools/test_waterfall.py

    # Use a specific FASTA directory:
    python tools/test_waterfall.py --fasta-dir /path/to/fastas

    # Keep outputs for manual inspection:
    python tools/test_waterfall.py --keep-outputs

    # Adjust score tolerance (default 0.001):
    python tools/test_waterfall.py --tol 0.005

Exit code: 0 if outputs match, 1 if they differ.
"""

import argparse
import glob
import os
import re
import shutil
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# Locate repo root regardless of where the script is called from
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
SRC_DIR    = os.path.join(REPO_ROOT, "src")

DEFAULT_FASTA_DIR = os.path.join(REPO_ROOT, "tests", "data", "proteome", "Core")

# Make sure the OrthoFinder package is importable
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# Step 1 — Run OrthoFinder (Rust waterfall) and find the WorkingDirectory
# ---------------------------------------------------------------------------

def run_orthofinder(fasta_dir, run_name, threads):
    """Call OrthoFinder's main() in-process up to orthogroups (-og) and return
    the WorkingDirectory path that was created.

    Running in-process avoids the subprocess Python-interpreter mismatch that
    would occur if the script is invoked with a different Python than the one
    that has OrthoFinder's dependencies (numpy, scipy, etc.).
    """
    # Import the full package first so circular-import guards are satisfied.
    try:
        import orthofinder  # noqa: F401 — side-effects initialise the package
        from orthofinder.run.__main__ import main
        from orthofinder.utils import files
    except ImportError as e:
        sys.exit(
            f"ERROR: Cannot import OrthoFinder ({e}).\n"
            f"Run this script with the Python that has OrthoFinder's dependencies:\n"
            f"  /opt/anaconda3/bin/python tools/test_waterfall.py\n"
            f"or activate your OrthoFinder conda environment first."
        )

    # Reset FileHandler global state before the run (mirrors conftest.py).
    for attr in ("wd_base_prev", "home_for_results", "wd_base", "wd_current",
                 "wd_trees", "wd1", "wd2", "rd1", "base_dir"):
        if hasattr(files.FileHandler, attr):
            setattr(files.FileHandler, attr, "")

    args = ["-f", fasta_dir, "-og", "-n", run_name, "-t", str(threads),
            "--no-print-info"]
    print(f"[Step 1] Running OrthoFinder in-process (Rust waterfall):\n"
          f"  orthofinder {' '.join(args)}")

    import multiprocessing as mp
    import platform
    if platform.system() == "Darwin":
        mp.set_start_method("fork", force=True)

    try:
        main(args)
    except SystemExit as e:
        if e.code not in (0, None):
            sys.exit(f"ERROR: OrthoFinder exited with code {e.code}")

    return _find_working_dir(fasta_dir, run_name)


def _find_working_dir(fasta_dir, run_name):
    """Locate WorkingDirectory inside the OrthoFinder results for run_name.
    If multiple matches exist (e.g. run_name, run_name_1, run_name_2),
    return the most recently modified one."""
    of_dir = os.path.join(fasta_dir, "OrthoFinder")
    if not os.path.isdir(of_dir):
        sys.exit(f"ERROR: OrthoFinder results dir not found at {of_dir!r}")

    matches = glob.glob(os.path.join(of_dir, f"Results_{run_name}", "WorkingDirectory"))
    matches += glob.glob(os.path.join(of_dir, f"Results_{run_name}_*", "WorkingDirectory"))
    if not matches:
        sys.exit(
            f"ERROR: Cannot find WorkingDirectory for run '{run_name}' in {of_dir!r}"
        )
    # Pick the most recently modified WorkingDirectory.
    return max(matches, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Step 2 — Parse seqsInfo from the WorkingDirectory ID files
# ---------------------------------------------------------------------------

def read_species_info(working_dir):
    """Return (species_to_use, n_seqs_per_species) from SpeciesIDs + SequenceIDs."""
    from collections import defaultdict

    species_ids_fn = os.path.join(working_dir, "SpeciesIDs.txt")
    seq_ids_fn     = os.path.join(working_dir, "SequenceIDs.txt")

    for fn in (species_ids_fn, seq_ids_fn):
        if not os.path.exists(fn):
            sys.exit(f"ERROR: {fn} not found — is this an OrthoFinder WorkingDirectory?")

    species_to_use = []
    with open(species_ids_fn) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sp_id = int(line.split(":")[0].strip())
            species_to_use.append(sp_id)
    species_to_use.sort()

    n_seqs_per_species = defaultdict(int)
    with open(seq_ids_fn) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key = line.split(":")[0].strip()
            sp_id = int(key.split("_")[0])
            n_seqs_per_species[sp_id] += 1

    return species_to_use, dict(n_seqs_per_species)


# ---------------------------------------------------------------------------
# Step 3 — Find the Rust graph file written by OrthoFinder
# ---------------------------------------------------------------------------

def find_rust_graph(working_dir):
    matches = glob.glob(os.path.join(working_dir, "*_graph.txt"))
    if not matches:
        sys.exit(
            f"ERROR: No *_graph.txt found in {working_dir!r}\n"
            "OrthoFinder may not have reached the orthogroup stage."
        )
    if len(matches) > 1:
        print(f"WARNING: Multiple graph files found, using {matches[0]!r}")
    return matches[0]


# ---------------------------------------------------------------------------
# Step 4 — Run the Python waterfall on the same BLAST files
# ---------------------------------------------------------------------------

def run_python_waterfall(working_dir, species_to_use, n_seqs_per_species,
                         output_path, threads, double_blast, v2_scores):
    """Call _run_python_waterfall_lt3 from gathering.py and copy the graph."""
    import orthofinder  # noqa: F401 — initialise package before sub-imports

    from orthofinder.utils import files, parallel_task_manager
    from orthofinder.utils.util import SequencesInfo
    from orthofinder.orthogroups.gathering import (
        GetSequenceLengths,
        _run_python_waterfall_lt3,
    )

    # Use OrthoFinder's own SequencesInfo namedtuple (module-level, picklable).
    seq_starts = []
    offset = 0
    for sp in species_to_use:
        seq_starts.append(offset)
        offset += n_seqs_per_species[sp]

    seqsInfo = SequencesInfo(
        speciesToUse       = species_to_use,
        nSeqsPerSpecies    = n_seqs_per_species,
        nSpecies           = len(species_to_use),
        nSeqs              = offset,
        seqStartingIndices = seq_starts,
    )

    class Options:
        pass
    opts = Options()
    opts.nProcessAlg = threads
    opts.qDoubleBlast = double_blast
    opts.v2_scores    = v2_scores
    opts.old_version  = False

    # Point FileHandler at the WorkingDirectory.
    wd = working_dir.rstrip(os.sep) + os.sep
    files.FileHandler.wd_base    = [wd]
    files.FileHandler.wd_current = wd

    blastDir_list = [wd]
    Lengths = GetSequenceLengths(seqsInfo)

    print("[Step 4] Running Python waterfall (_run_python_waterfall_lt3) …")
    graph = _run_python_waterfall_lt3(
        seqsInfo, blastDir_list, Lengths, opts,
        i_unassigned=None, GRACE_PERIOD=10., STALL_TIMEOUT=200.,
    )
    shutil.copy(graph, output_path)
    print(f"[Step 4] Python graph written to {output_path}")


# ---------------------------------------------------------------------------
# Step 5 — Parse MCL graph files and compare
# ---------------------------------------------------------------------------

def parse_graph(path):
    """Return dict (query_id, hit_id) → score for all edges in an MCL graph."""
    edges = {}
    in_matrix = False
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line == "begin":
                in_matrix = True
                continue
            if not in_matrix:
                continue
            if line in (")", ""):
                continue
            # Format: "query    hit1:score1 hit2:score2 ... $"
            parts = line.rstrip("$ ").split()
            if not parts:
                continue
            try:
                query_id = int(parts[0])
            except ValueError:
                continue
            for token in parts[1:]:
                if ":" not in token:
                    continue
                hit_str, score_str = token.rsplit(":", 1)
                try:
                    edges[(query_id, int(hit_str))] = float(score_str)
                except ValueError:
                    pass
    return edges


def compare_graphs(rust_path, python_path, tol):
    print(f"\n[Step 5] Comparing:\n  Rust  : {rust_path}\n  Python: {python_path}")
    rust_edges   = parse_graph(rust_path)
    python_edges = parse_graph(python_path)

    rust_keys   = set(rust_edges)
    python_keys = set(python_edges)
    only_rust   = rust_keys - python_keys
    only_python = python_keys - rust_keys
    common      = rust_keys & python_keys

    score_diffs = [
        (k, rust_edges[k], python_edges[k], abs(rust_edges[k] - python_edges[k]))
        for k in common
        if abs(rust_edges[k] - python_edges[k]) > tol
    ]

    print(f"\n{'='*60}")
    print(f"  Rust   edges : {len(rust_edges):>8,}")
    print(f"  Python edges : {len(python_edges):>8,}")
    print(f"  Common edges : {len(common):>8,}")
    print(f"  Only in Rust : {len(only_rust):>8,}")
    print(f"  Only in Python: {len(only_python):>7,}")
    print(f"  Score diffs > {tol}: {len(score_diffs):>4,}")

    if only_rust:
        print(f"\n  Sample edges only in Rust (first 5):")
        for k in list(only_rust)[:5]:
            print(f"    {k}  score={rust_edges[k]:.4f}")
    if only_python:
        print(f"\n  Sample edges only in Python (first 5):")
        for k in list(only_python)[:5]:
            print(f"    {k}  score={python_edges[k]:.4f}")
    if score_diffs:
        print(f"\n  Sample score differences (key, rust, python, |diff|):")
        for row in score_diffs[:10]:
            print(f"    {row}")

    passed = not only_rust and not only_python and not score_diffs
    print(f"\n  {'PASS ✓' if passed else 'FAIL ✗'}")
    print('='*60)
    return passed


# ---------------------------------------------------------------------------
# Step 3.5 — Time blast2mcl directly on existing BLAST files
# ---------------------------------------------------------------------------

def time_rust_waterfall(working_dir, species_to_use, n_seqs_per_species,
                         tmpdir, threads, double_blast, v2_scores):
    """Re-run blast2mcl on the BLAST files already in working_dir and return elapsed seconds."""
    import subprocess
    import orthofinder  # noqa: F401
    from orthofinder.utils import parallel_task_manager

    blast2mcl_bin = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "orthofinder", "bin", "blast2mcl",
    )
    if not os.path.isfile(blast2mcl_bin):
        print(f"WARNING: blast2mcl binary not found at {blast2mcl_bin!r}, skipping Rust timing")
        return 0.0

    wd = working_dir.rstrip(os.sep) + os.sep
    output_path = os.path.join(tmpdir, "rust_timed_graph.txt")
    cmd = [
        blast2mcl_bin,
        "--blast-dir", wd,
        "--fasta-dir", wd,
        "--species-to-use", ",".join(str(s) for s in species_to_use),
        "--n-seqs-per-species", ",".join(str(n_seqs_per_species[s]) for s in species_to_use),
        "--output", output_path,
        "--threads", str(threads),
    ]
    if v2_scores:
        cmd.append("--v2-scores")
    if not double_blast:
        cmd.extend(["--double-blast", "false"])

    print(f"[Step 3.5] Timing blast2mcl directly…")
    t = time.perf_counter()
    result = subprocess.run(cmd, env=parallel_task_manager.my_env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - t
    if result.returncode != 0:
        print("WARNING: blast2mcl returned non-zero exit code during timing run")
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fasta-dir",
        default=DEFAULT_FASTA_DIR,
        help=f"Directory of input FASTA files (default: {DEFAULT_FASTA_DIR})",
    )
    parser.add_argument("--threads",      type=int,   default=1)
    parser.add_argument("--tol",          type=float, default=1e-3,
                        help="Float tolerance for score comparison (default 0.001)")
    parser.add_argument("--double-blast", action="store_true", default=True)
    parser.add_argument("--v2-scores",    action="store_true", default=False)
    parser.add_argument("--keep-outputs", action="store_true",
                        help="Keep temporary graph files after the test")
    args = parser.parse_args()

    fasta_dir = os.path.abspath(args.fasta_dir)
    if not os.path.isdir(fasta_dir):
        sys.exit(f"ERROR: {fasta_dir!r} is not a directory")

    tmpdir   = tempfile.mkdtemp(prefix="of_waterfall_test_")
    run_name = "waterfall_comparison"

    try:
        # 1. OrthoFinder run (Rust waterfall)
        t0 = time.perf_counter()
        working_dir = run_orthofinder(fasta_dir, run_name, args.threads)
        rust_total = time.perf_counter() - t0
        print(f"[Step 1] WorkingDirectory: {working_dir}")

        # 2. Species info
        species_to_use, n_seqs_per_species = read_species_info(working_dir)
        print(f"[Step 2] Species: {species_to_use}")

        # 3. Rust graph (already written by OrthoFinder)
        rust_graph = find_rust_graph(working_dir)
        print(f"[Step 3] Rust graph: {rust_graph}")

        # 3.5. Re-run blast2mcl alone on existing BLAST files for a clean timing
        rust_waterfall_time = time_rust_waterfall(
            working_dir, species_to_use, n_seqs_per_species,
            tmpdir, args.threads, args.double_blast, args.v2_scores,
        )

        # 4. Python waterfall
        python_graph = os.path.join(tmpdir, "python_graph.txt")
        t1 = time.perf_counter()
        run_python_waterfall(
            working_dir, species_to_use, n_seqs_per_species,
            python_graph, args.threads, args.double_blast, args.v2_scores,
        )
        python_waterfall_time = time.perf_counter() - t1

        # Timing summary
        print(f"\n[Timing] Full OrthoFinder run (incl. diamond+MCL): {rust_total:.2f}s")
        print(f"[Timing] blast2mcl (Rust waterfall) only:           {rust_waterfall_time:.2f}s")
        print(f"[Timing] Python waterfall only:                     {python_waterfall_time:.2f}s")
        if rust_waterfall_time > 0:
            print(f"[Timing] Speedup (Python/Rust):                     {python_waterfall_time/rust_waterfall_time:.1f}x")

        # 5. Compare
        passed = compare_graphs(rust_graph, python_graph, args.tol)

        if args.keep_outputs:
            print(f"\nTemporary files kept in: {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)

        sys.exit(0 if passed else 1)

    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        if args.keep_outputs:
            print(f"\nPartial outputs in: {tmpdir}", file=sys.stderr)
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
