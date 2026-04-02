from __future__ import absolute_import

import os
import subprocess
import numpy as np

import numpy.core.numeric as numeric
import multiprocessing as mp

from .. import __location__
from ..tools import mcl, trees_msa, waterfall
from . import orthogroups_set
from ..utils import util, files, matrices, parallel_task_manager


def WriteGraph_perSpecies(args):
    seqsInfo, graphFN, iSpec, d_pickle = args
    # calculate the 2-way connections for one query species
    with open(graphFN + "_%d" % iSpec, "w") as graphFile:
        connect2 = []
        for jSpec in range(seqsInfo.nSpecies):
            m1 = matrices.LoadMatrix("connect", iSpec, jSpec, d_pickle)
            m2tr = numeric.transpose(
                matrices.LoadMatrix("connect", jSpec, iSpec, d_pickle)
            )
            connect2.append(m1 + m2tr)
            del m1, m2tr
        B = matrices.LoadMatrixArray("B", seqsInfo, iSpec, d_pickle)
        B_connect = matrices.MatricesAnd_s(connect2, B)
        del B, connect2

        W = [b.sorted_indices().tolil() for b in B_connect]
        del B_connect
        for query in range(seqsInfo.nSeqsPerSpecies[seqsInfo.speciesToUse[iSpec]]):
            offset = seqsInfo.seqStartingIndices[iSpec]
            graphFile.write("%d    " % (offset + query))
            for jSpec in range(seqsInfo.nSpecies):
                row = W[jSpec].getrowview(query)
                jOffset = seqsInfo.seqStartingIndices[jSpec]
                for j, value in zip(row.rows[0], row.data[0]):
                    graphFile.write("%d:%.3f " % (j + jOffset, value))
            graphFile.write("$\n")
        if iSpec == (seqsInfo.nSpecies - 1):
            graphFile.write(")\n")
        # util.PrintTime("Written final scores for species %d to graph file" % iSpec)


def WriteGraph_perSpecies_homology(args):
    seqsInfo, graphFN, iSpec, d_pickle = args
    # calculate the 2-way connections for one query species
    # W = [matrices.LoadMatrix("B", iSpec, jSpec, d_pickle).tolil() for jSpec in range(seqsInfo.nSpecies)]
    W = []
    for jSpec in range(seqsInfo.nSpecies):
        w1 = matrices.LoadMatrix("B", iSpec, jSpec, d_pickle)
        matrices.DumpMatrix("H", (w1 > 0).tolil(), iSpec, jSpec, d_pickle)
        w2tr = numeric.transpose(matrices.LoadMatrix("B", jSpec, iSpec, d_pickle))
        W.append((w1 + w2tr > 0).tolil())  # symmetrise
    # matrices.DumpMatrixArray("H", W, iSpec, d_pickle)
    with open(graphFN + "_%d" % iSpec, "w") as graphFile:
        for query in range(seqsInfo.nSeqsPerSpecies[seqsInfo.speciesToUse[iSpec]]):
            offset = seqsInfo.seqStartingIndices[iSpec]
            graphFile.write("%d    " % (offset + query))
            for jSpec in range(seqsInfo.nSpecies):
                row = W[jSpec].getrowview(query)
                jOffset = seqsInfo.seqStartingIndices[jSpec]
                for j in row.rows[0]:
                    graphFile.write("%d:%.3f " % (j + jOffset, 1.0))
            graphFile.write("$\n")
        if iSpec == (seqsInfo.nSpecies - 1):
            graphFile.write(")\n")
        util.PrintTime("Written final scores for species %d to graph file" % iSpec)


def GetSequenceLengths(seqsInfo):
    sequenceLengths = []
    for iSpecies, iFasta in enumerate(seqsInfo.speciesToUse):
        sequenceLengths.append(np.zeros(seqsInfo.nSeqsPerSpecies[iFasta]))
        fastaFilename = files.FileHandler.GetSpeciesFastaFN(iFasta)
        currentSequenceLength = 0
        iCurrentSequence = -1
        qFirstLine = True
        with open(fastaFilename) as infile:
            for row in infile:
                if len(row) > 1 and row[0] == ">":
                    if qFirstLine:
                        qFirstLine = False
                    else:
                        sequenceLengths[iSpecies][
                            iCurrentSequence
                        ] = currentSequenceLength
                        currentSequenceLength = 0
                    _, iCurrentSequence = util.GetIDPairFromString(row[1:])
                else:
                    currentSequenceLength += len(row.rstrip())
        sequenceLengths[iSpecies][iCurrentSequence] = currentSequenceLength
    return sequenceLengths


def _run_rust_waterfall(seqsInfo, blastDir_list, options, i_unassigned):
    """Call the blast2mcl Rust binary to produce the MCL input graph.

    Replaces WaterfallMethod.ProcessBlastHits + ConnectCognates +
    WriteGraphParallel for gathering_version < (3, 0).

    Raises FileNotFoundError if the binary is missing (no Python fallback —
    build blast2mcl from blast2mcl/ with 'cargo build --release' and copy
    the result to src/orthofinder/bin/).

    NOTE: unassigned-genes mode (i_unassigned is not None) is not yet
    supported by blast2mcl; add --allow-empty handling in blast_parser.rs
    and wire it through if needed.
    """
    if i_unassigned is not None:
        raise NotImplementedError(
            "blast2mcl does not yet support the incremental unassigned-genes "
            "mode (--fast-add).  Run without --fast-add, or implement "
            "--allow-empty in blast2mcl/src/blast_parser.rs."
        )

    blast2mcl_bin = os.path.join(__location__, "bin", "blast2mcl")
    if not os.path.isfile(blast2mcl_bin):
        raise FileNotFoundError(
            f"blast2mcl binary not found at {blast2mcl_bin!r}.\n"
            "Build it from blast2mcl/ with:\n"
            "  cargo build --release\n"
            "  cp blast2mcl/target/release/blast2mcl src/orthofinder/bin/"
        )

    graphFilename = files.FileHandler.GetGraphFilename(i_unassigned)

    cmd = [
        blast2mcl_bin,
        "--blast-dir", ",".join(blastDir_list),
        "--fasta-dir", ",".join(blastDir_list),
        "--species-to-use", ",".join(str(s) for s in seqsInfo.speciesToUse),
        "--n-seqs-per-species",
            ",".join(str(seqsInfo.nSeqsPerSpecies[s]) for s in seqsInfo.speciesToUse),
        "--output", graphFilename,
        "--threads", str(options.nProcessAlg),
    ]
    if options.v2_scores:
        cmd.append("--v2-scores")
    if not options.qDoubleBlast:
        cmd.extend(["--double-blast", "false"])

    util.PrintTime("Running blast2mcl (Rust waterfall)")
    result = subprocess.run(cmd, env=parallel_task_manager.my_env)
    if result.returncode != 0:
        files.FileHandler.LogFailAndExit("ERROR: blast2mcl failed (see output above)")

    util.PrintTime("blast2mcl complete")
    return graphFilename


def _run_python_waterfall_lt3(seqsInfo, blastDir_list, Lengths, options,
                               i_unassigned, GRACE_PERIOD, STALL_TIMEOUT):
    """Original Python waterfall for gathering_version < (3, 0).

    Kept intact for correctness testing via tools/test_waterfall.py.
    Do not call from DoOrthogroups directly; use _run_rust_waterfall instead.
    """
    total_tasks = seqsInfo.nSpecies
    files.FileHandler.GetPickleDir()

    if options.old_version:
        cmd_queue = mp.Queue()
        for iSpeciesJob in range(seqsInfo.nSpecies):
            cmd_queue.put(iSpeciesJob)
        runningProcesses = [
            mp.Process(
                target=waterfall.WaterfallMethod.Worker_ProcessBlastHits,
                args=(seqsInfo, blastDir_list, Lengths, cmd_queue,
                      files.FileHandler.GetPickleDir(), options.qDoubleBlast,
                      options.v2_scores, i_unassigned is not None),
            )
            for _ in range(options.nProcessAlg)
        ]
        for proc in runningProcesses:
            proc.start()
        parallel_task_manager.ManageQueue(runningProcesses, cmd_queue)
    else:
        cmd_queue = mp.Queue()
        for iSpeciesJob in range(seqsInfo.nSpecies):
            cmd_queue.put(iSpeciesJob)
        for _ in range(options.nProcessAlg):
            cmd_queue.put(None)
        result_queue = mp.Queue()
        runningProcesses = [
            mp.Process(
                target=waterfall.WaterfallMethod.Worker_ProcessBlastHits_New,
                args=(seqsInfo, blastDir_list, Lengths, cmd_queue,
                      files.FileHandler.GetPickleDir(), options.qDoubleBlast,
                      options.v2_scores, i_unassigned is not None, result_queue),
            )
            for _ in range(options.nProcessAlg)
        ]
        parallel_task_manager.ManageQueueNew(
            runningProcesses, total_tasks, options.nProcessAlg, result_queue,
            GRACE_PERIOD=GRACE_PERIOD, STALL_TIMEOUT=STALL_TIMEOUT
        )

    util.PrintTime("Connected putative homologues")

    if options.old_version:
        cmd_queue = mp.Queue()
        for iSpecies in range(seqsInfo.nSpecies):
            cmd_queue.put((seqsInfo, iSpecies))
        runningProcesses = [
            mp.Process(
                target=waterfall.WaterfallMethod.Worker_ConnectCognates,
                args=(cmd_queue, files.FileHandler.GetPickleDir(), options.v2_scores),
            )
            for _ in range(options.nProcessAlg)
        ]
        for proc in runningProcesses:
            proc.start()
        parallel_task_manager.ManageQueue(runningProcesses, cmd_queue)
    else:
        cmd_queue = mp.Queue()
        for iSpecies in range(seqsInfo.nSpecies):
            cmd_queue.put((seqsInfo, iSpecies))
        for _ in range(options.nProcessAlg):
            cmd_queue.put(None)
        result_queue = mp.Queue()
        runningProcesses = [
            mp.Process(
                target=waterfall.WaterfallMethod.Worker_ConnectCognates_New,
                args=(cmd_queue, result_queue,
                      files.FileHandler.GetPickleDir(), options.v2_scores),
            )
            for _ in range(options.nProcessAlg)
        ]
        parallel_task_manager.ManageQueueNew(
            runningProcesses, total_tasks, options.nProcessAlg, result_queue,
            GRACE_PERIOD=GRACE_PERIOD, STALL_TIMEOUT=STALL_TIMEOUT
        )

    return waterfall.WaterfallMethod.WriteGraphParallel(
        WriteGraph_perSpecies, seqsInfo, options.nProcessAlg, i_unassigned
    )


def DoOrthogroups(
        options,
        speciesInfoObj,
        seqsInfo,
        speciesNamesDict,
        speciesXML=None,
        i_unassigned=None,
        GRACE_PERIOD = 10.,
        STALL_TIMEOUT = 200.
    ):

    # Run Algorithm, cluster and output cluster files with original accessions
    q_unassigned = i_unassigned is not None
    util.PrintUnderline(
        "Running OrthoFinder algorithm"
        + (" for clade-specific genes" if q_unassigned else "")
    )

    blastDir_list = files.FileHandler.GetBlastResultsDir()
    if q_unassigned:
        blastDir_list = blastDir_list[:1]

    if options.gathering_version < (3, 0):
        graphFilename = _run_rust_waterfall(
            seqsInfo, blastDir_list, options, i_unassigned
        )
        clustersFilename, clustersFilename_pairs = (
            files.FileHandler.CreateUnusedClustersFN(
                "_I%0.1f" % options.mclInflation, i_unassigned
            )
        )
        mcl.MCL.RunMCL(
            graphFilename, clustersFilename, options.nProcessAlg, options.mclInflation
        )
        mcl.ConvertSingleIDsToIDPair(
            seqsInfo, clustersFilename, clustersFilename_pairs, q_unassigned
        )

    elif options.gathering_version == (3, 2):
        # TODO (blast2mcl Phase 3): gathering_version == (3, 2) uses an unweighted
        # homology-only graph (WriteGraph_perSpecies_homology).  It does not need
        # score normalisation or ConnectCognates — just boolean B matrices written
        # directly.  Port WriteGraph_perSpecies_homology to Rust following the
        # blast2mcl template, adding a --homology-only flag.
        Lengths = GetSequenceLengths(seqsInfo)
        util.PrintTime("Initial processing of each species")
        files.FileHandler.GetPickleDir()
        total_tasks = seqsInfo.nSpecies
        cmd_queue = mp.Queue()
        for iSpeciesJob in range(seqsInfo.nSpecies):
            cmd_queue.put(iSpeciesJob)
        for _ in range(options.nProcessAlg):
            cmd_queue.put(None)
        result_queue = mp.Queue()
        runningProcesses = [
            mp.Process(
                target=waterfall.WaterfallMethod.Worker_ProcessBlastHits_New,
                args=(seqsInfo, blastDir_list, Lengths, cmd_queue,
                      files.FileHandler.GetPickleDir(), options.qDoubleBlast,
                      options.v2_scores, q_unassigned, result_queue),
            )
            for _ in range(options.nProcessAlg)
        ]
        parallel_task_manager.ManageQueueNew(
            runningProcesses, total_tasks, options.nProcessAlg, result_queue,
            GRACE_PERIOD=GRACE_PERIOD, STALL_TIMEOUT=STALL_TIMEOUT
        )
        graphFilename = waterfall.WaterfallMethod.WriteGraphParallel(
            WriteGraph_perSpecies_homology, seqsInfo, options.nProcessAlg, i_unassigned
        )
        clustersFilename, clustersFilename_pairs = (
            files.FileHandler.CreateUnusedClustersFN(
                "_I%0.1f" % options.mclInflation, i_unassigned
            )
        )
        mcl.MCL.RunMCL(
            graphFilename, clustersFilename, options.nProcessAlg, options.mclInflation
        )
        mcl.ConvertSingleIDsToIDPair(
            seqsInfo, clustersFilename, clustersFilename_pairs, q_unassigned
        )
    if not q_unassigned:
        post_clustering_orthogroups(
            clustersFilename_pairs,
            speciesInfoObj,
            seqsInfo,
            speciesNamesDict,
            options,
            speciesXML,
        )
    return clustersFilename_pairs


def post_clustering_orthogroups(
        clustersFilename_pairs,
        speciesInfoObj,
        seqsInfo,
        speciesNamesDict,
        options,
        speciesXML,
        q_incremental=False,
    ):
    """
    Write OGs & statistics to results files, write Fasta files.
    Args:
        q_incremental - These are not the final orthogroups, don't write results
    """
    ogs = mcl.GetPredictedOGs(clustersFilename_pairs)
    resultsBaseFilename = files.FileHandler.GetOrthogroupResultsFNBase()

    # if not options.fix_files:
    if not options.qStopAfterTrees:
        util.PrintUnderline("Writing orthogroups to file")
        idsDict = mcl.MCL.WriteOrthogroupFiles(
            ogs,
            [files.FileHandler.GetSequenceIDsFN()],
            resultsBaseFilename,
            clustersFilename_pairs,
        )
    else:
        try:
            idsDict = mcl.IDFullDict(
                [files.FileHandler.GetSequenceIDsFN()], 
                func=util.FirstWordExtractor
            )
        except:
            idsDict = mcl.IDFullDict(
                [files.FileHandler.GetSequenceIDsFN()], 
                func=util.FullAccession
            )
    
    ## --------- this doesn't need to run at this point with the new process --------
    if not options.fix_files:
        if not q_incremental:
            mcl.MCL.CreateOrthogroupTable(
                ogs,
                idsDict,
                speciesNamesDict,
                speciesInfoObj.speciesToUse,
                resultsBaseFilename,
            )

    # Write Orthogroup FASTA files
    ogSet = orthogroups_set.OrthoGroupsSet(
        options.min_seq,
        files.FileHandler.GetWorkingDirectory1_Read(),
        speciesInfoObj.speciesToUse,
        speciesInfoObj.nSpAll,
        options.qAddSpeciesToIDs,
        options.tree_program,
        idExtractor=util.FirstWordExtractor,
    )


    treeGen = trees_msa.TreesForOrthogroups(None, None, None)
    fastaWriter = trees_msa.FastaWriter(
        files.FileHandler.GetSpeciesSeqsDir(), speciesInfoObj.speciesToUse
    )

    # d_seqs = files.FileHandler.GetResultsSeqsDir()
    # if not os.path.exists(d_seqs):
    #     os.mkdir(d_seqs)

    # treeGen.WriteFastaFiles(fastaWriter, ogSet.OGsAll(), idsDict, False)

    d_seqs_id = files.FileHandler.GetSeqsIDDir()
    if not os.path.exists(d_seqs_id):
        os.mkdir(d_seqs_id)

    qResults=False
    if not options.fix_files:
        d_seqs = files.FileHandler.GetResultsSeqsDir()
        if not os.path.exists(d_seqs):
            os.mkdir(d_seqs)
        qResults = True 

    treeGen.WriteFastaFiles(fastaWriter, ogSet.OGsAll(), idsDict, qID=True, qResults=qResults)

    if not q_incremental:
        # stats.Stats(ogs, speciesNamesDict, speciesInfoObj.speciesToUse, files.FileHandler.iResultsVersion)
        if options.speciesXMLInfoFN:
            mcl.MCL.WriteOrthoXML(
                speciesXML,
                ogs,
                seqsInfo.nSeqsPerSpecies,
                idsDict,
                resultsBaseFilename + ".orthoxml",
                speciesInfoObj.speciesToUse,
            )
        # print("")
        util.PrintTime("Done orthogroups")
        files.FileHandler.LogOGs()
