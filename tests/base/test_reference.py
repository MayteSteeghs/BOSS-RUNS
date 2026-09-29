from types import SimpleNamespace
import logging
import pytest
import numpy as np

import boss.runs.reference as refc
from boss.runs.core import BossRuns
from boss.runs.readstartdist import ReadStartDist
from boss.runs.sequences import Scoring

from ..constants import PATHS


@pytest.mark.parametrize("name, seq, ploidy, barcodes", [
    ("ch1", "ACGTACGT", 1, None),
    ("ch1", "ACGTACGTNnWwIi", 1, None),
    ("ch1", "ACGTACGT", 2, None),
    ("ch2", "ACgtacGT", 1, None),
    ("ch3  r", "ACGTACGT", 1, None),
    ("ch_garbage", "NotaREALSequenceGT", 1, None),
    ("ch1_bc", "ACGTACGT", 1, ["barcode01", "barcode02"])
])

def test_contig(name, seq, ploidy,barcodes):
    c = refc.Contig(name=name, seq=seq, ploidy=ploidy,barcodes=barcodes)
    logging.info(c.name)
    logging.info(c.seq)
    logging.info(c.seq_int)
    logging.info(c.barcodes)
    if not barcodes:
        len_bc = 1
    else:
        len_bc = len(barcodes)
    assert c.length == len(seq)
    assert c.length == len(c.seq_int)
    assert " " not in c.name
    assert c.coverage.shape == (len(seq), 5, len_bc)
    assert c.coverage.sum() == 0
    assert c.bucket_switches.shape == (max(1, len(seq) // 20_000), len_bc)
    assert all(c.initial_scores[0] == c.score0)


@pytest.mark.xfail(raises=ValueError)
def test_contig_ploidy():
    c = refc.Contig(name="contig1", seq="ACGT", ploidy=3)
    logging.info(c.name)


@pytest.mark.xfail(raises=AssertionError)
def test_contig_seq():
    c = refc.Contig(name="contig1", seq="Not a REAL Sequence", ploidy=1)
    logging.info(c.name)




@pytest.mark.parametrize("ref, mmi, reject_refs, nsites, barcodes", [
    (PATHS.fasta, None, "", 31012581, None),
    (PATHS.fasta, PATHS.mmi, "", 31012581, None),
    (PATHS.fasta, PATHS.mmi, "NZ_CP041014.1,NZ_VFAE01000004.1,NZ_VFAG01000001.1", 27910526, None),
    (PATHS.fasta, PATHS.mmi, "NZ_CP041014.1,NZ_VFAE01000004.1,NZ_VFAG01000001.1", 27910526, ["barcode01", "barcode02"]),
])
def test_reference(ref, mmi, reject_refs, nsites, barcodes, request):
    # only grab fixture if mmi is actually passed as param
    r = refc.Reference(
        ref=ref,
        mmi=mmi,
        reject_refs=reject_refs,
        barcodes=barcodes
    )
    assert len(r.contigs) == 9
    assert r.n_sites == nsites
    assert r.barcodes == barcodes


@pytest.mark.xfail(raises=FileNotFoundError)
def test_reference_notafile():
    _ = refc.Reference(ref="not_a_real_file")


@pytest.mark.xfail(raises=ValueError)
def test_reference_notfasta():
    _ = refc.Reference(ref=PATHS.fastq)



@pytest.mark.xfail(raises=FileNotFoundError)
def test_reference_unrealmmi():
    _ = refc.Reference(ref=PATHS.fasta, mmi="not_a_real_file")


def test_contig_dicts():
    r = refc.Reference(ref=PATHS.fasta, mmi=PATHS.mmi)
    cs = r.contig_sequences()
    cl = r.contig_lengths()
    assert isinstance(cs, dict)
    assert isinstance(cl, dict)
    assert len(cs) == 9
    assert len(cl) == 9



def test_contig_buckets_folded():
    # 50 kb contig: a bucket of 20 kb, and a last bucket holding the remaining 30 kb
    rng = np.random.default_rng(0)
    c = refc.Contig(name="c0", seq="".join(rng.choice(list("ACGT"), size=50_000)))
    assert list(c.bucket_starts) == [0, 20_000]
    assert list(c.bucket_lengths) == [20_000, 30_000]
    assert c.bucket_switches.shape == (2, 1)
    # coverage of 8 only in the remainder after 40 kb: mean over the last bucket is 8 * 10 / 30 < 5
    c.coverage[40_000:, 0, 0] = 8
    c.check_buckets(threshold=5)
    assert not c.bucket_switches.any()
    # coverage of 6 across the whole last bucket switches it on, including its remainder
    c.coverage[20_000:, 0, 0] = 6
    c.check_buckets(threshold=5)
    assert c.bucket_switches[:, 0].tolist() == [False, True]



@pytest.mark.parametrize("ws", [100, 60])
@pytest.mark.parametrize("remainder", [False, True])
def test_window_array_lengths(ws, remainder):
    # contig lengths that are a multiple of the decision window size, or leave a remainder
    base = [30_000, 42_000]
    lengths = [l + 37 for l in base] if remainder else base
    assert all((l % ws != 0) == remainder for l in lengths)
    rng = np.random.default_rng(0)
    contigs = {f"c{i}": refc.Contig(name=f"c{i}", seq="".join(rng.choice(list("ACGT"), size=l)), window_size=ws)
               for i, l in enumerate(lengths)}
    # read lengths up to longer than a contig, to also exercise the clamped moving sums
    approx_ccl = np.linspace(1_000, 50_000, 10).astype(int)
    for l, c in zip(lengths, contigs.values()):
        # one window per started block of ws bases, and the last base falls into the last window
        assert c.n_windows == -(-l // ws)
        assert (l - 1) // ws == c.n_windows - 1
        c.calc_smu()
        c.calc_u(approx_ccl=approx_ccl)
        for arr in (c.strat, c.scores_ds, c.smu, c.expected_benefit, c.additional_benefit):
            assert arr.shape[0] == c.n_windows
    n_total = sum(c.n_windows for c in contigs.values())
    # merged arrays used to calculate the strategy
    benefit, smu = Scoring.merge_benefit(contigs)
    rsd = ReadStartDist(contigs=contigs)
    fhat = rsd.update_f_pointmass()
    assert benefit.shape[0] == smu.shape[0] == fhat.shape[0] == rsd.target_size == n_total
    # distributing a merged strategy gives each contig exactly its own windows
    for c in contigs.values():
        c.bucket_switches[:] = True
    merged_strat = rng.random((n_total, 2, 1)) < 0.5
    runs = SimpleNamespace(contigs_filt=contigs, args=SimpleNamespace(
        optional=SimpleNamespace(window_size=ws), general=SimpleNamespace(barcodes=None)))
    BossRuns._distribute_strategy(runs, strat=merged_strat)
    offset = 0
    for c in contigs.values():
        assert np.array_equal(c.strat, merged_strat[offset: offset + c.n_windows])
        offset += c.n_windows
    assert offset == n_total
