from types import SimpleNamespace

import pytest
import numpy as np
from scipy.special import beta, betaln

import boss.runs.readstartdist as br_rsd
from boss.runs.reference import Contig


@pytest.fixture
def read_start_dist(zymo_ref):
    contigs = zymo_ref.contigs
    rsd = br_rsd.ReadStartDist(contigs=contigs)
    return rsd



def test_init(read_start_dist, zymo_ref):
    assert len(read_start_dist.read_starts) == 9
    # bins of 20 decision windows (2000 bases) per contig, the remainder is folded into the last bin
    assert read_start_dist.windows_per_bin == 20
    assert read_start_dist.total_len == 15502
    assert all(bw.min() >= 20 for bw in read_start_dist.bin_windows.values())
    # target size for expanding fhat
    # this should be: total number of decision windows across contigs, as in the strategy arrays
    n_windows = sum(c.n_windows for c in zymo_ref.contigs.values())
    assert n_windows == sum(c.strat.shape[0] for c in zymo_ref.contigs.values())
    assert read_start_dist.target_size == n_windows


def test_count_read_starts(read_start_dist, paf_dict, zymo_ref):
    read_starts_merged = read_start_dist.merge()
    assert read_starts_merged.shape == (15502, 2)
    assert np.sum(read_starts_merged) == 0
    read_start_dist.count_read_starts(paf_dict=paf_dict)
    read_starts_merged = read_start_dist.merge()
    assert read_starts_merged.shape == (15502, 2)
    # every read is counted, including those starting in the remainder at the end of a contig
    len_paf_dict = len(paf_dict)
    assert np.sum(read_starts_merged) == len_paf_dict
    # update fhat after counting read starts
    fhat = read_start_dist.update_f_pointmass()
    assert np.isclose(fhat.min(), 1.1566900720720178e-06)
    assert np.isclose(fhat.max(), 1.63650472347512e-05)
    n_windows = sum(c.n_windows for c in zymo_ref.contigs.values())
    assert fhat.shape == (n_windows, 2)


def test_estimate_priors(read_start_dist, paf_dict):
    read_starts_merged = read_start_dist.merge()
    assert np.sum(read_starts_merged) == 0
    read_start_dist.count_read_starts(paf_dict=paf_dict)
    alpha, p0 = read_start_dist.estimate_priors()
    assert np.isclose(alpha, 0.08691508698761234)
    assert np.isclose(p0, 0.8897884143981422)



@pytest.mark.parametrize("n_reads", [0, 1_000, 100_000])
def test_update_f_pointmass_empty_windows(n_reads):
    # empty windows follow Suppl. Eq. S.23: their weight relative to a window with C reads
    # is (1 - p0 / (p0 + (1 - p0) * P(C=0|F>0))) * a / (a + C)
    contigs = {"c0": SimpleNamespace(length=200_000, n_windows=2000, window_size=100)}
    rsd = br_rsd.ReadStartDist(contigs=contigs)
    counts = rsd.read_starts["c0"]
    counts[0, 0] = n_reads
    a, p0 = rsd.alpha, rsd.p0
    n_pairs = counts.size
    csum = counts.sum()
    p_zero = np.exp(betaln(a, (n_pairs - 1) * a + csum) - betaln(a, (n_pairs - 1) * a))
    lhs = 1 - p0 / (p0 + (1 - p0) * p_zero)
    fhat = rsd.update_f_pointmass()
    # first window of the contig holds the counted reads (or is empty if n_reads == 0), the last window is empty
    ratio = fhat[-1, 1] / fhat[0, 0]
    expected = lhs * a / (a + n_reads) if n_reads else 1.0
    assert np.isclose(ratio, expected)
    # empty windows are down-weighted more as more reads are observed
    if n_reads:
        assert lhs < 1 - p0



@pytest.mark.parametrize("ws", [100, 60])
def test_fhat_aligned_with_contigs(ws):
    # unequal, non-round contig lengths, and decision windows that do and don't divide 2000
    lengths = [25_111, 40_007, 12_003]
    rng = np.random.default_rng(0)
    contigs = {f"c{i}": Contig(name=f"c{i}", seq="".join(rng.choice(list("ACGT"), size=l)), window_size=ws)
               for i, l in enumerate(lengths)}
    rsd = br_rsd.ReadStartDist(contigs=contigs)
    # read starts only on the middle contig, including its last base on both strands
    starts = [0, 1_000, 20_000, 39_999, 40_006]
    paf_dict = {f"r{i}": [SimpleNamespace(tname="c1", tstart=s, tend=s + 1, rev=bool(i % 2))]
                for i, s in enumerate(starts)}
    paf_dict["r_end"] = [SimpleNamespace(tname="c1", tstart=39_000, tend=40_007, rev=True)]
    rsd.count_read_starts(paf_dict=paf_dict)
    assert rsd.merge().sum() == len(paf_dict)
    fhat = rsd.update_f_pointmass()
    # exactly one row per decision window, no padding or trimming needed
    offsets = np.cumsum([0] + [c.n_windows for c in contigs.values()])
    assert fhat.shape == (offsets[-1], 2)
    # the read starts raise fhat on the middle contig only, in the windows they start in
    empty = np.r_[offsets[0]:offsets[1], offsets[2]:offsets[3]]
    empty_max = fhat[empty].max()
    for i, s in enumerate(starts):
        assert fhat[offsets[1] + s // ws, i % 2] > 1.5 * empty_max
    assert fhat[offsets[2] - 1, 1] > 1.5 * empty_max



def test_folded_last_bin_prior():
    # the remainder of a contig is folded into its last bin, whose prior is weighted by the windows it holds
    ws = 100
    contigs = {"c0": SimpleNamespace(length=5_550, n_windows=56, window_size=ws)}
    rsd = br_rsd.ReadStartDist(contigs=contigs)
    assert rsd.windows_per_bin == 20
    # 56 windows: one bin of 20 and a last bin holding the remaining 36
    assert list(rsd.bin_windows["c0"]) == [20, 36]
    fhat = rsd.update_f_pointmass()
    assert fhat.shape == (56, 2)
    # without any reads every decision window gets the same probability, the longer last bin gets no more per window
    assert np.allclose(fhat, fhat[0, 0])
    # a read starting in the remainder is counted in the last bin
    paf_dict = {"r0": [SimpleNamespace(tname="c0", tstart=5_500, tend=5_501, rev=False)]}
    rsd.count_read_starts(paf_dict=paf_dict)
    assert rsd.read_starts["c0"][:, 0].tolist() == [0, 1]



def test_weighted_prior_formula():
    # a contig of 73 windows: two full bins of 20 and a last bin holding the remaining 33
    contigs = {"c0": SimpleNamespace(length=7_300, n_windows=73, window_size=100)}
    rsd = br_rsd.ReadStartDist(contigs=contigs)
    bin_windows = rsd.bin_windows["c0"]
    assert list(bin_windows) == [20, 20, 33]
    counts = np.array([[3, 0], [0, 0], [5, 1]], dtype=float)
    rsd.read_starts["c0"][:] = counts
    fhat = rsd.update_f_pointmass()
    # expected values, computed independently: dirichlet parameter of each bin weighted by its size
    a = rsd.alpha * (bin_windows / rsd.windows_per_bin)[:, np.newaxis] * np.ones((1, 2))
    a_sum, c_sum = a.sum(), counts.sum()
    # Suppl. Eq. S.21 for bins with read starts
    f_bins = (a + counts) / (a_sum + c_sum)
    # Suppl. Eq. S.23 and S.24 for bins without read starts
    p_zero = beta(a, a_sum - a + c_sum) / beta(a, a_sum - a)
    lhs = 1 - rsd.p0 / (rsd.p0 + (1 - rsd.p0) * p_zero)
    f_bins[counts == 0] = (lhs * a / (a_sum + c_sum))[counts == 0]
    # each bin's probability spread over its decision windows, normalised to sum to on_target
    expected = np.repeat(f_bins / bin_windows[:, np.newaxis], bin_windows, axis=0)
    expected *= rsd.on_target / expected.sum()
    assert fhat.shape == (73, 2)
    assert np.allclose(fhat, expected)



def test_weighted_prior_equal_density():
    # a full bin of 20 windows and a folded last bin of 30 windows with the same density of read starts
    contigs = {"c0": SimpleNamespace(length=5_000, n_windows=50, window_size=100)}
    rsd = br_rsd.ReadStartDist(contigs=contigs)
    assert list(rsd.bin_windows["c0"]) == [20, 30]
    rsd.read_starts["c0"][:, 0] = [20, 30]
    fhat = rsd.update_f_pointmass()
    # with the prior weighted by bin size, both bins give the same probability per decision window
    assert np.isclose(fhat[0, 0], fhat[-1, 0])
    # an unweighted prior (alpha per bin) would not: (alpha + 20) / 20 != (alpha + 30) / 30
    a = rsd.alpha
    assert not np.isclose((a + 20) / 20, (a + 30) / 30)
