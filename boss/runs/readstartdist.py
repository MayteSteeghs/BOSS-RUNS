from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln

from boss.paf import Paf



class ReadStartDist:

    def __init__(self, contigs: dict, windows_per_bin: int | None = None, alpha: float = 1.0, p0: float = 0.1):
        """
        Initialise the probability distribution of fragment start sites

        :param contigs: Dictionary of contig objects
        :param windows_per_bin: Number of decision windows per bin of counting read starts.
        :param alpha: prior for alpha (hyperparameter for dirichlet prior)
        :param p0: Prior for sites with 0 observations
        """
        self.alpha = alpha
        self.p0 = p0
        # size of the decision windows of the strategy, the same for all contigs
        self.decision_window = next(iter(contigs.values())).window_size
        self.windows_per_bin = windows_per_bin or max(1, round(2000 / self.decision_window))
        # number of decision windows per contig, fhat is expanded to exactly these
        self.n_windows = {cname: c.n_windows for cname, c in contigs.items()}
        # number of decision windows in each bin. The remainder of a contig is folded into its last bin
        self.bin_windows = {}
        for cname, c in contigs.items():
            n_bins = max(1, c.n_windows // self.windows_per_bin)
            bin_windows = np.full(n_bins, self.windows_per_bin)
            bin_windows[-1] = c.n_windows - (n_bins - 1) * self.windows_per_bin
            self.bin_windows[cname] = bin_windows
        # track read start positions (forward and rev) in bins of windows_per_bin decision windows
        self.read_starts = {cname : np.zeros(shape=(self.bin_windows[cname].shape[0], 2)) for cname in contigs.keys()}  # NOTE: No additional dimension for barcodes in this initial implementation
        # fhat exists only in its merged form, i.e. for use in updating on a merged array
        self.total_len = np.sum([a.shape[0] for a in self.read_starts.values()])
        self.target_size = int(np.sum([c.n_windows for c in contigs.values()]))
        self.on_target = 1   # TODO
        self.fhat = self.update_f_pointmass()



    def merge(self) -> NDArray:
        """
        :return: Concatenated array of read starting sites across all contigs
        """
        return np.concatenate(list(self.read_starts.values()))



    def count_read_starts(self, paf_dict: dict[str, list]) -> None:
        """
        Keep track of the read starting positions C_{i,o}, used to update F
        Read starts are saved in non-overlapping bins of windows_per_bin decision windows,
        the last bin of a contig also holds the remaining windows

        :param paf_dict: Dictionary of read mappings
        :return:
        """
        # collect all starting positions
        starts_fwd = defaultdict(list)
        starts_rev = defaultdict(list)
        
        for rid in paf_dict.keys():
            rec = paf_dict[rid]
            # choose the highest ranked mapping
            if len(rec) > 1:
                rec = Paf.choose_best_mapper(rec)[0]
            else:
                rec = rec[0]

            if rec.rev:
                starts_rev[rec.tname].append(rec.tend - 1)
            else:
                starts_fwd[rec.tname].append(rec.tstart)

        # only contigs with new read starts need updating
        for cname in (starts_fwd.keys() | starts_rev.keys()) & self.read_starts.keys():
            r_starts = self.read_starts[cname]
            n_bins = int(r_starts.shape[0])
            for strand, starts in enumerate((starts_fwd[cname], starts_rev[cname])):
                # decision window each read start falls into, ignoring positions outside of the contig
                windows = np.asarray(starts, dtype=int) // self.decision_window
                windows = windows[(windows >= 0) & (windows < self.n_windows[cname])]
                # count the number of read starts in bins, the remaining windows belong to the last bin
                bins = np.minimum(windows // self.windows_per_bin, n_bins - 1)
                # add new counts to the array
                r_starts[:, strand] += np.bincount(bins, minlength=n_bins)



    def update_f_pointmass(self) -> NDArray:
        """
        Bayesian approach to update posterior values of f_hat using counts of read starting positions.
        Version with point mass at 0 (for sites with C == 0)

        :return: Array of read starting probabilities for all positions
        """
        # concatenate arrays
        merged = self.merge()
        fhat = np.zeros(shape=merged.shape)
        # dirichlet parameter of each bin, weighted by the number of decision windows it holds
        bin_windows = np.concatenate(list(self.bin_windows.values()))
        alphas = np.repeat((self.alpha * bin_windows / self.windows_per_bin)[:, np.newaxis], 2, axis=1)
        # equals 2 * n_bins * alpha if all bins are full
        alpha_sum = np.sum(alphas)
        # First, sites with C > 0
        nonzero_indices = np.nonzero(merged)
        nonzero = merged[nonzero_indices]
        num = np.add(alphas[nonzero_indices], nonzero)
        Csum = np.sum(nonzero)
        denom = alpha_sum + Csum
        fhat[nonzero_indices] = np.divide(num, denom)
        # then sites with C == 0
        rhs = (alphas / (alpha_sum + Csum))
        # ratio of beta functions (Suppl. Eq. S.24), calculated in log space to avoid underflow
        beta_num = betaln(alphas, (alpha_sum - alphas + Csum))
        beta_denom = betaln(alphas, (alpha_sum - alphas))
        p0_bit = self.p0 / (self.p0 + (1 - self.p0) * np.exp(beta_num - beta_denom))
        lhs = 1 - p0_bit
        expectedPost = lhs * rhs
        # mask for the zero count sites - derived from nonzero indices
        zero_indices = np.ones(shape=fhat.shape, dtype="bool")
        zero_indices[nonzero_indices] = 0
        fhat[zero_indices] = expectedPost[zero_indices]
        # expand from downsampled size
        fhat_exp = self._expand_fhat(fhat)
        return fhat_exp



    def _expand_fhat(self, fhat: NDArray) -> NDArray:
        """
        Expand and normalise Fhat from bins to the decision windows of each contig.

        :param fhat: probabilities of read starting positions in bins, merged across contigs
        :return: read start probs in decision windows, merged across contigs
        """
        fhat_parts = []
        offset = 0
        for bin_windows in self.bin_windows.values():
            n_bins = bin_windows.shape[0]
            fhat_c = fhat[offset: offset + n_bins] / bin_windows[:, np.newaxis]
            fhat_parts.append(np.repeat(fhat_c, bin_windows, axis=0))
            offset += n_bins
        fhat_exp = np.concatenate(fhat_parts)
        assert fhat_exp.shape[0] == self.target_size

        # normalise only if not empty
        fhat_sum = np.sum(fhat_exp)
        if fhat_sum != 0:
            # normalise for ratio of on/off target reads
            # uses estimated on-target proportion from initially accepted sites
            # TODO: better understand what this ratio means in a barcoded sample and whether this is still sensible
            normalizer = self.on_target / fhat_sum     # TODO get the on-target estimator
            fhat_exp = np.multiply(fhat_exp, normalizer)
        return fhat_exp



    def estimate_priors(self) -> tuple[float, float]:
        """
        Estimate alpha, the concentration hyperparameter of the dirichlet prior,
        by equating the variance of Fhat with the variance of the dirichlet.

        :return: Prior of dirichlet and proportion of gap sites
        """
        # TODO: scale priors to bin sizes
        merged = self.merge()
        n_windows = merged.shape[0]
        # filter readStartCounts for positions with 0
        zeroCounts = np.count_nonzero(merged == 0)
        p0 = zeroCounts / (n_windows * 2)
        # get sum of mapped reads
        Csum = np.sum(merged) or 1e-30
        # simple estimator of alpha
        Fhat = np.divide(merged, Csum)
        # variance of Fhat to equate with var of D(alpha)
        Vhat = np.var(Fhat, ddof=0) or 1e-30
        # if Vhat is very small, alpha becomes very large
        lhs = (2 * n_windows - 1) / (Vhat * 8 * (n_windows ** 3))
        rhs = 1 / (2 * n_windows)
        alpha = float(lhs - rhs)
        return alpha, p0



