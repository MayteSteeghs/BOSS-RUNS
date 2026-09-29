import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from boss.core import Boss
from boss.mapper import Mapper
from boss.runs.abundance_tracker import AbundanceTracker
from boss.runs.readstartdist import ReadStartDist
from boss.runs.reference import Reference
from boss.runs.sequences import CoverageConverter, Scoring


class BossRuns(Boss):


    def init(self) -> None:
        """
        Initialise necessary components for a BOSS-RUNS experiment
        These components are shared across the live and simulation experiments,
        i.e. this init function is also run in the init of the simulation

        :return:
        """
        if not self.args.general.barcodes:
            self.barcodes_index = {"": 0}
        else:
            self.barcodes_index = {int(bc.split('barcode')[1]): i for i, bc in enumerate(self.args.general.barcodes)}
        self.nbarcodes = len(self.barcodes_index)
        # initialise reference
        assert self.args.general.ref is not None
        self.ref = Reference(ref=self.args.general.ref, mmi=self.args.general.mmi, reject_refs=self.args.optional.reject_refs, barcodes=self.args.general.barcodes, window_size=self.args.optional.window_size)
        self.contigs = self.ref.contigs
        self.contigs_filt = {n: c for n, c in self.contigs.items() if not c.rej}  # NOTE: This could potentially be different for different barcodes, consider and implement if applicable
        # initialise a mapper using the reference
        assert self.ref.mmi is not None
        self.mapper = Mapper(ref=self.ref.mmi)
        # initialise a translator for coverage conversions
        self.cc = CoverageConverter()
        # initialise the abundance tracker
        self.tracker = AbundanceTracker(contigs=self.contigs)
        # initialise the tracker for read start position distribution
        self.read_starts = ReadStartDist(contigs=self.contigs_filt)
        # initialise scoring array
        self.scoring = Scoring(ploidy=self.args.optional.ploidy)
        self.scoring.init_score_array()
        # write initial strategies to file
        strat_dict = self.ref.get_strategy_dict()
        self._write_contig_strategies(contig_strats=strat_dict)



    def _write_contig_strategies(self, contig_strats: dict[str, NDArray]) -> None:
        """
        Write the strategies for all contigs to a single file.

        :param contig_strats: A dictionary containing the strategies for contigs.
        """
        cpath_tmp = f'{self.out_dir}/masks/boss_tmp.npz'
        np.savez(cpath_tmp, **contig_strats)
        # after writing to tmpfile, rename to replace the current strat
        cpath = f'{self.out_dir}/masks/boss.npz'
        Path(cpath_tmp).rename(cpath)
        # NOTE: Come back here and think about how the dimension change impacts loading the strat etc. This might also impact readfish
        # Example how to load these:
        # container = np.load(f'{cpath}.npz')
        # data = {key: container[key] for key in container}



    def _effect_increments(self, increments: defaultdict) -> None:
        """
        Loop through the increments of each contig and add the new coverage
        :param increments: defaultdict of lists of coverage counts per contig
        :return:
        """
        for cname, cont in self.contigs_filt.items():
            # grab the list of increments for this contig
            inc_list = increments[cname]
            cont.increment_coverage(increment_list=inc_list)



    def _update_scores_contigs(self) -> None:
        """
        Update the scores for each contig
        :return:
        """
        for cont in self.contigs_filt.values():
            # main score updating function
            cont.scores, cont.entropy = self.scoring.update_scores(contig=cont)
            # modify scores after updating
            cont.modify_scores()


    def _check_buckets_contigs(self) -> bool:
        """
        Check if the buckets within the contigs have enough coverage to be switched on
        :return: Boolean if any strategy has been switched on
        """
        for cont in self.contigs_filt.values():
            cont.check_buckets(threshold=self.args.optional.bucket_threshold)
        # check if any switches are on
        switched_on = [any(c.switched_on) for c in self.contigs.values()]
        return any(switched_on)


    def _update_benefits(self) -> None:
        """
        Update the expected benefit for all contigs
        :return:
        """
        for cont in self.contigs_filt.values():
            cont.calc_smu()
            cont.calc_u(approx_ccl=self.rl_dist.approx_ccl)



    def _distribute_strategy(self, strat: NDArray) -> None:
        """
        Place new decision strategies into the contigs strategies

        :param strat: Merged array of the updated strategy
        :return:
        """
        i = 0
        for cname, cont in self.contigs_filt.items():
            # get the bucket of each decision window by its start position,
            # the remainder of the contig belongs to the last bucket
            window_starts = np.arange(cont.n_windows) * self.args.optional.window_size
            bucket_idx = np.minimum(window_starts // cont.bucket_size, cont.bucket_switches.shape[0] - 1)
            buckets = cont.bucket_switches[bucket_idx]
            assert buckets.shape[0] == cont.strat.shape[0]
            # grab the new strategy
            cstrat = strat[i: i + cont.n_windows, :]
            assert cstrat.shape == cont.strat.shape
            # assign new strat
            if not self.args.general.barcodes:
                b = 0
                cont.strat[buckets[:, b], :, b] = cstrat[buckets[:, b], :, b]
            else:
                for bc_name in self.args.general.barcodes:
                    b = self.barcodes_index[int(bc_name.split('barcode')[1])]  # type: ignore
                    cont.strat[buckets[:, b], :, b] = cstrat[buckets[:, b], :, b]
            # log number of accepted sites
            f_perc = np.count_nonzero(cont.strat[:, 0]) / cont.strat.shape[0]
            r_perc = np.count_nonzero(cont.strat[:, 1]) / cont.strat.shape[0]
            logging.info(f'{cname}: {f_perc}, {r_perc}') # NOTE: Maybe think about whether this log is confusing because it can report more sites than exist with barcodes
            i += cont.n_windows




    def update_wrapper(self) -> None:
        """
        Second part of updates after a new batch of data.
        This is run after counting the coverage from new data

        :return:
        """
        # update the scores of all contigs
        self._update_scores_contigs()
        # flip strategy switches if threshold is reached
        switched_on = self._check_buckets_contigs()
        # UPDATE STRATEGY
        if switched_on:
            # update Fhat: unpack and normalise
            fhat_exp = self.read_starts.update_f_pointmass()
            fhat_exp = np.repeat(fhat_exp[:, :, np.newaxis], self.nbarcodes, axis=2)
            self._update_benefits()
            # merge the benefits into one array for combined calculation
            benefit, smu = self.scoring.merge_benefit(self.contigs_filt)
            # all arrays hold one row per decision window of the contigs
            target_size = sum(cont.n_windows for cont in self.contigs_filt.values())
            assert benefit.shape == smu.shape == fhat_exp.shape
            assert benefit.shape[0] == target_size
            # find the current decision strategy
            strat, _threshold = self.scoring.find_strat_thread(
                benefit=benefit,
                smu=smu,
                fhat=fhat_exp,
                time_cost=self.rl_dist.time_cost,
                window = self.args.optional.window_size
            )
            # distribute the strategy to the contigs
            self._distribute_strategy(strat=strat)
            # write strategies to file
            strat_dict = self.ref.get_strategy_dict()
            self._write_contig_strategies(contig_strats=strat_dict)



    def process_batch_runs(self, new_reads: dict[str, str], new_quals: dict[str, str]) -> None:
        """
        Process a batch of new data for the BOSS-RUNS mode
        This function is to be passed into the process_batch of the superclass

        :param new_reads: Dictionary of new sequences
        :param new_quals: Dictionary of qualities of new data
        :return:
        """
        # map the new reads to the reference
        # TODO: Read about paf_dict and see if I can add barcode information there or how else I should carry it through
        # Lukas: Barcode info should be in header of fastq file just like channel info, so could grab it in a similar way in _read_single_batch()
        paf_dict = self.mapper.map_sequences(sequences=new_reads)
        # convert coverage counts to increment arrays
        increments = self.cc.convert_records(paf_dict=paf_dict, seqs=new_reads, quals=new_quals)
        # effect the coverage increments for each contig
        self._effect_increments(increments=increments)
        # update the abundance tracker with new data
        self.tracker.update(n=len(new_reads), paf_dict=paf_dict)
        # update read starting position distribution
        self.read_starts.count_read_starts(paf_dict=paf_dict)
        # note: read length dist is updated in _get_new_data() of superclass
        self.update_wrapper()

