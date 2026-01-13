import itertools
import random

import numpy as np
from scipy.spatial import distance_matrix

from utilities import dB_to_natural

# tf.config.run_functions_eagerly(True)


class BeamFinder:
    def __init__(self, path_loss_compensation="none", inhibit_training=False):
        """
        `path_loss_compensation` can be 'none', 'free_space',
        'linear_regression'

        `inhibit_training`: If True, calls to `self.train()` are ignored. This
        is useful when a pre-trained BeamFinder is to be tested.
        """
        self.path_loss_compensation = path_loss_compensation
        self.inhibit_training = inhibit_training
        self.m_decoder = None

    @staticmethod
    def get_dft_beams(num_beams, num_ant=None, offset=0):
        """
            - `num_ant`: It must be greater than or equal to
                `num_beams`. If None, then num_ant is set to num_beams.

            - `num_beams` x `num_ant` matrix whose i-th row is the i-th
                beam of a DFT beam dictionary of `num_beams` elements
                padded with num_ant-num_beams zeros.

        Returns:

            - `num_beams` x `num_ant` matrix whose i-th row is the i-th
            beam of a DFT beam dictionary of `num_beams` elements
            padded with num_ant-num_beams zeros.
        """
        num_ant = num_beams if num_ant is None else num_ant
        assert num_ant >= num_beams
        v_inds = np.arange(num_beams)[:, None]
        m_beams = (1 / np.sqrt(num_beams)) * np.exp(
            1j * ((2 * np.pi / num_beams) * v_inds + offset) @ v_inds.T
        )

        return np.concatenate(
            [m_beams, np.zeros((num_beams, num_ant - num_beams))], axis=1
        )

    def get_m_decoder(self, num_ue_ant):
        """This method may be overridden to implement other choices of the
        combiner matrix."""

        return self.get_dft_beams(num_ue_ant)

    def get_m_encoder(self, num_bs_ant):
        """This method may be overridden to implement other choices of the
        combiner matrix."""
        return self.get_dft_beams(num_bs_ant)

    def train(self, d_data_train, **kwargs):
        self.m_decoder = self.get_m_decoder(num_ue_ant=d_data_train["t_ch"].shape[1])

        self.m_encoder = self.get_m_encoder(num_bs_ant=d_data_train["t_ch"].shape[2])

        if self.inhibit_training:
            return None

        return self._train(d_data_train, **kwargs)

    def _train(self, d_data_train, **kwargs):
        """To be overridden by subclasses."""
        pass

    def find_good_beam_pair(self, f_measure_beam, max_num_pilot_symb=np.inf):
        """
        This function invokes `f_measure_beam` for a certain set of beams chosen
        by the subclass and returns a hopefully good beamformer.

        """

        def return_best_beam_pair(m_meas_amp):
            # 1. Replace NaNs with the minimum value of the matrix.
            m_meas_amp_changed_nan = np.where(
                np.isnan(m_meas_amp), np.nanmin(np.abs(m_meas_amp)), m_meas_amp
            )
            # 2. Retrieve the indices of the best beam pair.
            ind_bs_beam, ind_ue_beam = np.unravel_index(
                np.argmax(np.abs(m_meas_amp_changed_nan)), m_meas_amp.shape
            )
            return self.m_encoder[ind_bs_beam], self.m_decoder[ind_ue_beam]

        num_ue_beams = len(self.m_decoder)
        num_bs_beams = len(self.m_encoder)
        m_meas_amp = np.tile(np.nan, (num_bs_beams, num_ue_beams)).astype(np.complex64)

        num_remaining_samples = max_num_pilot_symb
        for _ in range(num_bs_beams * num_ue_beams):
            ind_ue_beam, ind_bs_beam, num_samples = self._choose_next_beam_pair_inds(
                m_meas_amp, num_remaining_samples
            )
            if (ind_ue_beam is None) or (ind_bs_beam is None):
                break
            if num_samples > num_remaining_samples:
                raise ValueError

            num_remaining_samples = (
                num_remaining_samples - num_samples
                if num_samples < np.inf
                else num_remaining_samples
            )

            m_meas_amp[ind_bs_beam, ind_ue_beam] = f_measure_beam(
                v_ue_beam=self.m_decoder[ind_ue_beam],
                v_bs_beam=self.m_encoder[ind_bs_beam],
                num_pilot_symb=num_samples,
            )

        return return_best_beam_pair(m_meas_amp)

    def find_best_beam_after_given_noiseless_samples(
        self, f_measure_beam, l_given_beam_inds
    ):
        """
        Returns the index of the best beam after measuring the beams whose
        indices are in `v_given_beam_inds`.
        """

        num_beams = len(self.m_decoder)
        v_meas_amp = np.tile(np.nan, (num_beams,)).astype(np.complex64)

        for ind_beam in l_given_beam_inds:
            v_meas_amp[ind_beam] = self.compensate_path_loss(
                f_measure_beam(self.m_decoder[ind_beam])
            )

        return self._choose_next_ue_beam(v_meas_amp)

    def _choose_next_beam_pair_inds(self, m_meas_amp, num_remaining_samples):
        """
        The default implementation is to first choose the BS beam and then the
        UE. Override this method if this is not the desired behavior.
        """
        ind_bs_beam = self._choose_next_bs_beam(m_meas_amp)
        ind_ue_beam, num_samples = self._choose_next_ue_beam(
            m_meas_amp[ind_bs_beam, ...], num_remaining_samples
        )
        return ind_ue_beam, ind_bs_beam, num_samples

    def _choose_next_ue_beam(self, v_measurements, num_remaining_samples):
        """
        Returns:

            - index of the next beam to measure

            - how many samples must be taken from that beam.
        """

        raise NotImplementedError("This method must be overridden by subclasses.")

    def _choose_next_bs_beam(self, m_measurements):
        """
        Returns:

            - index of the next beam to measure
        """
        v_ind_bs_not_measured = np.where(np.isnan(m_measurements).any(axis=1))[0]
        return np.random.choice(v_ind_bs_not_measured)

    def _predict_best_beam(self, v_meas_amp):
        """
        Returns:

            - (num_ue_ant,) vector with the complex coefficients of a beamformer.
        """
        v_meas_indicator = ~np.isnan(v_meas_amp)
        v_meas_inds = np.where(v_meas_indicator)[0]
        best_ind_rel_meas = np.argmax(np.abs(v_meas_amp[v_meas_indicator]))
        best_ind = v_meas_inds[best_ind_rel_meas]

        return self.m_decoder[best_ind]

    def set_metadata(self, bs_loc, ue_loc):
        self.bs_loc = bs_loc
        self.ue_loc = ue_loc

    def __str__(self):
        return "[Override __str__]"

    @staticmethod
    def get_amp_measurement(
        m_decoder, t_ch, m_encoder=None, dB_avg_snr_per_ant=None, num_pilot_symb=np.inf
    ):
        """
        Args:

            - `m_decoder` is num_ue_beams x num_ue_antennas

            - `t_ch` is num_ues x num_ue_antennas x num_bs_antennas

            - `m_encoder` is num_bs_beams x num_bs_antennas. If None, it is
                assumed that m_encoder = [[1]].

            - `dB_avg_snr_per_ant`: average SNR per antenna. If equal to
                None, then the noise power at each antenna is 1 power unit. In
                this case, the output equals an estimate of the rx. power
                (recall that power = SNR if noise power = 1)

        Returns:

            - meas_amp:

                - If m_encoder is None: meas_amp is a num_ues x num_ue_beams
                matrix whose [m,n]-th entry is the complex amplitude
                received by the m-th UE with the n-th beam normalized such
                that its squared magnitude is an estimator of the SISO SNR
                in natural units. For num_pilot_symb = np.inf, the squared
                magnitude is exactly the SISO SNR.

                - If m_encoder is not None: meas_amp is a a num_ues x
                    num_ue_beams x num_bs_beams tensor
                whose [i,m,n]-th entry is the complex amplitude received by
                the i-th UE with the m-th UE beam when the BS uses the n-th
                BS beam normalized such that its squared magnitude is an
                estimator of the SISO SNR in natural units. For
                num_pilot_symb = np.inf, the squared magnitude is exactly
                the SISO SNR.
        """

        _m_encoder = np.array([[1]]) if m_encoder is None else m_encoder

        m_norm_decoder = m_decoder / np.linalg.norm(m_decoder, axis=1)[:, None]
        m_norm_encoder = _m_encoder / np.linalg.norm(_m_encoder, axis=1)[:, None]

        meas_amp = (
            np.conj(m_norm_decoder)[None, ...] @ t_ch @ m_norm_encoder.T[None, ...]
        )

        if dB_avg_snr_per_ant is not None:
            avg_snr_per_ant = dB_to_natural(dB_avg_snr_per_ant)
            num_ue_ant = t_ch.shape[1]
            m_noise_std = np.linalg.norm(t_ch, axis=1) / np.sqrt(
                num_ue_ant * avg_snr_per_ant
            )
            meas_amp = meas_amp / m_noise_std

        if num_pilot_symb < np.inf:
            num_ues, num_beams = meas_amp.shape
            meas_amp += (
                np.random.normal(size=(num_ues, num_beams))
                + 1j * np.random.normal(size=(num_ues, num_beams))
            ) / np.sqrt(2 * num_pilot_symb)

        # At this point, meas_amp should be num_ues x num_ue_beams x num_bs_beams
        if m_encoder is None:
            return meas_amp[:, :, 0]
        return meas_amp


class ExhaustiveSearchBeamFinder(BeamFinder):
    def __str__(self):
        return "ESBF"

    def _choose_next_ue_beam(self, v_measurements, num_remaining_samples):
        v_ind_not_measured = np.where(np.isnan(v_measurements))[0]
        num_remaining_beams = len(v_ind_not_measured)
        if num_remaining_beams == 0:
            return None
        num_samples = np.floor(num_remaining_samples / num_remaining_beams)

        return np.random.choice(v_ind_not_measured), num_samples


class HierarchicalBeamFinder(BeamFinder):
    def __init__(
        self,
        *args,
        wide_beam_type="Fourier",
        num_ue_levels=None,
        num_bs_levels=None,
        **kwargs,
    ):
        """
        Args:
        - `num_levels`: If it is None, by default it equals log2(num_ue_ant).
        """
        super().__init__(*args, **kwargs)
        self.wide_beam_type = wide_beam_type
        self.num_ue_levels = num_ue_levels
        self.num_bs_levels = num_bs_levels

    def get_m_beams(self, num_ants, num_levels=None):
        """
        Args:
            'num_ants': number of antennas,

            'num_levels': number of levels in the hierarchical beamformer. If
            None, then it is set to log2(num_ants).

        Returns:
            'm_beams': (sum of num_beams for all level, num_ants) matrix of beams

            'v_num_beams': (num_levels,) vector with the number of beams at each
            level.

            'num_levels': number of levels in the hierarchical beamformer.
        """
        if num_ants == 1:
            return self.get_dft_beams(num_ants), [1], 0
        if num_levels is None:
            num_levels = np.log2(num_ants)
            assert num_levels % 1 == 0, "`num_ant` must be a power of 2."
            num_levels = int(num_levels)

        v_num_beams = 2 ** np.arange(1, num_levels + 1)
        if self.wide_beam_type == "Fourier":
            l_levels = [
                self.get_dft_beams(
                    num_beams,
                    num_ants,
                    offset=np.pi / num_beams
                    # The following is to have 0 offset at the last level -->
                    # ensure fair comparison with the other beamfinders
                    - np.pi / v_num_beams[-1],
                )
                / np.sqrt(num_beams)
                for num_beams in v_num_beams
            ]

            m_beams = np.concatenate(l_levels, axis=0)
        else:
            raise ValueError
        return m_beams, v_num_beams, num_levels

    def get_m_decoder(self, num_ue_ant):
        m_ue_beams, v_num_ue_beams, num_levels = self.get_m_beams(
            num_ue_ant, self.num_ue_levels
        )
        self.v_num_ue_beams = v_num_ue_beams
        self.num_ue_levels = num_levels
        return m_ue_beams

    def get_m_encoder(self, num_bs_ant):
        m_bs_beams, v_num_bs_beams, num_levels = self.get_m_beams(
            num_bs_ant, self.num_bs_levels
        )
        self.v_num_bs_beams = v_num_bs_beams
        self.num_bs_levels = num_levels
        return m_bs_beams

    def _choose_next_beam_pair_inds(self, m_meas_amp, num_remaining_samples):
        def get_v_num_meas(v_num_beams):
            if (len(v_num_beams) == 1) and (v_num_beams[0] == 1):
                return np.array([], dtype=np.int32)
            elif (len(v_num_beams)) == 1 and (v_num_beams[0] > 1):
                return v_num_beams
            v_num_narrower_beams = v_num_beams[1:] // v_num_beams[:-1]
            v_num_meas = np.concatenate(
                [[v_num_beams[0]], v_num_narrower_beams], axis=0
            )
            return v_num_meas

        m_meas_pow = (np.abs(m_meas_amp)) ** 2
        num_meas_beams = np.sum(~np.isnan(m_meas_pow))

        v_num_meas_ue = get_v_num_meas(self.v_num_ue_beams)
        v_num_meas_bs = get_v_num_meas(self.v_num_bs_beams)

        # extent v_num_meas_ue and v_num_meas_bs to have the same length
        if self.num_ue_levels > self.num_bs_levels:
            v_num_meas_bs = np.concatenate(
                [v_num_meas_bs, [1] * (self.num_ue_levels - self.num_bs_levels)], axis=0
            )
        elif self.num_bs_levels > self.num_ue_levels:
            v_num_meas_ue = np.concatenate(
                [v_num_meas_ue, [1] * (self.num_bs_levels - self.num_ue_levels)], axis=0
            )
        v_cum_num_meas = np.concatenate(
            [[0], np.cumsum(v_num_meas_bs * v_num_meas_ue)], axis=0
        )
        in_level = next(
            (i for i, j in enumerate(num_meas_beams < v_cum_num_meas) if j), None
        )

        if in_level is None:
            return None, None, num_remaining_samples
        elif in_level == 1:
            if self.num_ue_levels == 0 and self.num_bs_levels == 0:
                ind_next_ue_beam = 0
                ind_next_bs_beam = 0
            elif self.num_ue_levels == 0 and self.num_bs_levels > 0:
                v_meas_pow_first_level = m_meas_pow[0:2, 0]
                ind_next_bs_beam = random.choice(
                    np.argwhere(np.isnan(v_meas_pow_first_level))
                )
                ind_next_ue_beam = 0
                ind_next_bs_beam = ind_next_bs_beam[0]
            elif self.num_bs_levels == 0 and self.num_ue_levels > 0:
                v_meas_pow_first_level = m_meas_pow[0, 0:2]
                ind_next_bs_beam = 0
                ind_next_ue_beam = random.choice(
                    np.argwhere(np.isnan(v_meas_pow_first_level))
                )
                ind_next_ue_beam = ind_next_ue_beam[0]
            else:
                m_meas_pow_first_level = m_meas_pow[0:2, 0:2]
                ind_next_beams = random.choice(
                    np.argwhere(np.isnan(m_meas_pow_first_level))
                )
                ind_next_bs_beam = ind_next_beams[0]
                ind_next_ue_beam = ind_next_beams[1]
        else:
            m_observed_beam_pairs = np.argwhere(~np.isnan(m_meas_pow))
            l_observed_beam_pairs = [
                (pair[0], pair[1]) for pair in m_observed_beam_pairs
            ]
            l_observed_beam_pairs.sort(key=lambda x: x[0] ** 2 + x[1] ** 2)
            l_previous_values = [
                (m_meas_pow[pair[0], pair[1]], pair[0], pair[1])
                for pair in l_observed_beam_pairs[
                    v_cum_num_meas[in_level - 2] : v_cum_num_meas[in_level - 1]
                ]
            ]
            l_previous_values.sort(reverse=True, key=lambda x: x[0])
            ind_previous_bs_beam = l_previous_values[0][1]
            ind_previous_ue_beam = l_previous_values[0][2]
            if in_level > self.num_bs_levels:
                v_next_bs_beams = np.array([ind_previous_bs_beam])
            else:
                v_next_bs_beams = np.arange(
                    2 * ind_previous_bs_beam + v_num_meas_bs[in_level - 2],
                    2 * ind_previous_bs_beam
                    + v_num_meas_bs[in_level - 2]
                    + v_num_meas_bs[in_level - 1],
                )
            if in_level > self.num_ue_levels:
                v_next_ue_beams = np.array([ind_previous_ue_beam])
            else:
                v_next_ue_beams = np.arange(
                    2 * ind_previous_ue_beam + v_num_meas_ue[in_level - 2],
                    2 * ind_previous_ue_beam
                    + v_num_meas_ue[in_level - 2]
                    + v_num_meas_ue[in_level - 1],
                )
            l_next_level_beams = list(
                itertools.product(v_next_bs_beams, v_next_ue_beams)
            )
            l_next_beam_pairs = [
                pair for pair in l_next_level_beams if pair not in l_observed_beam_pairs
            ]
            ind_next_beam_pair = random.choice(l_next_beam_pairs)
            ind_next_bs_beam = ind_next_beam_pair[0]
            ind_next_ue_beam = ind_next_beam_pair[1]
        return ind_next_ue_beam, ind_next_bs_beam, num_remaining_samples

    def _predict_best_beam(self, v_meas_amp):
        v_meas_pow = (np.abs(v_meas_amp)) ** 2
        v_ind_obs = np.where(~np.isnan(v_meas_pow))[0]
        ind_best_beam = v_ind_obs[np.argmax(v_meas_pow[v_ind_obs])]
        return self.m_decoder[ind_best_beam]

    def __str__(self):
        return "HBF"


class LocAwareBeamFinder(BeamFinder):
    def __init__(self, *args, test_loc_err_std=0, train_loc_err_std=0, **kwargs):
        """
        If `loc_err_std==0`, the nearest training point is chosen to obtain
        self.l_beam_inds.

        Else, self.l_beam_inds is obtained from the beam power estimates that
        result from weighted averages of the power of each beam at all training
        locations.
        """
        super().__init__(*args, **kwargs)
        self.test_loc_err_std = test_loc_err_std
        self.train_loc_err_std = train_loc_err_std

    def _train(self, d_data_train, **kwargs):
        self.d_data_train = d_data_train

        t_amp = BeamFinder.get_amp_measurement(
            m_decoder=self.m_decoder,
            m_encoder=self.m_encoder,
            t_ch=d_data_train["t_ch"],
        )
        self.t_pow = np.abs(t_amp) ** 2  # num_ue x num_ue_beams x num_bs_beams

        if self.train_loc_err_std:
            num_ue, num_ue_beams, num_bs_beams = self.t_pow.shape

            m_train_ue_loc = self.d_data_train["ue_loc"]
            m_dist = distance_matrix(m_train_ue_loc, m_train_ue_loc)
            m_prob_training_loc = np.exp(
                -(m_dist**2) / (2 * self.train_loc_err_std**2)
            ) / (
                np.sqrt(2 * np.pi * self.train_loc_err_std**2)
                ** m_train_ue_loc.shape[1]
            )
            m_weights = (
                m_prob_training_loc / np.sum(m_prob_training_loc, axis=1)[:, None]
            )
            m_pow = np.reshape(self.t_pow, [num_ue, -1]).T
            m_weighted_pow = m_pow @ m_weights  # num_ue_beams*num_bs_beams x num_ue
            self.t_pow = np.reshape(
                m_weighted_pow.T, [num_ue, num_ue_beams, num_bs_beams]
            )

    def set_metadata(self, bs_loc, ue_loc):
        def get_beam_ind_list(t_pow, m_train_ue_loc, ue_loc, loc_err_std):
            """
            Args:
                't_pow': num_ue x num_ue_beams x num_bs_beams with the beam
                powers in the training set.

                'm_train_ue_loc': num_ue x 3

                'ue_loc': location of the UE

                'loc_err_std': standard deviation of the location error. If
                `loc_err_std==0`, the nearest training point is chosen to obtain
                self.l_beam_inds. Else, self.l_beam_inds is obtained from the
                beam power estimates that result from weighted averages of the
                power of each beam at all training locations.

            Returns:
                list of beam indices
            """
            v_dist = np.linalg.norm(m_train_ue_loc - ue_loc, axis=-1)
            if loc_err_std == 0:
                # Find the nearest training point to the UE location
                ind_nearest = np.argmin(v_dist)
                m_weighted_pow = t_pow[ind_nearest]
            else:
                v_prob_training_loc = np.exp(-(v_dist**2) / (2 * loc_err_std**2)) / (
                    np.sqrt(2 * np.pi * loc_err_std**2) ** len(ue_loc)
                )
                m_weighted_pow = (
                    (t_pow.T @ v_prob_training_loc) / np.sum(v_prob_training_loc)
                ).T
            # sorted power according to the power of the beams
            flattened_m_weighted_pow = [
                (m_weighted_pow[ue_beam_ind, bs_beam_ind], ue_beam_ind, bs_beam_ind)
                for ue_beam_ind in range(m_weighted_pow.shape[0])
                for bs_beam_ind in range(m_weighted_pow.shape[1])
            ]
            flattened_m_weighted_pow.sort(reverse=True, key=lambda x: x[0])
            # list of beam index pairs (ue_beam_ind, bs_beam_ind)
            l_sorted_inds = [(idx[1], idx[2]) for idx in flattened_m_weighted_pow]
            return l_sorted_inds

        super().set_metadata(bs_loc, ue_loc)

        self.l_beam_pair_inds = get_beam_ind_list(
            self.t_pow,
            self.d_data_train["ue_loc"],
            ue_loc,
            self.test_loc_err_std,
        )

    def _choose_next_beam_pair_inds(self, m_measurements, num_remaining_samples):
        """
        Returns:
            - index of the next beam pair to measure
        """
        if num_remaining_samples < np.inf:
            raise NotImplementedError
        return self.l_beam_pair_inds.pop(0) + (num_remaining_samples,)

    def __str__(self):
        return (
            r"LABEF, $\hat \sigma_{\mathrm{{train}}}$"
            + f"={self.train_loc_err_std}"
            + r", $\hat \sigma_{\mathrm{test}}$"
            + f"={self.test_loc_err_std}"
        )


class BIM(BeamFinder):
    def __init__(self, *args, num_neighbors=1, **kwargs):
        """
        Args:
            'num_neighbors': Number of neighbors
        """
        super().__init__(*args, **kwargs)
        self.num_neighbors = num_neighbors

    def _train(self, d_data_train, **kwargs):
        self.d_data_train = d_data_train

        t_amp = BeamFinder.get_amp_measurement(
            m_decoder=self.m_decoder,
            m_encoder=self.m_encoder,
            t_ch=d_data_train["t_ch"],
        )
        t_pow = np.abs(t_amp) ** 2  # num_ue x num_ue_beams x num_bs_beams
        # Find the best pair of beams for each training point
        self.l_beam_pair_inds_train = [
            np.unravel_index(t_pow[ind].argmax(), t_pow[0].shape)
            for ind in range(t_pow.shape[0])
        ]

    def set_metadata(self, bs_loc, ue_loc):
        super().set_metadata(bs_loc, ue_loc)
        v_dist = np.linalg.norm(self.d_data_train["ue_loc"] - ue_loc, axis=-1)
        # Find self.num_neighbors nearest training points to the UE location
        indices = np.argpartition(v_dist, self.num_neighbors)[: self.num_neighbors]
        # Find the best pair of beams for each training point
        self.l_beam_pair_inds = [self.l_beam_pair_inds_train[ind] for ind in indices]

    def _choose_next_beam_pair_inds(self, m_measurements, num_remaining_samples):
        """
        Returns:
            - index of the next beam pair to measure
        """
        if num_remaining_samples < np.inf:
            raise NotImplementedError
        if len(self.l_beam_pair_inds) == 0:
            return None, None, None
        return self.l_beam_pair_inds.pop(0) + (num_remaining_samples,)

    def __str__(self):
        return f"BIM, k={self.num_neighbors}"
