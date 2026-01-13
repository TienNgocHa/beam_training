import numpy as np
from scipy.spatial import distance_matrix
from tqdm import tqdm

from beam_finders.beam_finder import BeamFinder
from beamforming_data_generator.data_generator import P2MPBeamformingDataGenerator as bg
from utilities import natural_to_dB


def simulate_beam_finding(
    d_data,
    d_data_test=None,
    lf_beam_finders=[],
    num_train_ues=None,
    num_test_ues=None,
    max_num_pilot_symb=np.inf,
    dB_avg_snr_per_ant=None,
    test_loc_err_std=0,
    train_loc_err_std=0,
    num_mc=1,
):
    """
    This trains every BeamFinder produced by the entries of `lf_beam_finders`
    with the channel matrices  corresponding to the `num_ue` UEs in `d_data`.
    Then, it evaluates a collection of performance metrics for each BeamFinder
    using the channel matrices corresponding to the `num_ue` UEs in
    `d_data_test`. If `d_data_test` is `None`, then `d_data` is split by using
    `num_train_ues` and `num_test_ues` entries for training and testing. If
    `num_train_ues` is None, `num_test_ues` ue use for testing, and the rest for
    training.

    Args:
        `d_data` : dict with the following key-values:

            + 'bs_loc' : 3-length vector with the (x,y,z) coords of the
              transmitter.

            + 'ue_loc': `num_ue` x 3 matrix with the locations of the receivers.

            + 't_ch': `num_ue` x `num_ue_antennas` x `num_bs_antennas` tensor
              whose n-th slab is the channel matrix between the BS and the n-th
              ue.

        `d_data_test`: same as `d_data`. If `None`, then `d_data` is split
        by using `num_test_ues` entries for testing and the rest for training.
        If not `None`, then `num_test_ues` must be None, and `d_data` is used
        for training whereas `d_data_test` is used for testing.

        `lf_beam_finders`: list containing `num_beam_finders` functions. Each
        function returns an object of class BeamFinder.

        `test_loc_err_std`: If different from 0, noise of this std is added to the
        test locations that the UEs receive.

        `train_loc_err_std`: If different from 0, noise of this std is added to the
        train locations.

        `num_mc`: number of Monte Carlo runs over which the performance metrics
        are averaged.

    Returns:
        list of `num_beam_finders` dicts where the n-th dict has key-values:

        'beam_finder_name': name of the n-th BeamFinder.

        'v_beam_power_vs_meas_ind': vector of length `num_beams` where the m-th
        entry provides the average power of the beam corresponding to the m-th
        measurement collected by the n-th BeamFinder.

        'v_best_beam_power_so_far_vs_meas_ind': vector of length `num_beams`
        where the m-th entry provides the accumulated power of the beam
        corresponding to the m-th measurement collected by the n-th BeamFinder.

    """

    def get_noisy_amp_measurements(m_decoder, t_ch, num_pilot_symb, m_encoder=None):
        """
        See BeamFinder.get_amp_measurement.

        """
        return BeamFinder.get_amp_measurement(
            m_decoder,
            t_ch,
            m_encoder=m_encoder,
            dB_avg_snr_per_ant=dB_avg_snr_per_ant,
            num_pilot_symb=num_pilot_symb,
        )

    def get_beam_pow_seq(beam_finder: BeamFinder, bs_loc, ue_loc, m_ch, loc_err_std):
        """
        Returns:
            list of length `num_beams` whose n-th entry is the power of
            the beam corresponding to the n-th measurement.
        """

        def f_measure_beam(v_ue_beam, num_pilot_symb=np.inf, v_bs_beam=None):
            """Returns the amplitude gain of the `ind_beam`-th beam."""
            nonlocal total_num_pilot_symb

            noisy_meas_amp = get_noisy_amp_measurements(
                v_ue_beam[None, :],
                m_ch[None, :, :],
                m_encoder=None if v_bs_beam is None else v_bs_beam[None, :],
                num_pilot_symb=num_pilot_symb,
            )

            noiseless_meas_amp = BeamFinder.get_amp_measurement(
                v_ue_beam[None, :],
                m_ch[None, :, :],
                m_encoder=None if v_bs_beam is None else v_bs_beam[None, :],
                dB_avg_snr_per_ant=dB_avg_snr_per_ant,
            )

            if v_bs_beam is None:
                noisy_meas_amp = noisy_meas_amp[0, 0]
                noiseless_meas_amp = noiseless_meas_amp[0, 0]
            else:
                noisy_meas_amp = noisy_meas_amp[0, 0, 0]
                noiseless_meas_amp = noiseless_meas_amp[0, 0, 0]

            l_beam_siso_snr.append(np.abs(noiseless_meas_amp) ** 2)

            total_num_pilot_symb += num_pilot_symb
            return noisy_meas_amp

        l_beam_siso_snr = []
        total_num_pilot_symb = 0

        ue_loc_est = ue_loc + np.random.normal(0, loc_err_std, 3)
        beam_finder.set_metadata(bs_loc, ue_loc_est)
        v_best_beamformer_bs, v_best_beamformer_ue = beam_finder.find_good_beam_pair(
            f_measure_beam, max_num_pilot_symb=max_num_pilot_symb
        )

        if total_num_pilot_symb > max_num_pilot_symb:
            raise ValueError(
                f"BeamFinder {beam_finder} exceeded the maximum allowed number of pilot symbols."
            )

        best_beam_siso_snr = (
            np.abs(
                BeamFinder.get_amp_measurement(
                    v_best_beamformer_ue[None, :],
                    m_ch[None, :, :],
                    dB_avg_snr_per_ant=dB_avg_snr_per_ant,
                    m_encoder=v_best_beamformer_bs[None, :],
                )[0, 0, 0]
            )
            ** 2
        )

        return l_beam_siso_snr, total_num_pilot_symb, best_beam_siso_snr

    def get_beam_power_vs_meas_ind(m_beam_pow):
        return np.mean(m_beam_pow, axis=0)

    def get_best_beam_power_so_far_vs_meas_ind(m_beam_pow):
        num_beam = m_beam_pow.shape[1]
        m_best_power_so_far = np.array(
            [
                np.max(m_beam_pow[:, : ind_beam + 1], axis=1)
                for ind_beam in range(num_beam)
            ]
        ).T
        if len(m_best_power_so_far) == 0:
            return []
        return np.mean(m_best_power_so_far, axis=0)

    def concat_l_beam_pow(ll_beam_pow):
        min_len = np.iinfo(np.int32).max
        for l_beam_pow in ll_beam_pow:
            min_len = np.minimum(len(l_beam_pow), min_len)

        return np.array([l_beam_pow[0:min_len] for l_beam_pow in ll_beam_pow])

    def add_loc_err_to_dict(d_data, loc_err_std):
        d_data["ue_loc"] = d_data["ue_loc"] + np.random.normal(
            0, loc_err_std, d_data["ue_loc"].shape
        )
        return d_data

    def simulate_one_mc(
        d_data_train,
        d_data_test,
        max_num_pilot_symb,
        lf_beam_finders,
        loc_err_std,
        train_loc_err_std,
    ):
        ld_metrics = []
        num_ue = len(d_data_test["t_ch"])
        for f_beam_finder in lf_beam_finders:
            beam_finder = f_beam_finder()

            # Training
            beam_finder.train(
                add_loc_err_to_dict(d_data_train, train_loc_err_std),
                get_noisy_amp_measurements=get_noisy_amp_measurements,
                max_num_pilot_symb=max_num_pilot_symb,
                online_phase=True,
            )

            # Testing
            ll_beam_pow = []
            avg_total_num_pilot_symb = 0
            avg_best_beam_siso_dB_snr = 0
            for ind_ue in range(num_ue):
                l_beam_siso_snr, total_num_pilot_symb, best_beam_siso_snr = (
                    get_beam_pow_seq(
                        beam_finder,
                        bs_loc=d_data_test["bs_loc"],
                        ue_loc=d_data_test["ue_loc"][ind_ue],
                        m_ch=d_data_test["t_ch"][ind_ue],
                        loc_err_std=loc_err_std,
                    )
                )

                ll_beam_pow.append(l_beam_siso_snr)
                avg_total_num_pilot_symb += total_num_pilot_symb
                avg_best_beam_siso_dB_snr += natural_to_dB(best_beam_siso_snr)
            avg_total_num_pilot_symb = avg_total_num_pilot_symb / num_ue
            avg_best_beam_siso_dB_snr = avg_best_beam_siso_dB_snr / num_ue

            # The following is a `num_test_ue` x `num_beams` matrix whose (m,n)-th
            # entry contains the power of the n-th measured beam for the m-th test UE.
            m_beam_pow = concat_l_beam_pow(ll_beam_pow)
            # Calculate the minimum average distance between training users.
            m_dist = distance_matrix(d_data_train["ue_loc"], d_data_train["ue_loc"])
            min_avg_dist = np.mean(
                np.min(m_dist + np.eye(m_dist.shape[0]) * 1e6, axis=1)
            )

            ld_metrics.append(
                {
                    "beam_finder_name": beam_finder.__str__(),
                    "v_beam_power_vs_meas_ind": get_beam_power_vs_meas_ind(m_beam_pow),
                    "v_best_beam_power_so_far_vs_meas_ind": get_best_beam_power_so_far_vs_meas_ind(
                        m_beam_pow
                    ),
                    "avg_total_num_pilot_symb": avg_total_num_pilot_symb,
                    "avg_best_beam_siso_dB_snr": avg_best_beam_siso_dB_snr,
                    "min_avg_dist": min_avg_dist,
                }
            )

        return ld_metrics

    if d_data_test is not None:
        assert num_test_ues is None
        assert num_train_ues is None
        num_mc = (
            1  # If d_data_test is not None, then we are not doing Monte Carlo runs.
        )
        print("WARNING: num_mc is set to 1 because d_data_test is not None.")
        d_data_train = d_data
        d_data_test = d_data_test
        ld_metrics = simulate_one_mc(
            d_data_train,
            d_data_test,
            max_num_pilot_symb,
            lf_beam_finders,
            test_loc_err_std,
            train_loc_err_std,
        )

    else:
        assert num_test_ues is not None
        if num_train_ues is not None:
            assert num_train_ues + num_test_ues <= len(d_data["ue_loc"])

        def get_average_mc(lld_metrics_mc, key, beam_name):
            return np.mean(
                [
                    d[key]
                    for ld in lld_metrics_mc
                    for d in ld
                    if d["beam_finder_name"] == beam_name
                ],
                axis=0,
            )

        lld_metrics_mc = []
        for _ in tqdm(range(num_mc)):
            if num_train_ues is not None:
                # Shuffle d_data
                v_perm = np.random.permutation(len(d_data["ue_loc"]))
                d_data["ue_loc"] = d_data["ue_loc"][v_perm]
                d_data["t_ch"] = d_data["t_ch"][v_perm]
                d_data_mc = {
                    "bs_loc": d_data["bs_loc"],
                    "ue_loc": d_data["ue_loc"][: num_train_ues + num_test_ues],
                    "t_ch": d_data["t_ch"][: num_train_ues + num_test_ues],
                }
            else:
                d_data_mc = d_data

            d_data_train, d_data_test = bg.get_test_split(
                d_data_mc, num_train_ue=len(d_data_mc["ue_loc"]) - num_test_ues
            )

            ld_metrics = simulate_one_mc(
                d_data_train,
                d_data_test,
                max_num_pilot_symb,
                lf_beam_finders,
                test_loc_err_std,
                train_loc_err_std,
            )
            lld_metrics_mc.append(ld_metrics)
        # Average over the Monte Carlo runs
        ld_metrics = []
        for ind_beam_finder in range(len(lf_beam_finders)):
            beam_name = lf_beam_finders[ind_beam_finder]().__str__()
            ld_metrics.append(
                {
                    "beam_finder_name": beam_name,
                }
            )
            for key in lld_metrics_mc[0][0].keys():
                if key == "beam_finder_name":
                    continue
                ld_metrics[-1][key] = get_average_mc(lld_metrics_mc, key, beam_name)
    return ld_metrics


def simulate_best_beam_prediction(
    d_data_train, d_data_test, l_given_beam_inds, lf_beam_finders
):
    """
    Returns a dict with the accuracy of the prediction of the index of the best
    unobserved beam after observing the beams with indices in l_given_beam_inds.
    """

    def get_best_beam_ind_after_given_meas(beam_finder, bs_loc, ue_loc, m_ch):
        """
        Returns:
            list of length `num_beams` whose n-th entry is the power of
            the beam corresponding to the n-th measurement.
        """

        if m_ch.shape[1] != 1:
            raise NotImplementedError

        def f_measure_beam(v_beam):
            """Returns the amplitude gain of the `ind_beam`-th beam."""

            meas_amp = BeamFinder.get_amp_measurement(
                v_beam[None, :], m_ch[None, :, :]
            )[0, 0]

            return meas_amp

        beam_finder.set_metadata(bs_loc, ue_loc)
        return beam_finder.find_best_beem_after_given_noiseless_samples(
            f_measure_beam, l_given_beam_inds
        )

    def get_true_best_beam_ind(m_ch, m_decoder, l_not_given_beam_inds):
        meas_amp = BeamFinder.get_amp_measurement(m_decoder, m_ch[None, :, :])[0, :]
        pow_meas = np.abs(meas_amp) ** 2
        return l_not_given_beam_inds[np.argmax(pow_meas[l_not_given_beam_inds])]

    num_ue_ant = d_data_test["t_ch"][0].shape[0]

    ld_metrics = []
    for f_beam_finder in lf_beam_finders:
        beam_finder = f_beam_finder()
        m_decoder = beam_finder.get_m_decoder(num_ue_ant)
        num_beams = m_decoder.shape[0]
        l_not_given_beam_inds = list(set(np.arange(num_beams)) - set(l_given_beam_inds))

        # Training
        beam_finder.train(d_data_train)

        # Testing
        l_error = []
        for ind_ue in range(len(d_data_test["t_ch"])):
            predicted_best_beam_ind = get_best_beam_ind_after_given_meas(
                beam_finder,
                bs_loc=d_data_test["bs_loc"],
                ue_loc=d_data_test["ue_loc"][ind_ue],
                m_ch=d_data_test["t_ch"][ind_ue],
            )
            true_best_beam_ind = get_true_best_beam_ind(
                d_data_test["t_ch"][ind_ue], m_decoder, l_not_given_beam_inds
            )

            l_error.append(predicted_best_beam_ind != true_best_beam_ind)

        ld_metrics.append(
            {
                "beam_finder_name": beam_finder.__str__(),
                "accuracy_after_given_meas": 1 - np.mean(l_error),
            }
        )

    return ld_metrics


def add_margin_to_dict(ld_metrics, margin):
    """
    Adds number of beam pairs measured to reach a power within a given margin
    from the best beam pair.
    """
    for d in ld_metrics:
        power_db = 10 * np.log10(d["v_best_beam_power_so_far_vs_meas_ind"])
        num_best_beam_pair = np.sum(np.max(power_db) - power_db >= margin) + 1
        d["num_meas_beam_pairs_to_max_minus_margin"] = num_best_beam_pair
    return ld_metrics
