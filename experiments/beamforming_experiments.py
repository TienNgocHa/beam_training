import random
from copy import deepcopy

import numpy as np

# import seaborn as sns
from tqdm import tqdm

import gsim
from beam_finders.beam_finder import (
    BIM,
    BeamFinder,
    ExhaustiveSearchBeamFinder,
    HierarchicalBeamFinder,
    LocAwareBeamFinder,
)
from beamforming_data_generator.data_generator import P2MPBeamformingDataGenerator
from gsim.gfigure import GFigure
from simulators.beam_finding_simulator import add_margin_to_dict, simulate_beam_finding


class ExperimentSet(gsim.AbstractExperimentSet):
    """
    Radio maps of the rx. power for a given beamformer.
    Figure 1 in the paper
    """

    def experiment_1001(args):
        generator = P2MPBeamformingDataGenerator(dataset="custom_01")

        m_ue_locs = generator.get_ue_locs()
        t_ch = generator.get_ch_mats()
        _, num_ue_ant, num_bs_ant = t_ch.shape

        # Beamformers for the BS
        m_encoder = BeamFinder.get_dft_beams(num_bs_ant)

        # Beamformers for the UEs
        m_decoder = BeamFinder.get_dft_beams(num_ue_ant)

        power_db = 10 * np.log10(
            np.abs(
                BeamFinder.get_amp_measurement(
                    m_decoder=m_decoder, m_encoder=m_encoder, t_ch=t_ch
                )
            )
            ** 2
        )

        # List of all pairs of beam indices
        l_beam_pairs = [
            (ind_ue_beam, ind_bs_beam)
            for ind_bs_beam in range(len(m_encoder))
            for ind_ue_beam in range(len(m_decoder))
        ]
        # Randomly select 'num_plots' pairs of beam indices
        num_plots = 4
        lm_ue_locs = [m_ue_locs] * num_plots
        l_beam_pairs = random.sample(l_beam_pairs, num_plots)
        l_power_db = []
        for ind_ue_beam, ind_bs_beam in l_beam_pairs:
            l_power_db.append(power_db[:, ind_ue_beam, ind_bs_beam])
        return plot_radio_map(lm_ue_locs=lm_ue_locs, lv_amp_gain=l_power_db)

    """
    Power of the best beam so far vs. number of measured beams.
    Figure 3 in the paper
    """

    def experiment_2001(args):
        # Data generation
        num_train_ue = 10000
        num_test_ue = 2000
        loc_err_std = 25.0
        dataset = "custom_01"  # For the case of multiple antennas, use 'ottawa_019'
        generator = P2MPBeamformingDataGenerator(dataset=dataset)
        d_data = generator.get_dataset(num_train_ue + num_test_ue, num_bs_ant=16)

        ld_metrics = simulate_beam_finding(
            d_data,
            num_test_ues=num_test_ue,
            num_mc=1,
            lf_beam_finders=[
                lambda: ExhaustiveSearchBeamFinder(),
                lambda: HierarchicalBeamFinder(wide_beam_type="Fourier"),
                lambda: BIM(num_neighbors=5),
                lambda: LocAwareBeamFinder(),  # MABEL
                lambda: LocAwareBeamFinder(test_loc_err_std=loc_err_std),  # LOREN
            ],
            test_loc_err_std=loc_err_std,
        )

        G_natural = GFigure(
            xlabel="Beam Index", ylabel="Best Measured Beam Power So Far [W]"
        )
        G_dB = GFigure(
            xlabel="Beam Index", ylabel="Best Measured Beam Power So Far [dB]"
        )
        for ind_beam_finder in range(len(ld_metrics)):
            G_natural.add_curve(
                xaxis=np.arange(
                    len(
                        ld_metrics[ind_beam_finder][
                            "v_best_beam_power_so_far_vs_meas_ind"
                        ]
                    )
                )
                + 1,
                yaxis=ld_metrics[ind_beam_finder][
                    "v_best_beam_power_so_far_vs_meas_ind"
                ],
                legend=ld_metrics[ind_beam_finder]["beam_finder_name"],
            )
            G_dB.add_curve(
                xaxis=np.arange(
                    len(
                        ld_metrics[ind_beam_finder][
                            "v_best_beam_power_so_far_vs_meas_ind"
                        ]
                    )
                )
                + 1,
                yaxis=10
                * np.log10(
                    ld_metrics[ind_beam_finder]["v_best_beam_power_so_far_vs_meas_ind"]
                ),
                legend=ld_metrics[ind_beam_finder]["beam_finder_name"],
            )
        return [G_natural, G_dB]

    """
    Number of beam pairs measured to reach a power within a given margin
    from the best beam pair. vs. different loc_err_std.
    Figure 4 in the paper
    """

    def experiment_2002(args):
        # Data generation
        num_train_ue = 10000
        num_test_ue = 2000
        l_loc_err_std = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
        margin = 1
        generator = P2MPBeamformingDataGenerator(dataset="custom_01")  # NLoS
        d_data = generator.get_dataset(num_train_ue + num_test_ue, num_bs_ant=16)
        lld_metrics = []
        for loc_err_std in tqdm(l_loc_err_std):
            ld_metrics = simulate_beam_finding(
                d_data,
                num_test_ues=num_test_ue,
                num_mc=1,
                lf_beam_finders=[
                    lambda: ExhaustiveSearchBeamFinder(),
                    lambda: HierarchicalBeamFinder(wide_beam_type="Fourier"),
                    lambda: LocAwareBeamFinder(test_loc_err_std=0.0),  # MABEL
                    lambda: LocAwareBeamFinder(test_loc_err_std=10.0),
                    lambda: LocAwareBeamFinder(test_loc_err_std=15.0),
                    lambda: LocAwareBeamFinder(test_loc_err_std=20.0),
                    lambda: LocAwareBeamFinder(test_loc_err_std=25.0),
                ],
                test_loc_err_std=loc_err_std,
            )
            ld_metrics = add_margin_to_dict(ld_metrics, margin)
            lld_metrics.append(ld_metrics)

        G = GFigure(
            xlabel="Location standard deviation error [m]",
            ylabel="Number of beam pairs",
        )
        for ind_beam_finder in range(len(lld_metrics[0])):
            G.add_curve(
                xaxis=l_loc_err_std,
                yaxis=[
                    ld[ind_beam_finder]["num_meas_beam_pairs_to_max_minus_margin"]
                    for ld in lld_metrics
                ],
                legend=lld_metrics[0][ind_beam_finder]["beam_finder_name"],
            )
        return G

    def get_nbpm_vs_training_point_or_dist(
        l_num_training_points, num_test_ue, loc_err_std, margin, dataset, num_mc
    ):
        """Number of beam pairs measured to reach a power within a given margin
        from the best beam pair. vs. num training points or average minimum
        distances of training points
        """
        # Data generation
        generator = P2MPBeamformingDataGenerator(dataset=dataset)
        l_num_training_points = l_num_training_points
        lld_metrics = []
        for num_training_points in tqdm(l_num_training_points):
            d_data = generator.get_dataset(
                num_training_points + num_test_ue, num_bs_ant=16
            )
            ld_metrics = simulate_beam_finding(
                d_data,
                num_test_ues=num_test_ue,
                num_mc=num_mc,
                lf_beam_finders=[
                    lambda: ExhaustiveSearchBeamFinder(),
                    lambda: HierarchicalBeamFinder(wide_beam_type="Fourier"),
                    lambda: LocAwareBeamFinder(test_loc_err_std=0.0),  # MABEL
                    lambda: LocAwareBeamFinder(test_loc_err_std=loc_err_std),
                ],
                test_loc_err_std=loc_err_std,
            )
            ld_metrics = add_margin_to_dict(ld_metrics, margin)
            lld_metrics.append(ld_metrics)
        return lld_metrics

    """
    Number of beam pairs measured to reach a power within a given margin
    from the best beam pair. vs. average minimum distances of training points
    Figure 5 in the paper
    """

    def experiment_2003(args):
        l_num_training_points = [
            100,
            200,
            300,
            500,
            700,
            1000,
            1500,
            2000,
            3000,
            5000,
            7000,
        ]
        lld_metrics = ExperimentSet.get_nbpm_vs_training_point_or_dist(
            l_num_training_points=l_num_training_points,
            num_test_ue=2000,
            loc_err_std=25.0,
            margin=1,
            dataset="custom_01",
            num_mc=1,
        )
        G = GFigure(
            xlabel="Average minimum distance of training points [m]",
            ylabel="Number of beam pairs",
        )
        for ind_beam_finder in range(len(lld_metrics[0])):
            l_dists = [ld[ind_beam_finder]["min_avg_dist"] for ld in lld_metrics]
            l_metrics = [
                ld[ind_beam_finder]["num_meas_beam_pairs_to_max_minus_margin"]
                for ld in lld_metrics
            ]
            # Sort the lists based on the distances
            l_dists, l_metrics = zip(*sorted(zip(l_dists, l_metrics)))
            G.add_curve(
                xaxis=list(l_dists),
                yaxis=list(l_metrics),
                legend=lld_metrics[0][ind_beam_finder]["beam_finder_name"],
            )
        return G

    """
    Number of beam pairs measured to reach a power within a given margin
    from the best beam pair. vs. num BS antennas
    Figure 6 in the paper
    """

    def experiment_2004(args):
        # Data generation
        num_train_ue = 10000
        num_test_ue = 2000
        loc_err_std = 25.0
        margin = 1
        generator = P2MPBeamformingDataGenerator(dataset="custom_01")  # NLoS
        d_data = generator.get_dataset(num_train_ue + num_test_ue)
        l_num_bs_ant = [1, 2, 4, 8, 16, 32, 64]
        lld_metrics = []
        for num_bs_ants in tqdm(l_num_bs_ant):
            d_data_temp = deepcopy(d_data)
            d_data_temp["t_ch"] = generator.get_sub_channel_matrix(
                d_data_temp["t_ch"], num_act_bs_ant=num_bs_ants
            )

            ld_metrics = simulate_beam_finding(
                d_data_temp,
                num_test_ues=num_test_ue,
                num_mc=1,
                lf_beam_finders=[
                    lambda: ExhaustiveSearchBeamFinder(),
                    lambda: HierarchicalBeamFinder(wide_beam_type="Fourier"),
                    lambda: LocAwareBeamFinder(),  # MABEL
                    lambda: LocAwareBeamFinder(test_loc_err_std=loc_err_std),  # LOREN
                ],
                test_loc_err_std=loc_err_std,
            )
            ld_metrics = add_margin_to_dict(ld_metrics, margin)
            lld_metrics.append(ld_metrics)

        G = GFigure(
            xlabel="Number of base station antennas", ylabel="Number of beam pairs"
        )
        for ind_beam_finder in range(len(lld_metrics[0])):
            G.add_curve(
                xaxis=l_num_bs_ant,
                yaxis=[
                    ld[ind_beam_finder]["num_meas_beam_pairs_to_max_minus_margin"]
                    for ld in lld_metrics
                ],
                legend=lld_metrics[0][ind_beam_finder]["beam_finder_name"],
            )
        return G

    """
    Power of the best beam so far vs. number of measured beams.
    Noise in the training data.
    Figure 7, 8 in the paper
    """

    def experiment_2005(args):
        # Data generation
        num_train_ues = 500
        num_test_ues = 40
        test_loc_err_std = 25.0
        train_loc_err_std = 25.0
        num_mc = 2
        # dataset = 'custom_01'  # For the case of multiple antennas, use 'ottawa_019'
        dataset = "ottawa_019"
        generator = P2MPBeamformingDataGenerator(dataset=dataset)
        d_data = generator.get_dataset(num_train_ues + num_test_ues, num_bs_ant=16)

        G_natural = GFigure(
            xlabel="Beam Index", ylabel="Best Measured Beam Power So Far [W]"
        )
        G_dB = GFigure(
            xlabel="Beam Index", ylabel="Best Measured Beam Power So Far [dB]"
        )

        if num_mc == 1:
            d_data_train, d_data_test = generator.get_test_split(
                d_data, num_train_ue=len(d_data["ue_loc"]) - num_test_ues
            )
            num_mc_in = None
            num_test_ues_in = None

        else:
            d_data_train = d_data
            d_data_test = None
            num_mc_in = num_mc
            num_test_ues_in = num_test_ues

        def plot_metrics_for_train_loc_err_std(
            train_loc_err_std, test_loc_err_std, lf_beam_finders
        ):
            ld_metrics = simulate_beam_finding(
                d_data_train,
                d_data_test,
                num_test_ues=num_test_ues_in,
                num_mc=num_mc_in,
                lf_beam_finders=lf_beam_finders,
                test_loc_err_std=test_loc_err_std,
                train_loc_err_std=train_loc_err_std,
            )

            for ind_beam_finder in range(len(ld_metrics)):
                G_natural.add_curve(
                    xaxis=np.arange(
                        len(
                            ld_metrics[ind_beam_finder][
                                "v_best_beam_power_so_far_vs_meas_ind"
                            ]
                        )
                    )
                    + 1,
                    yaxis=ld_metrics[ind_beam_finder][
                        "v_best_beam_power_so_far_vs_meas_ind"
                    ],
                    legend=ld_metrics[ind_beam_finder]["beam_finder_name"]
                    + r", $\sigma_{\mathrm{train}}=$"
                    + f"{train_loc_err_std}"
                    + r", $\sigma_{\mathrm{test}}=$"
                    + f"{test_loc_err_std}",
                )
                G_dB.add_curve(
                    xaxis=np.arange(
                        len(
                            ld_metrics[ind_beam_finder][
                                "v_best_beam_power_so_far_vs_meas_ind"
                            ]
                        )
                    )
                    + 1,
                    yaxis=10
                    * np.log10(
                        ld_metrics[ind_beam_finder][
                            "v_best_beam_power_so_far_vs_meas_ind"
                        ]
                    ),
                    legend=ld_metrics[ind_beam_finder]["beam_finder_name"]
                    + r", $\sigma_{\mathrm{train}}=$"
                    + f"{train_loc_err_std}"
                    + r", $\sigma_{\mathrm{test}}=$"
                    + f"{test_loc_err_std}",
                )

        plot_metrics_for_train_loc_err_std(
            train_loc_err_std=0,
            test_loc_err_std=0,
            lf_beam_finders=[
                lambda: ExhaustiveSearchBeamFinder(),
                lambda: HierarchicalBeamFinder(wide_beam_type="Fourier"),
                lambda: LocAwareBeamFinder(),
            ],
        )
        plot_metrics_for_train_loc_err_std(
            train_loc_err_std=train_loc_err_std,
            test_loc_err_std=0,
            lf_beam_finders=[
                lambda: LocAwareBeamFinder(train_loc_err_std=0, test_loc_err_std=0),
                lambda: LocAwareBeamFinder(
                    train_loc_err_std=train_loc_err_std, test_loc_err_std=0
                ),
                # lambda: LocAwareBeamFinder(test_loc_err_std=
                #                            test_loc_err_std)
            ],
        )
        plot_metrics_for_train_loc_err_std(
            train_loc_err_std=train_loc_err_std,
            test_loc_err_std=test_loc_err_std,
            lf_beam_finders=[
                lambda: LocAwareBeamFinder(train_loc_err_std=0, test_loc_err_std=0),
                lambda: LocAwareBeamFinder(
                    train_loc_err_std=train_loc_err_std, test_loc_err_std=0
                ),
                lambda: LocAwareBeamFinder(
                    train_loc_err_std=0, test_loc_err_std=test_loc_err_std
                ),
                lambda: LocAwareBeamFinder(
                    train_loc_err_std=train_loc_err_std,
                    test_loc_err_std=test_loc_err_std,
                ),
            ],
        )

        return [G_natural, G_dB]


def plot_radio_map(lm_ue_locs, lv_amp_gain, num_subplot_columns=2):
    G = GFigure(
        num_subplot_columns=num_subplot_columns,
        global_color_bar=True,
        global_color_bar_position=[0.92, 0.25, 0.02, 0.5],
    )
    for ind, (m_ue_locs, v_amp_gain) in enumerate(zip(lm_ue_locs, lv_amp_gain)):
        v_x = np.sort(list(set(m_ue_locs[:, 0])))
        v_y = np.flip(np.sort(list(set(m_ue_locs[:, 1]))))

        m_z = np.tile(np.nan, (len(v_y), len(v_x)))

        for v_ue_loc, amp_gain in zip(m_ue_locs, v_amp_gain):
            ind_x = np.where(v_x == v_ue_loc[0])[0]
            ind_y = np.where(v_y == v_ue_loc[1])[0]
            m_z[ind_y, ind_x] = amp_gain
        G.next_subplot(
            xaxis=v_x,
            yaxis=v_y,
            zaxis=m_z,
            mode="imshow",
            ylabel="y [m]",
            xlabel="x [m]" if ind == len(lm_ue_locs) - 1 else "",
        )
    return G
