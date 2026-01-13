import os
import pickle

import numpy as np
import pandas as pd

d_metadata = {
    "ottawa_019": {
        "v_bs_loc": np.array([560.0966, 304.8544, 50]),
        "v_bottom_left_corner": np.array([51.2898, -53.3991, 0]),
        "num_hmat_files_to_read": 23424,
        "filename_ue_locs": "./data/Ottawa_NLoS_1BS_16yAnt_GridUE_4yAnt/ue_locs.txt",
        "filename_hmat": lambda ind_ue: f"./data/Ottawa_NLoS_1BS_16yAnt_GridUE_4yAnt/hmatrix/hmatrix_ue_{ind_ue + 1}.csv",
        "filename_pickle": "./data/Ottawa_NLoS_1BS_16yAnt_GridUE_4yAnt/beamforming_data.pickle",
        "discription": "This data was generated using   "
        "the city Ottawa with Wireless Insite"
        "ray-tracing software with 1 BS with 16 antennas and"
        "a grid of 23424 UEs with 4 antennas separated by 5m"
        "The antennas are along the y-axis separated by lambda/2."
        "The height of BS is 50m and UEs are 0m (NLoS)"
        "Frequency: 30 GHz, Bandwidth: 1 MHz",
    },
    "custom_01": {
        "v_bs_loc": np.array([730, 110, 50]),
        "v_bottom_left_corner": np.array([643, 28.69, 0]),
        "num_hmat_files_to_read": 14641,
        "filename_ue_locs": "./data/Custom_128x4/ue_locs.txt",
        "filename_hmat": lambda ind_ue: f"./data/Custom_128x4/hmatrix/hmatrix_ue_{ind_ue + 1}.csv",
        "filename_pickle": "./data/Custom_128x4/beamforming_data.pickle",
        "discription": "This data was generated using   "
        "ray-tracing software with 1 BS with 128 antennas along the y-axis separated by lambda/2 and"
        "a grid of 14641 UEs with 4 ULA antennas separated by 1m"
        "The height of BS is 50m and UEs are 0m (NLoS)"
        "Frequency: 30 GHz, Bandwidth: 1 MHz",
    },
}


class P2MPBeamformingDataGenerator:
    """
    Point to multipoint data generation.
    """

    def __init__(self, units="natural", dataset="ottawa_001"):
        """
        Args:
            - `dataset`

            - `units`: can be "natural" or "dB"
        """
        self.metadata = d_metadata[dataset]
        self.units = units

    def _adapt_units(self, t_ch):
        if self.units == "natural":
            pass
        elif self.units == "dB":
            t_ch = 20 * np.log10(np.abs(t_ch))
        else:
            raise ValueError

        return t_ch

    def get_bs_loc(self):
        """ "
        Returns:
            3-length vector with the (x,y,z) coords of the transmitter.
        """
        return self.metadata["v_bs_loc"]

    def get_ue_locs(self):
        """ "
        Returns:
            `num_ue x 3` matrix with the locations of the receivers.
        """
        ue_locs, _ = self._read_pickle()
        return ue_locs

    def get_ch_mats(self, num_bs_ant=None, num_ue_ant=None):
        """ "
        Returns:
            `num_ue x num_ue_antennas x num_bs_antennas` tensor whose n-th
            slab is the channel matrix between the BS and the n-th ue.

            If `num_bs_ant` is not None, it is set to the number of antennas
            in the dataset.

            If `num_ue_ant` is not None, it is set to the number of antennas
            in the dataset.

        """
        _, t_ch = self._read_pickle()
        if num_bs_ant is not None:
            assert num_bs_ant <= t_ch.shape[2]
            t_ch = t_ch[:, :, :num_bs_ant]
        if num_ue_ant is not None:
            assert num_ue_ant <= t_ch.shape[1]
            t_ch = t_ch[:, :num_ue_ant, :]
        return self._adapt_units(t_ch)

    @property
    def num_ue_ant(self):
        t_ch = self.get_ch_mats()
        return t_ch.shape[1]

    def get_dataset(self, num_ue=None, num_bs_ant=None, num_ue_ant=None):
        """
        Args:
            - `num_ue` if None it is set to the number of UEs in the dataset.

            - `num_bs_ant`: number of antennas at the BS. If None, it is set to
              the number of antennas in the dataset.

            - `num_ue_ant`: number of antennas at the UE. If None, it is set to
                the number of antennas in the dataset.

        Returns:
            dict with the following key-values:

            'bs_loc' : 3-length vector with the (x,y,z) coords of the transmitter.

            'ue_loc': `num_ue` x 3 matrix with the locations of the receivers.

            't_ch': `num_ue` x `num_ue_antennas` x `num_bs_antennas` tensor whose n-th
                slab is the channel matrix between the BS and the n-th ue.
        """

        t_ch = self.get_ch_mats(num_bs_ant=num_bs_ant, num_ue_ant=num_ue_ant)
        ue_locs = self.get_ue_locs()
        if num_ue is None:
            num_ue = len(ue_locs)

        v_ue_ind = np.random.choice(len(t_ch), num_ue, replace=False)

        return {
            "bs_loc": self.get_bs_loc(),
            "ue_loc": ue_locs[v_ue_ind],
            "t_ch": t_ch[v_ue_ind],
        }

    def get_sub_channel_matrix(self, t_ch, num_act_bs_ant=None, num_act_ue_ant=None):
        """Returns the subchannel matrix of the channel matrix `t_ch` with
        respect to num_bs_ant and num_ue_ant.

        Args:
            't_ch': (num_ue x num_ue_antennas x num_bs_antennas) tensor.

            'num_act_bs_ant': number of active antennas at the BS. If None, it
            is set to the number of antennas in the dataset.

            'num_act_ue_ant': number of active antennas at the UE. If None, it
            is set to the number of antennas in the dataset.

        Returns:
            (num_ue x num_act_ue_ant x num_act_bs_ant) tensor.
        """
        num_act_bs_ant = t_ch.shape[2] if num_act_bs_ant is None else num_act_bs_ant
        num_act_ue_ant = t_ch.shape[1] if num_act_ue_ant is None else num_act_ue_ant
        return t_ch[:, :num_act_ue_ant, :num_act_bs_ant]

    def nearest_ue_to_pt(self, v_pt):
        """Returns the index of the UE that is nearest the `v_pt`, where `v_pt`
        is a vector of length 3."""

        m_ue_locs = self.get_ue_locs()
        return np.argmin(np.sum((m_ue_locs - v_pt) ** 2, axis=1))

    @staticmethod
    def get_test_split(d_data, num_train_ue):
        # Shuffle the data
        v_perm = np.random.permutation(len(d_data["ue_loc"]))
        d_data["ue_loc"] = d_data["ue_loc"][v_perm]
        d_data["t_ch"] = d_data["t_ch"][v_perm]

        d_data_train = {
            "bs_loc": d_data["bs_loc"],
            "ue_loc": d_data["ue_loc"][:num_train_ue],
            "t_ch": d_data["t_ch"][:num_train_ue],
        }
        d_data_test = {
            "bs_loc": d_data["bs_loc"],
            "ue_loc": d_data["ue_loc"][num_train_ue:],
            "t_ch": d_data["t_ch"][num_train_ue:],
        }
        return d_data_train, d_data_test

    def _read_pickle(self):
        """ "
        Returns:
            -  `ue_locs`: `num_ue x 3` matrix with the locations of the receivers.
            -  `t_ch`: `num_ue x num_ue_antennas x num_bs_antennas` tensor.

        """
        filename_pickle = self.metadata["filename_pickle"]
        if not os.path.exists(filename_pickle):
            # Create pickle file
            print("Reading CSV...")
            ld_data = self._read_csv_and_txt()
            print("Saving pickle...")
            self._write_pickle(ld_data)

        else:
            # Load ld_data from pickle file
            with open(filename_pickle, "rb") as f:
                ld_data = pickle.load(f)

        ue_locs = np.array([d["ue_loc"] for d in ld_data])
        t_ch = np.array([d["m_ch_mat"] for d in ld_data])
        return ue_locs, t_ch

    def _read_csv_and_txt(self):
        """
        Returns:
            `ld_data`: a list of dictionary whose fields are
                    - `ue_loc`: [x, y, z] coordinate of a user equipment
                    - `m_ch_mat`: a num_ue_antennas x num_bs_antennas matrix
                                    that contains channel information
                                    between an user equipment and a base station.
        """

        ld_data = []
        # read user equipment locations (ue_locs) from a txt file

        df_ue_locs = pd.read_csv(
            self.metadata["filename_ue_locs"],
            skiprows=3,
            index_col=False,
            header=None,
            sep=" ",
        )

        m_ue_locs = df_ue_locs.drop(columns=0).to_numpy()

        for ind_ue in range(self.metadata["num_hmat_files_to_read"]):
            # create an empty dictionary for each ue
            d_ch_mat_and_ue_loc = {}

            df_hmat = pd.read_csv(
                self.metadata["filename_hmat"](ind_ue),
                skiprows=2,
                index_col=False,
                header=None,
            )

            m_hmat = df_hmat.drop(
                columns=0
            ).to_numpy()  # it is of shape num_ue_antennas x (num_bs_antennas + 2)

            # split m_hmatrix into real and imaginary parts
            m_hmat_real_part = m_hmat[:, ::2]
            m_hmat_imag_part = m_hmat[:, 1::2]

            m_chmat_complex = m_hmat_real_part + m_hmat_imag_part * 1j
            if np.linalg.norm(m_chmat_complex) == 0:
                continue

            d_ch_mat_and_ue_loc["ue_loc"] = m_ue_locs[ind_ue, :3]
            d_ch_mat_and_ue_loc["m_ch_mat"] = m_chmat_complex.T

            ld_data.append(d_ch_mat_and_ue_loc)

        return ld_data

    def _write_pickle(self, data):
        # save the data using pickle

        with open(self.metadata["filename_pickle"], "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
