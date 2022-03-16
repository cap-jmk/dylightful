from tqdm import tqdm
import numpy as np
from hmmlearn.hmm import GaussianHMM
import parser
from plot_hmm import plot_state_diagram, plot_transmat_map
from metrics import calculate_mean_probas
import pandas as pd


class HMM_Analyzer:
    """
    Class to analyze a Dynophore trajectory with an HMM
    """

    def __init__(
        self, path_traj, model=GaussianHMM, num_states=15, n_iter=10000, save_path=None
    ):

        self.num_states = num_states
        self.n_iter = n_iter
        self.path_traj = path_traj

        if save_path is None:
            self.save_path = parser.get_dir(self.path_traj)
        else:
            self.save_path = save_path
        self.file_name = parser.get_name(self.path_traj)
        self.model = model
        self.time_ser = self.load_traj(self.save_path)

    def find_states(
        self,
        time_ser=None,
        num_states=None,
        n_iter=None,
    ):
        """Computes the probability for finding the states

        Args:
            num_states ([type]): [description]
            n_iter ([type]): [description]
        """
        if time_ser is None:
            time_ser = self.time_ser
        if num_states is None:
            num_states = self.num_states
        if n_iter is None:
            n_iter = self.n_iter
        if (
            num_states > self.num_obs
        ):  # avoid senseless fitting of more hidden states than present obsv.
            num_states = self.num_obs + 1
        elif num_states <= self.num_obs:
            num_states += 1
        num_states = num_states
        scores = np.zeros(num_states)
        probas = np.zeros(num_states)
        for i in tqdm(range(2, num_states)):
            model = GaussianHMM(
                n_components=i, n_iter=n_iter, params="st", init_params="st"
            )
            try:
                model.fit(time_ser)
                scores[i] = model.score(time_ser)
                probas[i] = calculate_mean_probas(time_ser, model)
            except:
                scores[i] = 0
                probas[i] = 0
                raise Warning("Did not fit HMM for state=" + str(i))
        self.scores = scores[1:]
        self.probas = probas[1:]
        scores.tofile(
            self.save_path + "/" + self.file_name + "_scores.csv",
            sep=",",
            format="%10.5f",
        )
        probas.tofile(
            self.save_path + "/" + self.file_name + "_probs.csv",
            sep=",",
            format="%10.5f",
        )
        return [scores, probas]

    def plot_analysis(self):
        """Plots the state plot and saves it to the directory

        Args:
            probas ([type]): [description]
            num_states ([type]): [description]
        """
        if self.num_states > self.num_obs:
            states = self.num_obs
        else:
            states = self.num_states
        try:
            plot_state_diagram(
                probabilities=self.probas,
                max_states=states,
                file_name=self.file_name,
                save_path=self.save_path,
            )
        except:
            raise Warning("Plot for ", self.file_name, "not generated.")

        try:
            plot_transmat_map(
                probabilities=self.probas,
                file_name=self.file_name,
                save_path=self.save_path,
            )
        except:
            raise Warning(
                "Plot of the transition matrix for ", self.file_name, "not generated."
            )

    def load_traj(self, path_traj=None):
        """Loads the in path_traj specified Dynophore trajectory

        Args:
            path ([type]): [description]
        """
        if path_traj is None:
            path_traj = self.path_traj

        time_series_json = pd.DataFrame.from_dict(
            parser.get_time_series(self.path_traj)
        )
        obs = time_series_json.drop_duplicates()
        self.num_obs = len(obs)
        print("There are ", self.num_obs, " obersvations present.")
        obs = obs.to_numpy()
        time_ser = time_series_json.to_numpy()
        print("The length of the observation sequence is ", len(time_ser))
        self.time_ser = time_ser
        return self.time_ser


if __name__ == "__main__":
    analysis = HMM_Analyzer("../Trajectories/M2_receptor/M2R-iperoxo_dynophore.pml")
    analysis.find_states()
    analysis.plot_analysis()
    analysis = HMM_Analyzer("../Trajectories/M2_receptor/M2R-QNB_dynophore.pml")
    analysis.find_states()
    analysis.plot_analysis()
    analysis = HMM_Analyzer("../Trajectories/ZIKV/ZIKV-Pro-427-1_dynophore.pml")
    analysis.find_states()
    analysis.plot_analysis()
    analysis = HMM_Analyzer(
        "../Trajectories/M2_receptor/M2Pil/M2R-pilocarpin_dynophore_1.pml"
    )
    analysis.find_states()
    analysis.plot_analysis()
    analysis = HMM_Analyzer(
        "../Trajectories/M2_receptor/M2Pil/M2R-pilocarpin_dynophore_2.pml"
    )
    analysis.find_states()
    analysis.plot_analysis()
    analysis = HMM_Analyzer(
        "../Trajectories/M2_receptor/M2Pil/M2R-pilocarpin_dynophore_3.pml"
    )
    analysis.find_states()
    analysis.plot_analysis()
