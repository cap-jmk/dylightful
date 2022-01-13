import matplotlib.pyplot as plt
from deeptime.markov import TransitionCountEstimator
import deeptime.markov as markov
import seaborn as sns
import os

from dylightful.utilities import make_name, get_dir, load_parsed_dyno
from dylightful.discretizer import tae_discretizer, smooth_projection_k_means
from dylightful.bar_plot import make_barplot


def build_tae_msm(traj_path, num_states):
    """does the tae analysis of a dynophore trajectory

    Args:
        traj_path (string): path to trajectory to be discretized
        num_states (int): assumed states
    """
    # dirname = os.path.dirname(__file__)
    # traj_path = os.path.join(dirname, traj_path)
    time_ser, num_obs = load_parsed_dyno(traj_path=traj_path)
    save_path = get_dir(traj_path)
    proj = tae_discretizer(time_ser=time_ser, save_path=save_path)
    labels = smooth_projection_k_means(proj, num_states)
    msm = fit_msm(trajectory=labels, save_path=save_path)
    make_barplot(time_ser, prefix=None, save_path=traj_path)
    return msm, labels, proj, time_ser


def fit_msm(trajectory, prefix=None, save_path=None):
    """Function to fit the msm to a given trajectory and save the vizualization

    Args:
        trajectory ([type]): Time series to be discretized
        prefix ([type], optional):  Name to save a file.
        save_path ([type], optional):  Wheree to save a file.

    Returns:
        [type]: msm
    """
    plt.cla()
    plt.clf()
    name = "_msm_transistion_matrix.png"
    file_name = make_name(prefix=prefix, name=name, dir=save_path)

    estimator = TransitionCountEstimator(lagtime=1, count_mode="sliding")
    counts = estimator.fit(trajectory).fetch_model()  # fit and fetch the model
    estimator = markov.msm.MaximumLikelihoodMSM(
        reversible=True, stationary_distribution_constraint=None
    )

    msm = estimator.fit(counts).fetch_model()  # TSM
    ax = sns.heatmap(msm.transition_matrix)
    fig = ax.get_figure()
    plt.xlabel("State")
    plt.ylabel("State")
    plt.savefig(file_name, dpi=300)
    return msm


def map_pharmacophore(labels, parsed_states):
    pass
