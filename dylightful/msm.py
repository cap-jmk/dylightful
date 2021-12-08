import matplotlib.pyplot as plt
from deeptime.markov import TransitionCountEstimator
import deeptime.markov as markov
import seaborn as sns

from dylightful.utilities import make_name


def create_msm(trajectory, prefix=None, save_path=None):

    name = "_msm_transistion_matrix.png"
    file_name = make_name(prefix=prefix, name=name, dir=save_path)

    estimator = TransitionCountEstimator(lagtime=1, count_mode="sliding")
    counts = estimator.fit(trajectory).fetch_model()  # fit and fetch the model
    estimator = markov.msm.MaximumLikelihoodMSM(
        reversible=True, stationary_distribution_constraint=None
    )

    msm = estimator.fit(counts).fetch_model()
    ax = sns.heatmap(msm.transition_matrix)
    fig = ax.figure()
    plt.xlabel("State")
    plt.ylabel("State")
    plt.savefig(file_name, dpi=300)
    return msm
