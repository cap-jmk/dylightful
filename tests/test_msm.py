import pytest
import pandas as pd
import json
import os


from dylightful.discretizer import tae_discretizer, smooth_projection_k_means
from dylightful.utilities import get_dir, load_parsed_dyno
from dylightful.msm import create_msm

# define relative paths properly
dirname = os.path.dirname(__file__)


@pytest.mark.parametrize(
    "traj_path, discretizer, num_states",
    [
        ("Trajectories/1KE7_dynophore_time_series.json", tae_discretizer, 4),
        (
            "Trajectories/ZIKV/ZIKV-Pro-427-1_dynophore_time_series.json",
            tae_discretizer,
            3,
        ),
    ],
)
def test_msm_generation(traj_path, discretizer, num_states):
    traj_path = os.path.join(dirname, traj_path)
    time_ser, num_obs = load_parsed_dyno(traj_path=traj_path)
    save_path = get_dir(traj_path)
    proj = discretizer(time_ser=time_ser, save_path=save_path)
    labels = smooth_projection_k_means(proj, num_states)
    create_msm(trajectory=labels, save_path=save_path)


@pytest.mark.parametrize("traj_path", ["Trajectories/1KE7_dynophore_time_series.json"])
def test_tae_discretizer(traj_path):
    """Test MSM generation with time lagged
    autoencoders"""
