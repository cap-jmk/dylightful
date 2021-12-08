import pytest
import pandas as pd
import json
import os


from dylightful.discretizer import tae_discretizer, smooth_projection_k_means
from dylightful.utilities import get_dir, load_parsed_dyno

# define relative paths properly
dirname = os.path.dirname(__file__)


@pytest.mark.parametrize(
    "traj_path, discretizer, num_states",
    [("Trajectories/1KE7_dynophore_time_series.json", tae_discretizer, 4)],
)
def test_msm_generation(traj_path, discretizer, num_states):
    traj_path = os.path.join(dirname, traj_path)
    time_ser, num_obs = load_parsed_dyno(traj_path=traj_path)
    save_path = get_dir(traj_path)
    proj = discretizer(time_ser=time_ser, save_path=save_path)
    labels = smooth_projection_k_means(proj, num_states)


@pytest.mark.parametrize("traj_path", ["Trajectories/1KE7_dynophore_time_series.json"])
def test_tae_discretizer(traj_path):
    """Test MSM generation with time lagged 
    autoencoders"""
