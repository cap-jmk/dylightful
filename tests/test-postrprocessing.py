import pytest
import pandas as pd
import json
import os


from dylightful.discretizer import tae_discretizer, smooth_projection_k_means
from dylightful.utilities import get_dir, load_parsed_dyno
from dylightful.msm import fit_msm, build_tae_msm
from dylightful.postprocess import postprocessing_msm

dirname = os.path.dirname(__file__)


@pytest.mark.parametrize(
    "traj_path, dyn_path, discretizer, num_states",
    [
        (
            "Trajectories/ZIKV/ZIKV-Pro-427-1_dynophore_time_series.json",
            "Trajectories/ZIKV/ZIKV-Pro-427-1_dynophore.json",
            tae_discretizer,
            4,
        ),
    ],
)
def test_postprocessing(traj_path, dyn_path, discretizer, num_states):
    traj_path = os.path.join(dirname, traj_path)
    dyn_path = os.path.join(dirname, dyn_path)
    time_ser, num_obs = load_parsed_dyno(traj_path=traj_path)
    save_path = get_dir(traj_path)
    proj = discretizer(time_ser=time_ser, save_path=save_path)
    labels = smooth_projection_k_means(proj, num_states)
    fit_msm(trajectory=labels, save_path=save_path)
    postprocessing_msm(
        labels_states=labels,
        dynophore_json=dyn_path,
        processed_dyn=traj_path,
        save_path=save_path,
    )
