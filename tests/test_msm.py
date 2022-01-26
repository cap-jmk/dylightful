from sys import prefix
import pytest
import pandas as pd
import json
import os
import numpy as np

from dylightful.discretizer import tae_discretizer, smooth_projection_k_means
from dylightful.utilities import get_dir, load_parsed_dyno
from dylightful.parser import load_env_partners, load_env_partners_mixed
from dylightful.msm import fit_msm, build_tae_msm

# define relative paths properly
dirname = os.path.dirname(__file__)


@pytest.mark.parametrize(
    "traj_path, discretizer, num_states",
    [
        ("Trajectories/ZIKV/ZIKV-Pro-427-1_dynophore.json", tae_discretizer, 4),
    ],
)
def test_protein_perspective(traj_path, discretizer, num_states):

    prefix = "protein"
    traj_path = os.path.join(dirname, traj_path)
    env_partners = load_env_partners(json_path=traj_path)
    env_partner_arr = []
    for partner in env_partners.keys():
        arr = np.array(env_partners[partner])
        traj = np.zeros(len(arr.T))
        for traject in env_partners[partner]:
            traj += np.array(traject).T
        env_partner_arr.append(traj)

    env_partner_arr = np.array(env_partner_arr)
    time_ser = env_partner_arr.T
    save_path = get_dir(traj_path)
    proj = discretizer(time_ser=time_ser, save_path=save_path, prefix=prefix)
    labels = smooth_projection_k_means(proj, num_states)
    fit_msm(
        trajectory=labels,
        save_path=save_path,
        prefix=prefix,
    )


@pytest.mark.parametrize(
    "traj_path, discretizer, num_states",
    [
        ("Trajectories/ZIKV/ZIKV-Pro-427-1_dynophore.json", tae_discretizer, 4),
    ],
)
def test_mixed_perspective(traj_path, discretizer, num_states):

    traj_path = os.path.join(dirname, traj_path)
    env_partners = load_env_partners_mixed(json_path=traj_path)
    env_partner_arr = []
    for partner in env_partners.keys():
        env_partner_arr.append(env_partners[partner])

    env_partner_arr = np.array(env_partner_arr)
    time_ser = env_partner_arr.T
    save_path = get_dir(traj_path)
    proj = discretizer(time_ser=time_ser, save_path=save_path, prefix="mixed")
    labels = smooth_projection_k_means(proj, num_states)
    fit_msm(trajectory=labels, save_path=save_path, prefix="mixed")


@pytest.mark.parametrize(
    "traj_path, discretizer, num_states",
    [
        ("Trajectories/1KE7_dynophore_time_series.json", tae_discretizer, 4),
        (
            "Trajectories/HIVProtease/HIVPro-DMP_dynophore_time_series.json",
            tae_discretizer,
            4,
        ),
    ],
)
def test_ligand_perspective(traj_path, discretizer, num_states):

    prefix = "ligan"
    traj_path = os.path.join(dirname, traj_path)
    time_ser, num_obs = load_parsed_dyno(traj_path=traj_path)
    save_path = get_dir(traj_path)
    proj = discretizer(time_ser=time_ser, save_path=save_path, prefix=prefix)
    labels = smooth_projection_k_means(proj, num_states)
    fit_msm(trajectory=labels, save_path=save_path, prefix=prefix)


@pytest.mark.parametrize(
    "traj_path,  num_states",
    [
        ("Trajectories/1KE7_dynophore_time_series.json", 4),
    ],
)
def test_build_tae_msm(traj_path, num_states):
    traj_path = os.path.join(dirname, traj_path)
    time_ser, num_obs = load_parsed_dyno(traj_path=traj_path)
    build_tae_msm(traj_path, time_ser=time_ser, num_obs=num_obs, num_states=num_states)
