# test postprocessing of MSM for validation purposes and additional viz.

import os

import pytest
from dylightful.discretizer import smooth_projection_k_means, tae_discretizer
from dylightful.mdanalysis import write_dcd, write_state
from dylightful.utilities import get_dir, load_parsed_dyno

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
def test_write_state(traj_path, dyn_path, discretizer, num_states):
    """Testing the writing function of the MDanalysis script

    Args:
        traj_path ([type]): [description]
        dyn_path ([type]): [description]
        discretizer ([type]): [description]
        num_states ([type]): [description]
    """

    topology = os.path.join(dirname, "Trajectories/ZIKV/startframe.pdb")
    coordinates = os.path.join(dirname, "Trajectories/ZIKV/test.dcd")
    base = os.path.join(dirname, "Trajectories/ZIKV/")
    prefix = "ligand_view_"
    traj_path = os.path.join(dirname, traj_path)
    dyn_path = os.path.join(dirname, dyn_path)
    time_ser, num_obs = load_parsed_dyno(traj_path=traj_path)
    save_path = get_dir(traj_path)
    proj = discretizer(time_ser=time_ser, num_states=num_states, save_path=save_path)
    labels = smooth_projection_k_means(proj, num_states)
    write_state(
        labels=labels[:100], topology=topology, coordinates=coordinates, base=base
    )


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
def test_write_dcd(traj_path, dyn_path, discretizer, num_states):
    """Testing the writing function of the MDanalysis script

    Args:
        traj_path ([type]): [description]
        dyn_path ([type]): [description]
        discretizer ([type]): [description]
        num_states ([type]): [description]
    """

    topology = os.path.join(dirname, "Trajectories/ZIKV/startframe.pdb")
    coordinates = os.path.join(dirname, "Trajectories/ZIKV/test.dcd")
    base = os.path.join(dirname, "Trajectories/ZIKV/")
    prefix = "ligand_view_"
    traj_path = os.path.join(dirname, traj_path)
    dyn_path = os.path.join(dirname, dyn_path)
    time_ser, num_obs = load_parsed_dyno(traj_path=traj_path)
    save_path = get_dir(traj_path)
    proj = discretizer(time_ser=time_ser, num_states=num_states, save_path=save_path)
    labels = smooth_projection_k_means(proj, num_states)
    write_dcd(
        labels=labels[:100], topology=topology, coordinates=coordinates, base=base
    )
