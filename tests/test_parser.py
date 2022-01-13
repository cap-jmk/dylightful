import pytest
import os

from dylightful.parser import get_time_series

dirname = os.path.dirname(__file__)


@pytest.mark.parametrize(
    "traj_path", [("Trajectories/HIVProtease/HIVPro-DMP_dynophore.pml")]
)
def test_dynoparser(traj_path):
    traj_path = os.path.join(dirname, traj_path)
    get_time_series(traj_path)
