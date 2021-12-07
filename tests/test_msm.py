import pytest 
import pandas as pd 
import json 
import os


from dylightful.discretizer import tae_discretizer
from dylightful.utilities import get_dir
#define relative paths properly 
dirname = os.path.dirname(__file__)

def test_msm_generation(): 
    pass

@pytest.mark.parametrize("traj_path", 
                         ["Trajectories/1KE7_dynophore_time_series.json"])
def test_tae_discretizer(traj_path): 
    """Test MSM generation with time lagged 
    autoencoders"""
    traj_path = os.path.join(dirname, traj_path)
    time_ser, num_obs = load_parsed_dyno(traj_path=traj_path)
    save_path = get_dir(traj_path)
    tae_discretizer(time_ser=time_ser, save_path=save_path)


def load_parsed_dyno(traj_path):
    """Loads the parsed trajectory from the parser

    Args:
        traj_path ([type]): Path to the parsed traj

    Returns:
        [type]: trajectory as pd.DataFrame, number of observations as int
    """
    with open(traj_path) as f:
        data = json.load(f)
     
    time_ser = pd.DataFrame(data)
    time_ser = time_ser.drop(columns="num_frames")
    obs = time_ser.drop_duplicates()
    num_obs = len(obs)
    print("There are actually ", num_obs, " present.")
    obs = obs.to_numpy()
    time_ser = time_ser.to_numpy()
    print("The length of the observation sequence is ", len(time_ser))  
    return [time_ser, obs]  