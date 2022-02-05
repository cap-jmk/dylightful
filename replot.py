import os
import numpy as np
from dylightful.utilities import load_parsed_dyno, get_dir
from dylightful.parser import load_env_partners_mixed
from dylightful.bar_plot import make_barplot

dirname = os.path.dirname(__file__)

save_dir = "/media/julian/INTENSO/ZIKV/"
name = "ZIKV-Pro-427-1_dynophore"
traj_path = save_dir + name + ".json"
prefix = "ligand"
pml = save_dir + name + ".pml"
ligand_path = save_dir + name + "_time_series.json"
topology = save_dir + "topo0.pdb"
trajectory = save_dir + "trajectory.dcd"
traj_path = os.path.join(dirname, traj_path)
time_ser, num_obs = load_parsed_dyno(traj_path=ligand_path)
time_ser = time_ser
print(type(time_ser))
print(time_ser.shape)
print(time_ser[0:2, :])
# traj_path = os.path.join(dirname, traj_path)
# env_partners = load_env_partners_mixed(json_path=traj_path)
# env_partner_arr = []
# for partner in env_partners.keys():
#    env_partner_arr.append(env_partners[partner])
# env_partner_arr = np.array(env_partner_arr)
# time_ser = env_partner_arr.T
# print(type(time_ser))
# print(time_ser.shape)
# print(time_ser[0:2,:])


save_path = get_dir(traj_path)
base = save_dir
make_barplot(
    time_ser=time_ser,
    ylabel="Ligand Perspective",
    yticks=np.arange(time_ser.shape[1]),
    prefix=prefix,
    save_path=save_path,
)
