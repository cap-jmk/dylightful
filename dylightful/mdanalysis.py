import MDAnalysis
import json
import numpy as np


def write_state(
    labels,
    topology,
    coordinates,
    base,
    prefix=None,
    selection_string="protein or resname *",
):
    """write out multipdb for markov state of given perspective

    Args:
        labels ([type]): markov state trajectory
        topology ([type]): [description]
        coordinates ([type]): [description]
        base ([type]): [description]
    """
    name = "_state_"
    if prefix is not None:
        name = prefix + name
    u = MDAnalysis.Universe(topology, coordinates)
    num_states = len(np.unique(labels))
    print("Writing states to", base + name)
    for i in range(num_states):
        ind = np.where(labels == i)[0]
        protein = u.select_atoms(selection_string)
        with MDAnalysis.Writer(base + name + str(i) + ".pdb", protein.n_atoms) as W:
            for ts in u.trajectory[ind]:
                W.write(protein)
    return


def load_validation():
    """[summary]

    Returns:
        [type]: [description]
    """

    f = open("../tests/Trajectories/ZIKV/markophore_validation.json")
    data = json.load(f)
    return data
