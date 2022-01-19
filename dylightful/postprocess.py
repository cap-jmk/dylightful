import numpy as np 


def postprocessing_msm():
    
    #procedures left from postprocessing
    indc =state_data["frameIndices"]
    state_data["env_partners_list"]=partners

    env_partners = load_env_partners("../tests/Trajectories/ZIKV/ZIKV-Pro-427-1_dynophore.json")
    store = {}  
    for i in np.unique(labels_states): 
        store[str(i)] = []
    
##
def generate_state_map(time_ser_superf, labels_states, state_data): 
    """Generates a map foar each time point of a Markov Sate to a Superfeature tuple such that the time series is the
    (0,1,2), (0,1,3), (0,1,2) etc...

    Args:
        time_ser_superf ([type]): loaded combined array of superfeature occurences such that shape is ( len_md_traj, num_superfeatures,)
        labels_states ([type]): time series of the labels of the different Markov states

    Returns:
        [np.ndarray]: array of 
    """

    for i in range(len(time_ser_superf)): 
        state = str(labels_states[i])
        state_map=state_data[state]
        state_map.append(np.where(time_ser_superf[i,:]==1)[0].astype(np.int16).tolist())
        state_data[state] = state_map
    return state_map

##
def ??(labels, state_data):  
    
    for state in state_data.keys(): 
        ## include information about the markov_state
        occ = np.where(labels == int(state))
        num_occ = len(occ)
        state_data = {}
        state_data["frameIndices_state"] = occ
        state_data["num"] = num_occ
        #cut unique pharmacophores/superf.patterns
        unique_parmc = np.unique(np.array(store[state])).tolist()
        state_data["pharmc"] = unique_parmc
        superfeats = set()
        ## cut 
        for pharmc in unique_parmc:
            superfeats = superfeats.union(set(pharmc)) 
        state_data["distinctSuperfeatures"] = list(superfeats)



def get_env_partners(frame_indices, env_partners):
    """

    Args:
        frameIndices_state ([type]): [description]
        env_partners ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    env_partner_arr = []
    residues = list(env_partners.keys())
    for partner in env_partners.keys():
        env_partner_arr.append(env_partners[partner])
        
    env_partner_arr = np.array(env_partner_arr)
    partners = []
    
    for i in range(len(frame_indices[0])):
        eps = np.array(residues)[np.where(env_partner_arr[:, indc][:,0,i] == 1)[0].tolist()]
        partners.append(eps)


    return partners
    
    
    


def get_unique_env_partner(partner_traj): 
    """Counts env_partners from the env_partner trajectory 

    Args:
        partner_traj ([type]): specific to a state returns the trajectory of the env partners 

    Returns:
        [(unique_partners, their_counts)]:
    """
    
    count = []
    for partner in partner_traj: 
        count+=(partner.tolist())
    unique, counts = np.unique(count, return_counts = True)
    return unique, counts