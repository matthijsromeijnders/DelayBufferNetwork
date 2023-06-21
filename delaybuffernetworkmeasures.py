import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import normalize
from delaybuffernetwork import DelayBufferNetwork


def calculate_propagator_matrix(network: DelayBufferNetwork, time):
    """Calculates a propagator matrix over a SINGLE TIMESTEP as discussed in the paper: 
    Hidden dependence of spreading vulnerability on topological complexity, 
    by Dekker, Panja, Schram, and Ou. 
    Doi: https://doi.org/10.1103/PhysRevE.105.054301 
    

    Args:
        network (DelayBufferNetwork): A DelayBufferNetwork
    Returns:
        propagator_matrix (np.ndarray): an n x n matrix.
    """
    network_at_time = network.network[network.network["t"] == time]
    events = np.unique(network_at_time["event_id"])
    number_of_events = len(events)
    n_agents = int(max(len(network.unique_agents), np.max(network.unique_agents)+1))
    q_in = np.zeros((n_agents, number_of_events))
    q_out = np.zeros((number_of_events, n_agents))
    propagator_matrix = np.zeros((n_agents, n_agents))
    agents_last_event = 0
    if network_at_time.empty:
        for i in range(n_agents):
            propagator_matrix[i, i] = 1
        return propagator_matrix
    # Loop over events, and build Qin and Qout matrices.
    for event_iterator, event_id in enumerate(events):
        network_at_event = network_at_time[network.network["event_id"] == event_id]
        agents_at_event = np.unique(np.concatenate((network_at_event["i"], network_at_event["j"])))
        # This if statement is done in order to get rid of duplicate events in the data.
        if event_id > 0:
            if np.all(agents_last_event != agents_at_event):
                try:
                    q_in[agents_at_event, event_iterator] = 1
                except IndexError:
                    print(q_in.shape, agents_at_event, np.max(agents_at_event))
                    q_in[agents_at_event, event_iterator] = 1
                q_out[event_iterator, agents_at_event] = network_at_event['weight'].to_numpy()[0]
                agents_last_event = agents_at_event
        else:
            network_at_event = network_at_time[network.network["event_id"] == event_id]
            q_in[agents_at_event, event_iterator] = 1
            q_out[event_iterator, agents_at_event] = network_at_event['weight'].to_numpy()[0]
            agents_last_event = agents_at_event
    q_out = np.where(
        q_out.sum(axis=1, keepdims=True) != 0, q_out / q_out.sum(axis=1, keepdims=True), 0)
    propagator_matrix = q_in @ q_out
    propagator_matrix = normalize(propagator_matrix, norm="l1", axis=0)
    propagator_matrix = normalize(propagator_matrix, norm="l1", axis=1)
    for i in range(n_agents):
        if propagator_matrix[i, i] == 0:
            propagator_matrix[i, i] = 1
    return propagator_matrix


def calculate_all_propagator_matrices(network: DelayBufferNetwork):
    """Returns propagator matrices for each timestep for a network.

    Args:
        network (DelayBufferNetwork): A DelayBufferNetwork.
    Returns:
        propagator_matrices (list): list of propagator matrices.
    """
    time_range = network.time_range
    if time_range is None:
        times = np.unique(network.network["t"])
    else:
        times = network.time_range
    propagator_matrices = []
    for time in times:
        propagator_matrix = calculate_propagator_matrix(network, time)
        propagator_matrices.append(propagator_matrix)
    return propagator_matrices


def calculate_entanglement_product_matrix(network: DelayBufferNetwork, 
                                          t_index,
                                          delta_t_index, 
                                          propagator_mats, 
                                          time_window=True):
    """Calculates entanglement product matrix with a given t, 
    and delta t, in the form of indices, not values.

    Args:
        network (DelayBufferNetwork): DelayBufferNetwork
        t_index (int): time
        delta_t_index (int): time window

    Returns:
        np.ndarray: Array which holds the product entanglement matrix
    """
    product_entanglement_mat = propagator_mats[t_index]
    time_range = network.time_range

    if time_window:
        if time_range is None:
            max_time = network.network["t"].max()
            t_idx = np.unique(network.network["t"])[t_index]
            t2_time = min(max_time, t_idx + delta_t_index)
            number_of_times = len(np.unique(
                network.network[network.network["t"].between(
                    t_idx, t2_time, inclusive='left')].t))
            for i in range(1, number_of_times):
                product_entanglement_mat = product_entanglement_mat @ propagator_mats[t_index + i]
        else:
            time_range = network.time_range
            max_time = network.network["t"].max()
            t_idx = time_range[t_index]
            t1_idx = np.argwhere(time_range > t_idx - 0.001)[0][0]
            t2_time = min(max_time, t_idx + delta_t_index)
           
            try:
                t2_idx = np.argwhere(time_range > t2_time - 0.001)[0][0]
            except IndexError:
                t2_idx = len(time_range)-1
            number_of_times = t2_idx - t1_idx
            for i in range(1, number_of_times):
                product_entanglement_mat = product_entanglement_mat @ propagator_mats[t_index + i]
    else:
        max_time = len(propagator_mats)
        delta_t_index = min(max_time - t_index, delta_t_index)
        for i in range(1, delta_t_index):
            product_entanglement_mat = product_entanglement_mat @ propagator_mats[t_index + i]
    return product_entanglement_mat


def calculate_entanglement_entropy(network: DelayBufferNetwork, t_index, delta_t_index, propagator_matrices, time_window=True):
    """Calculates the entanglement entropy at a given time with a given time window by summing array elements

    Args:
        network (DelayBufferNetwork): DelayBufferNetwork
        t_index (int): time
        delta_t_index (int): time window

    Returns:
        float: entanglement entropy
    
    Notes:
        Could be faster by precalculating N. Kept like this for clarity.
    """
    rho_matrix = calculate_entanglement_product_matrix(network, t_index, delta_t_index, propagator_matrices, time_window)
    n_agents = len(network.unique_agents)
    rho_elements = rho_matrix.flatten()
    entropy = -np.sum(np.where(rho_elements != 0, rho_elements * np.log(rho_elements) / (n_agents * np.log(n_agents)), 0))
    if entropy < 0 or entropy > 1.00001:
        print("Entropy exceeds 1")
        print(rho_elements, entropy)
    return entropy


def entropy_fast(network: DelayBufferNetwork, delta_t, prop_mats, shorthand_mats_per_delta_t, time_window=True):
    """Attempts a high level optimization on the entropy calculation by precalculating running matrices.
    and multiplying by the new matrix and the inverse of the first matrix (in a special way). Does not work since the matrices are not invertible sadly.

    Args:
        network (_type_): _description_
        delta_t_index (_type_): _description_
        propagator_matrices (_type_): _description_

    Returns:
        entropies (list): list that holds all the calculated entropies
    """
    max_time_idx = len(prop_mats)
    n_agents = len(network.unique_agents)
    
    time_value_range = network.network.t.max() - network.network.t.min()
    time_per_prop_mat = int(time_value_range / len(prop_mats)) # Take the max to solve the case where time window = time step window.
    

    if time_window:
        prop_mats_inside_shorthand_mat = int(delta_t / time_per_prop_mat / shorthand_mats_per_delta_t)
        original_delta_t = delta_t
    else:
        prop_mats_inside_shorthand_mat = int(delta_t / shorthand_mats_per_delta_t)
        
    number_shorthand_matrices = int(len(prop_mats) / prop_mats_inside_shorthand_mat)
    shorthand_mats = []
    
    #lists to keep track of running matrix indices
    shorthand_mat_idx_0 = [] 
    shorthand_mat_idx_1 = []
    print("Running matrices:")
    for i in tqdm(range(number_shorthand_matrices)):
        prop_mat_idx = i * prop_mats_inside_shorthand_mat
        shorthand_mat_curr = prop_mats[prop_mat_idx]
        shorthand_mat_idx_0.append(prop_mat_idx)
        shorthand_mat_idx_1.append(prop_mat_idx + prop_mats_inside_shorthand_mat)
        for j in range(1, prop_mats_inside_shorthand_mat):
            shorthand_mat_curr = shorthand_mat_curr @ prop_mats[prop_mat_idx + j]
        shorthand_mats.append(shorthand_mat_curr)

    start_indices = np.array(shorthand_mat_idx_0)
    end_indices = np.array(shorthand_mat_idx_1)
    
    if network.time_range is None:
        time_range = len(np.unique(network.network["t"]))
    else:
        time_range = network.time_range
    
    entropies = []     
    print("Entropies:")
    for t_idx in tqdm(range(time_range - 1)):
        
        if not time_window:
            delta_t = min(max_time_idx - t_idx, delta_t)
        else:
            max_time = network.network["t"].max()
            t_var = np.unique(network.network["t"])[t_idx]
            t_plus_dt = min(max_time, t_var + original_delta_t)
            time_window_size = len(np.unique(network.network[network.network["t"].between(t_var, t_plus_dt, inclusive='left')].t))
            delta_t = time_window_size
        
        # Find index of first upcoming running matrix
        shorthand_matrix_start_idx = (np.abs(start_indices - t_idx)).argmin()
        shorthand_matrix_start_idx_mapped = start_indices[shorthand_matrix_start_idx]
        if shorthand_matrix_start_idx_mapped <= t_idx:
            if shorthand_matrix_start_idx + 1 < len(start_indices):
                shorthand_matrix_start_idx += 1
                shorthand_matrix_start_idx_mapped = start_indices[shorthand_matrix_start_idx]
        
        shorthand_matrix_end_idx = (np.abs(end_indices - (t_idx + delta_t))).argmin()
        shorthand_matrix_end_idx_mapped = end_indices[shorthand_matrix_end_idx]
        if shorthand_matrix_end_idx_mapped >= t_idx + delta_t and shorthand_matrix_end_idx > 0:
            shorthand_matrix_end_idx -= 1
            shorthand_matrix_end_idx_mapped = end_indices[shorthand_matrix_end_idx]       
        
        if t_idx < (number_shorthand_matrices - 1) * prop_mats_inside_shorthand_mat and shorthand_matrix_start_idx_mapped < shorthand_matrix_end_idx_mapped:
            front_end_product_matrix = prop_mats[t_idx]
            for i in range(t_idx + 1, shorthand_matrix_start_idx_mapped):
                front_end_product_matrix = front_end_product_matrix.dot(prop_mats[i])

            shorthand_product_matrix = shorthand_mats[shorthand_matrix_start_idx]
            for i in range(shorthand_matrix_start_idx + 1, shorthand_matrix_end_idx + 1):
                shorthand_product_matrix = shorthand_product_matrix.dot(shorthand_mats[i])

            back_end_product_matrix = prop_mats[shorthand_matrix_end_idx_mapped]
            for i in range(shorthand_matrix_end_idx_mapped + 1, t_idx + delta_t):
                try:
                    back_end_product_matrix = back_end_product_matrix.dot(prop_mats[i])
                except IndexError:
                    print(t_var, t_plus_dt, delta_t, t_idx)

            product_matrix = front_end_product_matrix.dot(shorthand_product_matrix.dot(back_end_product_matrix))
        else:
            if not time_window:
                delta_t = min(max_time_idx - t_idx, delta_t)
            else:  
                max_time = network.network["t"].max()
                t_var = network.network["t"].iloc[t_idx]
                t_plus_dt = min(max_time, t_var + delta_t)
                time_window_size = len(np.unique(network.network[network.network["t"].between(t_var, t_plus_dt, inclusive='left')].t))
                delta_t = time_window_size
            product_matrix = prop_mats[t_idx]
            for i in range(1, delta_t):
                product_matrix = product_matrix.dot(prop_mats[t_idx + i])

        rho_elements = product_matrix.flatten()
        entropy = -np.sum(np.where(rho_elements != 0, rho_elements * np.log(rho_elements) / (n_agents * np.log(n_agents)), 0))
        entropies.append(entropy)
    return entropies


def entanglement_entropy(network: DelayBufferNetwork, delta_t_index, fast=True, time_window=True):
    """Calculates entanglement entropy over the whole timeframe of the network

    Args:
        network (DelayBufferNetwork): The DelayBufferNetwork
        delta_t_index (float/int): Time window within which to look.

    Returns:
        entropies: list of entanglement entropies at each time interval.
    """
    time_range = network.time_range
    #if network.propagator_matrices == []:
    network.propagator_matrices = calculate_all_propagator_matrices(network)
    
    if time_range is not None:
        fast = False
        entropies = []
        for time in range(len(time_range)):
            entropies.append(calculate_entanglement_entropy(network, time, delta_t_index, network.propagator_matrices, time_window))
        return entropies
    
    print("Calculating entropies")
    if fast:
        entropies = entropy_fast(network, delta_t_index, network.propagator_matrices, 5, time_window)
    else:
        entropies = []
        for time in tqdm(range(len(np.unique(network.network["t"])))):
            entropies.append(calculate_entanglement_entropy(network, time, delta_t_index, network.propagator_matrices, time_window))

    return entropies
    
    
def entanglement_entropy_per_delta_t(network: DelayBufferNetwork, t_index, delta_t_max, fast=True, time_window=True):
    """Calculates entanglement entropy over the whole timeframe of the network

    Args:
        network (DelayBufferNetwork): The DelayBufferNetwork
        delta_t_index (float/int): Time window within which to look.

    Returns:
        entropies: list of entanglement entropies at each time interval.
    """
    if network.propagator_matrices == []:
        network.propagator_matrices = calculate_all_propagator_matrices(network)
    entropies = []
    
    print("Calculating entropies")
    if fast:
        n_agents = len(network.unique_agents)
        for delta_t in tqdm(range(delta_t_max)):
            max_time = len(network.propagator_matrices)
            delta_t_index = min(max_time - delta_t, delta_t)
            if delta_t == 0:
                product_entanglement_matrix = network.propagator_matrices[0]
            else:
                product_entanglement_matrix = product_entanglement_matrix @ network.propagator_matrices[delta_t_index]
            rho_elements = product_entanglement_matrix.flatten()
            entropy = -np.sum(np.where(rho_elements != 0, rho_elements * np.log(rho_elements) / (n_agents * np.log(n_agents)), 0)) 
            entropies.append(entropy)
    else:
        for delta_t in tqdm(range(delta_t_max)):
            entropies.append(calculate_entanglement_entropy(network, t_index, delta_t, network.propagator_matrices, time_window))

    return entropies
    
    
def entropy_convergence_time(dbn: DelayBufferNetwork, dbn_delayed: DelayBufferNetwork, tdelta : float, alpha : float, consecutivity: float, plot: bool = False):
    """Calculates the entropy convergence time between two networks. Generally pass one undelayed network (no delays are processed)
        and a delayed network (delays processed).

    Args:
        dbn (DelayBufferNetwork): undelayed DelayBufferNetwork
        dbn_delayed (DelayBufferNetwork): delayed DelayBufferNetwork
        tdelta (float): delta t for the entropy calculation
        alpha (float): alpha condition for deciding two entropies are the same
        consecutivity (float): How many steps do entropies need to be the same before we call them the same for the rest of the time series?

    Returns:
        t - consecutivity - 1: Entropy convergence time.
    """
    entropy_base = entanglement_entropy(dbn, tdelta)
    entropy_delay = entanglement_entropy(dbn_delayed, tdelta)
    
    consecutive_success_count = 0 
    for time in np.unique(dbn.network.t):
        entropy_base_t = entropy_base[int(time)]
        idx_t_delay = np.argmin(np.abs(np.unique(dbn_delayed.network.t)-time))
        entropy_delay_t = entropy_delay[idx_t_delay]
        if abs(entropy_base_t - entropy_delay_t) < alpha:
            consecutive_success_count += 1
        else:
            consecutive_success_count = 0
        
        if consecutive_success_count >= consecutivity:
            return time - consecutivity + 1, (np.unique(dbn.network.t), entropy_base), (np.unique(dbn_delayed.network.t), entropy_delay)
    if plot:
        plt.plot(np.unique(dbn.network.t), entropy_base)
        plt.plot(np.unique(dbn_delayed.network.t), entropy_delay)
        plt.show()

    return None, (np.unique(dbn.network.t), entropy_base), (np.unique(dbn_delayed.network.t), entropy_delay)
    

def entropy_difference(dbn: DelayBufferNetwork, dbn_delayed: DelayBufferNetwork, tdelta : float):
    """Calculates the entropy difference over two DBNs, by comparing their time series.

    Args:
        dbn (DelayBufferNetwork): Original DBN.
        dbn_delayed (DelayBufferNetwork): Delayed DBN.
        tdelta (float): Delta t for the entropy calculation.

    Returns:
        (float): Sum of entropy difference.
    """
    entropy_base = entanglement_entropy(dbn, tdelta)
    entropy_delay = entanglement_entropy(dbn_delayed, tdelta)
    entropy_diff = []
    for time in np.unique(dbn.network.t):
        entropy_base_t = entropy_base[int(time)]
        idx_t_delay = np.argmin(np.abs(np.unique(dbn_delayed.network.t)-time))
        entropy_delay_t = entropy_delay[idx_t_delay]
        entropy_diff.append(np.abs(entropy_base_t - entropy_delay_t))
    return np.sum(entropy_diff)