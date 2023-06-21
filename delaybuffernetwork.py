import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from tqdm import tqdm
import warnings
import scipy.optimize
import warnings
import networkx as nx
import pickle
warnings.filterwarnings("ignore")

class DelayBufferNetwork():
    def __init__(self, uniform_time_range=False,
                 dont_build_df=False, del_df=False, load=False, path=None, random_event_dict=False):
        #super().__init__(N, T, nettype, from_df, from_array, from_dict, from_edgelist, timetype, diagonal,
        #         timeunit, desc, starttime, nodelabels, timelabels, hdf5, hdf5path, forcesparse, dense_threshold)
        if not random_event_dict:
            if load:
                self.load_event_arrays(path)
                self.load_dicts(path)
            if from_df is not None:
                self.network = from_df.copy()
            else:
                self.network = None  
            self.unique_agents = self.get_unique_agents()
            self.last_event_list = self.unique_agents
            self.N_agents = len(self.unique_agents)
            self.propagator_matrices = []
            self.centralities = None
            self.uniform_sampling = uniform_time_range
            self.time_range = None
            self.set_time_range()
            
            if not load:
                #self.add_self_edges()
                self.wipe_buffers()
                self.wipe_delays()
                if not dont_build_df:
                    self.construct_agent_delays()
                    self.construct_agent_delays_realtime()
                
                
                self.build_event_dict()
                self.build_event_arrays()
                if del_df:
                    del from_df
                    self.network=None
        else:
            self.propagator_matrices = []
            self.centralities = None
            self.time_range = None
            
    
    def save_dicts(self, path):
        """Saves the dictionaries to a path.
        """
        with open(path+"_dict.pkl", "wb") as f:
            pickle.dump(self.event_dict, f)
        with open(path+"_time_dict.pkl", "wb") as f:
            pickle.dump(self.time_event_dict, f)
    
    
    def load_dicts(self, path):
        """Loads the dictionaries of the DBN from a path.
        """
        with open(path+"_dict.pkl", 'rb') as f:
            self.event_dict = pickle.load(f)
        try:
            with open(path+"_time_dict.pkl", "rb") as f:
                self.time_event_dict = pickle.load(f)
        except:
            pass
        
        
    def save_event_arrays(self, path):
        """Saves the event arrays to a path.
        """
        event_array_list = [
                            self.event_current_delay_array,
                            self.event_buffer_array,
                            self.event_added_delay_array
                            ]
        np.save(path + "_arrays" + ".npy", event_array_list)
        np.save(path + "_agent_array" + ".npy", self.event_agent_array)
        np.save(path + "_time_array" + ".npy", self.event_time_array)
        np.save(path + "_agent_delays_array" + ".npy", self.agent_delays)
        
    def load_event_arrays(self, path):
        """Loads the event arrays of the DBN from a path.
        """
        event_array_list = np.load(path+ "_arrays.npy", allow_pickle=True)
        event_time_array = np.load(path+"_time_array.npy", allow_pickle=True)
        event_agent_array = np.load(path+"_agent_array.npy", allow_pickle=True)
        event_agent_delays_array = np.load(path+"_agent_delays_array.npy", allow_pickle=True)
        
        self.event_agent_array = event_agent_array
        self.event_current_delay_array = event_array_list[0]
        self.event_buffer_array = event_array_list[1]
        self.event_added_delay_array = event_array_list[2]
        self.event_time_array = event_time_array
        self.agent_delays = event_agent_delays_array
        
        self.total_events = self.event_current_delay_array.shape[0]
     
    
    def build_event_dict(self):
        """Build event dictionary from a df, and event time array. Necessary for running process_delays_dict().
        """
        self.event_dict = {}
        self.time_event_dict = {}
        self.total_events = len(np.unique(self.network.event_id))
        self.agent_delays = np.zeros((self.total_events, len(self.unique_agents)))
        self.event_time_array = np.zeros((self.total_events,2))
        self.unique_agents = np.unique(np.concatenate((self.network["i"], self.network["j"])))
        #if np.max(self.unique_agents) > np.len(self.unique_agents):
            
            
        for index, row in self.network.iterrows():
            i = row['i']
            j = row['j']
            t = row['t']
            buffer = row["buffer"]
            delay = row["delay"]
            event_id = row['event_id']

            if event_id not in self.event_dict:
                # If not, add it to the dictionary and create a new set to hold unique (i,j,t) values
                self.event_dict[event_id] = {"agents": np.unique([i, j]).astype(int).tolist(), "t": t, "delay": delay, "buffer": buffer, "current_event_delay": 0}
                self.event_time_array[int(event_id),:] = np.array([int(event_id), t])
            else:
                if i not in self.event_dict[event_id]["agents"]:
                    self.event_dict[event_id]["agents"].append(int(i))
                if j not in self.event_dict[event_id]["agents"]:
                    self.event_dict[event_id]["agents"].append(int(j))
            
        
    def build_random_event_dict(self, N, T, k, B):
        """Build a random dictionary, and event time array. Necessary for running process_delays_dict().
        """
        self.event_dict = {}
        self.time_event_dict = {}
        self.total_events = int(T*N/k)
        self.agent_delays = np.zeros((self.total_events, N))
        self.event_time_array = np.zeros((self.total_events,2))
        self.unique_agents = np.arange(N)
        delays = scipy.stats.expon.rvs(scale=1, loc=0, size = self.total_events)
        for event_id in range(self.total_events):
            agents = np.random.choice(N, k)
            t = np.floor(event_id/(int(self.total_events/T)))
            self.event_dict[event_id] = {"agents": agents, "t": t, "delay": delays[event_id], "buffer": B, "current_event_delay": 0}
            self.event_time_array[int(event_id),:] = np.array([int(event_id), t])
        #self.build_event_arrays()
        
        
    def build_event_arrays(self):
        """Build event arrays, and event time array. Necessary for running process_delays_fast_arrays().
        """
        self.event_agent_array = np.zeros((self.total_events, len(self.unique_agents)))
        self.event_current_delay_array = np.zeros((self.total_events))
        self.event_buffer_array = np.zeros((self.total_events))
        self.event_added_delay_array = np.zeros((self.total_events))
    
        
        for event_id, event_dict in self.event_dict.items():
            self.event_added_delay_array[int(event_id)] = event_dict["delay"]
            self.event_buffer_array[int(event_id)] = event_dict["buffer"]
            self.event_current_delay_array[int(event_id)] = event_dict["current_event_delay"]
            
            agents_at_event = event_dict["agents"]
            for agent in agents_at_event:
                self.event_agent_array[int(event_id), agent] = True
        
        max_events_per_time = 0
        for unique_time in np.unique(self.event_time_array[:,1]):
            events_at_time = self.event_time_array[self.event_time_array[:,1] == unique_time]
            if len(events_at_time) > max_events_per_time:
                max_events_per_time = len(events_at_time)
        
        self.time_event_array = np.zeros((len(np.unique(self.event_time_array[:,1])), max_events_per_time))
        self.time_event_dict = {}
        for unique_time in np.unique(self.event_time_array[:,1]):
            events = self.event_time_array[self.event_time_array[:,1] == unique_time].astype(int)
            self.time_event_dict[unique_time] = events[:,0]
        
    
    def get_unique_agents(self):
        """Gets unique agents.
        """
        if self.network is not None:
            return np.unique(np.concatenate((self.network["i"], self.network["j"])))
        else:
            return np.arange(self.event_agent_array.shape[1])
    
    
    def set_rules(self, rules, agents, agent_types):
        """Set event rules and agent types.

        Args:
            rules (dict): rules of how many agents must be present of each type at an event.
            agents (list(tuples)): agent types of tuple (agent id, agent type).
            agent_types (list): list of available agent types
        """
        self.rules = rules
        self.agents = np.array(agents)
        self.agent_types = agent_types
        
    
    def swap_agent_id(self, t, id_1, id_2):
        """Swap agent ids at a point in time for the future events. 
        This is done by swapping the agent ids in events in the future, so by changing the current and future schedule.

        Args:
            tnet (tn.TemporalNetwork): The temporal network.
            t (float): time step at which agents should be swapped.
            id_1 (_type_): agent 1. 
            id_2 (_type_): agent 2.
        """
        
        self.network.loc[(self.network['i'] == id_1) & (self.network['t'] >= t), ['j']] = [id_2]
        self.network.loc[(self.network['j'] == id_2) & (self.network['t'] >= t), ['i']] = [id_1]
        pass
    
    
    def set_time_range(self):
        """
        Finds the time range between min and max time. Can only work if uniform sampling is true.
        """
        if self.uniform_sampling:
            time_array_1 = np.unique(self.network["t"])[:-1]
            time_array_2 = np.unique(self.network["t"])[1:]
            time_step_array = np.diff(np.unique(self.network["t"]))
            
            values, counts = np.unique(time_step_array, return_counts=True)
            most_frequent_timestep = values[np.argmax(counts)]
            
            time_step = most_frequent_timestep
            self.time_range = np.arange(self.network["t"].min(), self.network["t"].max() + time_step, time_step)
        else:
            try:
                self.time_range = np.unique(self.network.t)
            except:
                self.time_range = self.event_time_array[:, 1]
            
            
    def construct_agent_delays(self):
        """
        Constructs columns in the temporal network back end dataframe to keep track of agent delays.
        """
        # Construct agent delays, agent name list
        all_agent_delays_list = []
        for i in range(len(self.get_unique_agents())):
            self.network[f"agent_{i}_delay"] = 0
            all_agent_delays_list.append(f"agent_{i}_delay")
        all_agent_delays_array = np.array(all_agent_delays_list)
        self.all_agent_delays_array = all_agent_delays_array


    def construct_agent_delays_realtime(self):
        """Constructs columns in the temporal network back end dataframe to keep track of realtime agent delays.
        These delays are the "True" delays when the interactive_with_topology is set to true in process delay functions.
        """
        agent_delays_list = []
        for i in range(len(self.get_unique_agents())):
            self.network[f"agent_{i}_delay_realtime"] = 0
            agent_delays_list.append(f"agent_{i}_delay_realtime")
        agent_delays_array = np.array(agent_delays_list)
        self.all_agent_delays_array_realtime = agent_delays_array

    
    def prepare_delays(self):
        """Prepares delays, gives them 0 value.
        """
        try:
            np.nan_to_num(self.network["delay"], 0)  
            np.nan_to_num(self.network["current_event_delay"], 0)  
        except:
            self.network["delay"] = 0
            self.network["current_event_delay"] = 0
    
    
    def add_delay(self, event_id=0, event_id_range_bool = False, 
                  event_id_range = (0, 1), event_array_bool = False, 
                  event_array=np.array([0]), delay=1, random_bool=False, random_range=(0, 1),
                  wipe_delays = False, expon_distr_bool=False, tau=1, using_event_dict=False):
        """Adds a delay to a delay column in the temporal network Pandas.DataFrame format. Multiple options are possible to add delays.

        Args:
            event_id (int, optional): id where you want to add a delay, beit set by the delay parameter or via the random range. Defaults to 0.
            event_range (bool, optional): Bool to enable changing a range of events. Defaults to False.
            event_id_range (tuple, optional): Range of events you want to add delays to. Defaults to (0, 1).
            event_id_range_bool (bool, optional): Bool to set if a range of event ids should be used.
            delay (int, optional): delay size. Defaults to 1.
            random (bool, optional): Random option bool. Defaults to False.
            random_range (tuple, optional): Random range bool. Defaults to (0, 1).
            wipe_delays (bool, optional): Bool to set a wipe of delays. 
            expon_distr_bool (bool, optional): Set true for ubiquitous random delay sampled from an exponential distribution.
            tau (float) : lambda variable of exponential distribution.
            using_event_dict(bool, optional): Set true to use event dicts. Only supported for ubiquitous delay.
        """
        if wipe_delays:
            self.wipe_delays()
        if not using_event_dict:
            self.prepare_delays()
            pd.options.mode.chained_assignment = None
        bool_gatherer = (random_bool * 1, event_id_range_bool * 1, event_array_bool * 1, expon_distr_bool*1) 

        match bool_gatherer:
            case 0, 0, 0, 0:
                self.network["delay"] = np.where(self.network["event_id"] == event_id, delay, 0)
                if using_event_dict:
                    self.event_dict[event_id]["delay"] = delay
                    self.event_added_delay_array[event_id] = delay
                
            case 0, 1, 0, 0:
                self.network["delay"] = np.logical_and(self.network["event_id"] >= event_id_range[0], self.network["event_id"] < event_id_range[1]) * delay     
            case 0, 0, 1, 0:
                self.network["delay"][np.isin(self.network["event_id"].astype(int), event_array)] = delay
            case 1, 0, 0, 0:
                a, b = random_range
                delay = (b - a) * np.random.rand((1)) + a
                self.network["delay"] = np.where(self.network["event_id"] == event_id, delay, self.network["delay"])
            case 1, 1, 0, 0:
                a, b = random_range
                for event_iterator in range(event_id_range[0], event_id_range[1]):
                    delay = (b - a) * np.random.rand((1)) + a
                    self.network["delay"] = np.where(self.network["event_id"] == event_iterator, delay, self.network["delay"])
            case 1, 0, 1, 0:
                a, b = random_range
                for event_identifier in event_array:
                    delay = (b - a) * np.random.rand((1)) + a
                    self.network["delay"] = np.where(self.network["event_id"] == event_identifier, delay, self.network["delay"])
            case 0, 0, 0, 1:
                import collections
                delay = scipy.stats.expon.rvs(scale=1, loc=0, size = self.total_events)
                
                if using_event_dict:
                    for i in range(self.total_events):
                        self.event_dict[i]["delay"] = delay[i]
                    self.event_added_delay_array = delay
                else:
                    for event_iterator in self.network.event_id:
                        delay = scipy.stats.expon.rvs(scale=tau, loc=0, size = 1)
                        self.network["delay"] = np.where(self.network["event_id"] == event_iterator, delay, self.network["delay"])


    def add_agent_delay(self, agent_delay_array=[], event_id=0):
        """Add delays to agents only. Requires an array or list to be passed which holds the specific delays for each agent. 

        Args:
            agent_delay_array (list/np.array, optional): agent delay list. Defaults to [].
            event_id (int, optional): event id. Defaults to 0.
        """
        self.prepare_delays()
        network_at_event = self.network[self.network["event_id"]==event_id]
        time_at_event = network_at_event.t.max()
        
        for i in self.unique_agents.astype(int):
            self.network.loc[self.network["t"] >= time_at_event, self.all_agent_delays_array[i]] = agent_delay_array[i]
        
        self.delay_magnitude = np.max(np.array(agent_delay_array))

    
    def wipe_delays(self):
        """
        Sets all delays to 0.
        """
        self.network["delay"] = 0
        
        
    def wipe_buffers(self):
        """
        Sets all buffers to 0.
        """
        self.network["buffer"] = 0


    def add_event_buffer(self, event_id=0, event_id_range_bool = False, 
                  event_id_range = (0, 1), event_array_bool = False, 
                  event_array=np.array([0]), buffer=1, random=False, random_range=(0, 1),
                  wipe_buffers = False, uniform_buffer_bool=False, using_event_dict=False):
        """Adds a buffer to a delay column in the temporal network Pandas.DataFrame format. Multiple options are possible to add buffers.

        Args:
            event_id (int, optional): id where you want to add a delay, beit set by the delay parameter or via the random range. Defaults to 0.
            event_range (bool, optional): Bool to enable changing a range of events. Defaults to False.
            event_id_range (tuple, optional): Range of events you want to add delays to. Defaults to (0, 1).
            buffer (int, optional): buffer size. Defaults to 1.
            random (bool, optional): _description_. Defaults to False.
            random_range (tuple, optional): _description_. Defaults to (0, 1).
            uniform_buffer_bool (bool, optional): toggle to set a uniform buffer everywhere.
            using_event_dict(bool, optional): Set true to use event dicts. Only supported for uniform buffers.
        """
        self.buffer_budget = 0
        if wipe_buffers:
            self.network["buffer"] = 0
        if not using_event_dict:
            try:
                np.nan_to_num(self.network["buffer"], 0)  
            except:
                self.network["buffer"] = 0
        bool_gatherer = (random * 1, event_id_range_bool * 1, event_array_bool * 1, uniform_buffer_bool * 1) 

        match bool_gatherer:
            case 0, 0, 0, 0:
                self.network["buffer"] = np.where(self.network["event_id"] == event_id, buffer, 0)
            case 0, 1, 0, 0:
                self.network["buffer"] = np.where(np.logical_and(self.network["event_id"] > event_id_range[0], self.network["event_id"] < event_id_range[1]), buffer, self.network["buffer"])
            case 0, 0, 1, 0:
                self.network["buffer"][np.isin(self.network["event_id"].astype(int), event_array)] = buffer
            case 0, 0, 0, 1:
                if using_event_dict:
                    for i in range(self.total_events):
                        self.event_dict[i]["buffer"] = buffer
                        self.buffer_budget += buffer
                    try:
                        self.event_buffer_array = np.ones(self.event_buffer_array.shape) * buffer
                    except:
                        pass
                else:
                    self.network["buffer"] = buffer
                
            case 1, 0, 0, 0:
                a, b = random_range
                self.network["buffer"] = np.where(self.network["event_id"] == event_id, (b - a) * np.random.rand(len(self.network)) + a, self.network["buffer"])
            case 1, 1, 0, 0:
                a, b = random_range
                len_events = len(np.logical_and(self.network["event_id"] > event_id_range[0], self.network["event_id"] < event_id_range[1]))
                self.network["buffer"] = np.where(np.logical_and(self.network["event_id"] > event_id_range[0], self.network["event_id"] < event_id_range[1]), (b - a) * np.random.rand(len_events) + a, self.network["buffer"])
            case 1, 0, 1, 0:
                a, b = random_range
                len_events = len(np.isin(self.network["event_id"], event_array))
                self.network["buffer"][np.isin(self.network["event_id"], event_array)] = (b - a) * np.random.rand(len_events) + a
            case 1, 0, 0, 1:
                a, b = random_range
                self.network["buffer"] = (b - a) * np.random.rand(len(self.network)) + a
        pass
    
    
    def process_delays(self, interact_with_topology=True, return_current_delays=False, delay_colors=False, bar=False):
        """
        Processes / propagates delays and buffers throughout the network.
        """
        if delay_colors:
            self.construct_agent_delay_colors()
        if bar:
            pbar = tqdm(total=len(np.unique(self.network.event_id)))
        # Loop over all events, assign the proper delays to each agent.
        for unique_event_iterator in range(len(np.unique(self.network.event_id))):
            # Take the df slice of the event
            pd.options.mode.chained_assignment = None
            network_at_event = self.network[self.network["event_id"]==unique_event_iterator]
            
            # Gather all unique agents
            unique_agents_at_event = np.unique(np.concatenate((network_at_event["i"], network_at_event["j"])))
            
            # Get the delay we need.
            max_agent_delay_in_event = np.max(network_at_event[self.all_agent_delays_array[unique_agents_at_event.astype(int)]].max())
            buffer_in_event = network_at_event["buffer"].iloc[0] # Take the first buffer we find
            delay_minus_buffer = max_agent_delay_in_event - buffer_in_event
            
            #Find added delay
            added_delay = self.network["delay"].loc[self.network["event_id"] == unique_event_iterator].max() # Take the first delay we find at the event

            # Adjust delays, and timesteps of each event
            # Delay of an event is the delay of the most delayed agent, cannot be lower than 0, the same can be said for the event time, however, it increases with the latest agent. It cannot decrease.
            self.network["current_event_delay"].loc[self.network["event_id"] == unique_event_iterator] = max(0, delay_minus_buffer) + added_delay
            if interact_with_topology:
                self.network["t"].loc[self.network["event_id"] == unique_event_iterator] += max(0, delay_minus_buffer) + added_delay
        
            # Change agent delays for the future according to the current max delay at the event.
            #network_at_event_after_delay_propagation = self.network[self.network["event_id"]==unique_event_iterator]
            
            event_time = network_at_event["t"].max()
            self.network.loc[self.network["t"] >= event_time, self.all_agent_delays_array[unique_agents_at_event.astype(int)]] = max(0, delay_minus_buffer) + added_delay
            if bar:
                pbar.update(1)
                pbar.refresh()
            # new_delay = max(0, delay_minus_buffer) + added_delay
            # event_time_delayed = event_time + new_delay
            
        if return_current_delays:
            for idx, row in self.network.iterrows():
                i_agent = self.all_agent_delays_array_realtime[int(row['i'])]
                j_agent = self.all_agent_delays_array_realtime[int(row['j'])]
                self.network.loc[self.network["t"] >= row["t"], i_agent] = row['current_event_delay']
                self.network.loc[self.network["t"] >= row["t"], j_agent] = row['current_event_delay']

        self.set_time_range()
    
    
    def process_delays_fast(self, interact_with_topology=True, delay_colors=False, bar=False):
        """
        Processes / propagates delays and buffers throughout the network, uses the event dictionaries for faster processing.
        """
        if delay_colors:
            self.construct_agent_delay_colors()
        if bar:
            pbar = tqdm(total=len(self.event_dict))
        #self.build_event_dict()
        # Loop over all events, assign the proper delays to each agent.
        for unique_event_iterator in range(len(self.event_dict)):
            # Take the df slice of the event
            pd.options.mode.chained_assignment = None
            current_event_dict = self.event_dict[unique_event_iterator]
            
            # Gather all unique agents
            
            #unique_agents_at_event = np.unique(np.concatenate((network_at_event["i"], network_at_event["j"])))
            unique_agents_at_event = current_event_dict["agents"]
            
            # Get the delay we need.
            agent_delays_at_event = self.agent_delays[unique_event_iterator, unique_agents_at_event]
            max_agent_delay_in_event = np.max(agent_delays_at_event)
            buffer_in_event = current_event_dict["buffer"]
            delay_minus_buffer = max_agent_delay_in_event - buffer_in_event
            
            # Find added delay
            added_delay = current_event_dict["delay"]

            # Adjust delays, and timesteps of each event
            # Delay of an event is the delay of the most delayed agent, cannot be lower than 0, the same can be said for the event time, however, it increases with the latest agent. It cannot decrease.
            current_event_dict["current_event_delay"] = max(0, delay_minus_buffer) + added_delay
            event_time = current_event_dict["t"]

            if interact_with_topology:
                self.event_dict[unique_event_iterator]["t"] += max(0, delay_minus_buffer) + added_delay
                self.event_time_array[self.event_time_array[:, 0] == unique_event_iterator, 1] += max(0, delay_minus_buffer) + added_delay

            # Change agent delays for the future according to the current max delay at the event.
            #network_at_event_after_delay_propagation = self.network[self.network["event_id"]==unique_event_iterator]
            
            if interact_with_topology:
                events_later_than_current_event = np.nonzero(self.event_time_array[:,1] >= event_time)
                self.event_time_array = self.event_time_array[self.event_time_array[:, 1].argsort()] # Could be quicker by using something that sorts only one row of the array I think.
                self.agent_delays[events_later_than_current_event[0][:, np.newaxis], unique_agents_at_event] = max(0, delay_minus_buffer) + added_delay
            else:
                self.agent_delays[unique_event_iterator:,unique_agents_at_event] = max(0, delay_minus_buffer) + added_delay
                
            if bar:
                pbar.update(1)
                pbar.refresh()

    
    
    def process_delays_fast_arrays(self, interact_with_topology=True, delay_colors=False, bar=False):
        """
        Processes / propagates delays and buffers throughout the network. Uses event arrays for faster processing.
        """
        if delay_colors:
            self.construct_agent_delay_colors()
        if bar:
            pbar = tqdm(total=self.event_time_array.shape[0])
        #self.build_event_dict()
        # Loop over all events, assign the proper delays to each agent.
        for unique_event_iterator in range(self.event_time_array.shape[0]):
            # Take the df slice of the event
            pd.options.mode.chained_assignment = None
            
            # Gather all unique agents
            
            #unique_agents_at_event = np.unique(np.concatenate((network_at_event["i"], network_at_event["j"])))
            unique_agents_at_event = self.unique_agents[np.nonzero(self.event_agent_array[unique_event_iterator, :].astype(int))]
            
            # Get the delay we need.
            agent_delays_at_event = self.agent_delays[unique_event_iterator, unique_agents_at_event]

            max_agent_delay_in_event = np.max(agent_delays_at_event)
            buffer_in_event = self.event_buffer_array[unique_event_iterator]
            delay_minus_buffer = max_agent_delay_in_event - buffer_in_event
            
            # Find added delay
            added_delay = self.event_added_delay_array[unique_event_iterator]

            # Adjust delays, and timesteps of each event
            # Delay of an event is the delay of the most delayed agent, cannot be lower than 0, the same can be said for the event time, however, it increases with the latest agent. It cannot decrease.
            self.event_current_delay_array[unique_event_iterator] = max(0, delay_minus_buffer) + added_delay
            event_time = self.event_time_array[unique_event_iterator,1]

            if interact_with_topology:
                self.event_dict[unique_event_iterator]["t"] += max(0, delay_minus_buffer) + added_delay
                self.event_time_array[self.event_time_array[:, 0] == unique_event_iterator, 1] += max(0, delay_minus_buffer) + added_delay

            # Change agent delays for the future according to the current max delay at the event.
            #network_at_event_after_delay_propagation = self.network[self.network["event_id"]==unique_event_iterator]
            
            if interact_with_topology:
                events_later_than_current_event = np.nonzero(self.event_time_array[:,1] >= event_time)
                self.event_time_array = self.event_time_array[self.event_time_array[:, 1].argsort()] # Could be quicker by using something that sorts only one row of the array I think.
                self.agent_delays[events_later_than_current_event[0][:, np.newaxis], unique_agents_at_event] = max(0, delay_minus_buffer) + added_delay
            else:
                self.agent_delays[unique_event_iterator:,unique_agents_at_event] = max(0, delay_minus_buffer) + added_delay
                
            if bar:
                pbar.update(1)
                pbar.refresh()
                
            # new_delay = max(0, delay_minus_buffer) + added_delay
            # event_time_delayed = event_time + new_delay
            
            ### TODO implement delay colors so that they are colored by the max delayed agent.
            # if delay_colors:
            #     self.network[self.all_agent_delays_array_colors[unique_agents_at_event]] = 
            
            
    def process_delays_two_level_vect(self):
        """
        Processes / propagates delays and buffers throughout the network. WIP.
        """
        
        # Loop over all events, assign the proper delays to each agent.
        for time in self.time_event_dict.keys():
            
            # Take the df slice of the event
            pd.options.mode.chained_assignment = None
            
            # Get all events
            events_at_time_idx = self.time_event_dict[time].astype(int)
            
            # Gather all agents at the events
            agents_at_events = np.nonzero(self.event_agent_array[events_at_time_idx, :].astype(int))

            # Get delays at events
            agent_delays_at_event = self.agent_delays[events_at_time_idx, :][:,agents_at_events[1]]

            # Take row-wise max
            max_agent_delays_in_events = np.max(agent_delays_at_event, axis=1)
            buffer_in_event = self.event_buffer_array[events_at_time_idx]
            #print(max_agent_delays_in_events)
            # Find propagated delay
            delay_minus_buffer = max_agent_delays_in_events - buffer_in_event

            # Find added delay
            added_delay = self.event_added_delay_array[events_at_time_idx]
            
            # Add delay to max(0, new delay)
            new_delays = np.where(delay_minus_buffer > 0, delay_minus_buffer, 0) + added_delay
            #print(new_delays)
            # Adjust delays, and timesteps of each event
            self.event_current_delay_array[events_at_time_idx] = new_delays
            
            ### Indexing for reassigning the delays.
            mask = np.zeros_like(agent_delays_at_event)
            event_agent_idx = self.event_agent_array[events_at_time_idx].astype(bool)

            num_rows, num_cols = event_agent_idx.shape

            repeated_values = np.repeat(new_delays[:, np.newaxis], num_cols, axis=1)
            mask[event_agent_idx] = repeated_values[event_agent_idx]

            mask_for_indexing = np.zeros_like(self.agent_delays)
            mask_for_indexing[events_at_time_idx] = mask 

            flattened_array = np.max(mask, axis=0)
            flattened_array = flattened_array.reshape((flattened_array.shape[0], 1)).T

            vals = {}
            for i in range(len(flattened_array.flatten())):
                val = flattened_array.flatten()[i]
                try: vals[str(val)] += 1
                except: vals[str(val)] = 1
            if self.agent_delays.shape[0]-events_at_time_idx[-1]-1 != 0:
                #print(mask_for_indexing[events_at_time_idx])
                flattened_array = np.repeat(flattened_array, self.agent_delays.shape[0]-events_at_time_idx[-1]-1, axis=0)
                reshaped_max_delays = flattened_array.reshape((self.agent_delays.shape[0]-events_at_time_idx[-1]-1, -1))
            
                self.agent_delays[events_at_time_idx] = np.where(mask_for_indexing[events_at_time_idx] > 0, mask_for_indexing[events_at_time_idx], self.agent_delays[events_at_time_idx])
                self.agent_delays[events_at_time_idx[-1]+1:] = np.where(reshaped_max_delays > 0, reshaped_max_delays, self.agent_delays[events_at_time_idx[-1]+1:])

    
    def get_agent_delay(self):
        """Returns an array of shape (total events, 1 + #agents) where the first column holds times t of events, the second column holds agent delay arrays.

        Returns:
            time_agent_delays: Array of [t, [agent delay1, agent delay2, ... , agent delayN]]
        """
        time_agent_delays = np.zeros((self.total_events, 1 + len(self.unique_agents)))
        for i in range(self.total_events):
            t = self.event_dict[i]["t"]
            agent_delays = self.agent_delays[i]
            time_agent_delays[i,0] = t
            time_agent_delays[i,1:] = agent_delays
        return time_agent_delays
    
        
    def process_delays_with_rules(self, swap_cost_function):
        """
        Processes / propagates delays and buffers throughout the network with event rules enabled. 
        This means that agent swapping can happen. This algorithm swaps agents in a greedy manner.
        """
        # Loop over all events, assign the proper delays to each agent.
        
        if self.rules == None:
            print("No rules are set. Cannot propagate delays with event rules without rules, use DelayBufferNetwork.set_rules()")
        
        
        #print("Processing delays")
        for unique_event_iterator in range(len(np.unique(self.network.event_id))):
            # Take the df slice of the event
            pd.options.mode.chained_assignment = None
            network_at_event = self.network[self.network["event_id"]==unique_event_iterator]
            
            # Gather all unique agents
            agents_at_event = np.unique(np.concatenate((network_at_event["i"], network_at_event["j"])))
            
            # Get the delay we need.
            max_agent_delay_in_event = np.max(network_at_event[self.all_agent_delays_array[agents_at_event]].max())
            max_agent_delay = network_at_event[self.all_agent_delays_array].max().idxmax()
            
            
            # Check if max delay can be lowered by swapping agents.
            event_time = network_at_event.t.max()
            network_at_time = self.network[self.network["t"] == event_time]
            agents_free_ids = np.nonzero(np.isin(self.unique_agents, network_at_time, invert=True))[0]
            
            
            ### SWAP HANDLING
            # Checking for agent swapping
            for a_type in self.agent_types:
                # See which agents can potentially be swapped with free agents
                a_type_at_event = [x for x in agents_at_event if self.agents[x][1] == a_type]
                a_type_free = [x for x in agents_free_ids if self.agents[x][1] == a_type]
                #swapped = False
                #max_agent_delay_of_type = network_at_event[agents_of_type_at_event].max().idxmax()
                for agent_at_event in a_type_at_event:
                    # Find current agent delay
                    agent_delay = network_at_event[self.all_agent_delays_array[agent_at_event]].max()
                    # Break if there is no delay
                    if agent_delay == 0:
                        break
                    
                    # Find the least delayed free agent.
                    agent_delays_free = network_at_event[self.all_agent_delays_array[a_type_free]].min()
                    least_delayed_free_agent = a_type_free[agent_delays_free.min().argmin()]
                    smallest_delay = network_at_event[self.all_agent_delays_array[least_delayed_free_agent]].iloc[0]

                    if smallest_delay + swap_cost_function < agent_delay:
                        self.swap_agent_id(event_time, least_delayed_free_agent, agent_at_event)
                        a_type_at_event.remove(agent_at_event)
                        a_type_at_event.append(least_delayed_free_agent)
                        a_type_free.remove(least_delayed_free_agent)
                        a_type_free.append(agent_at_event)
                        print(f"Swapped agents {least_delayed_free_agent} and {agent_at_event} at time {event_time}")

            buffer_in_event = network_at_event["buffer"].iloc[0] # Take the first buffer we find
            
            delay_minus_buffer = max_agent_delay_in_event - buffer_in_event
            
            #Find added delay
            added_delay = self.network["delay"].loc[self.network["event_id"] == unique_event_iterator].max() # Take the first delay we find at the event

            # Adjust delays, and timesteps of each event
            # Delay of an event is the delay of the most delayed agent, cannot be lower than 0, the same can be said for the event time, however, it increases with the latest agent. It cannot decrease.
            self.network["current_event_delay"].loc[self.network["event_id"] == unique_event_iterator] = max(0, delay_minus_buffer + added_delay)
            self.network["t"].loc[self.network["event_id"] == unique_event_iterator] += max(0, delay_minus_buffer + added_delay)
            
            # Change agent delays for the future according to the current max delay at the event.
            #network_at_event_after_delay_propagation = self.network[self.network["event_id"]==unique_event_iterator]
            event_time = network_at_event["t"].max()
            self.network.loc[self.network["t"] >= event_time, self.all_agent_delays_array[agents_at_event]] = max(0, delay_minus_buffer + added_delay)
        self.set_time_range()
    
    
    def get_average_delays(self):
        """Gets mean delays over the system. Uses the real time delays.
        """
        avg_delay = []
        try:
            for i in self.time_range:
                curr_netw = self.network[self.network["t"] == i]
                agent_array = curr_netw[self.all_agent_delays_array_realtime].to_numpy()
                avg_delay.append(np.mean(agent_array[0,:]))
        except IndexError:
            for i in np.unique(self.network.t):
                curr_netw = self.network[self.network["t"] == i]
                agent_array = curr_netw[self.all_agent_delays_array_realtime].to_numpy()
                avg_delay.append(np.mean(agent_array[0,:]))
            
        return avg_delay
    
    
    def check_event_order_violation(self):
        """
        Function to check whether the order of events has been violated. Does not matter for some networks but this should not be allowed in a transport network.
        """
        events = self.network["event_id"]
        for i in range(len(events)-1):
            network_at_event1 = self.network[self.network["event_id"] == events.iloc[i]]
            network_at_event2 = self.network[self.network["event_id"] == events.iloc[i+1]]

            if events.iloc[i] > events.iloc[i+1]:
                if any(np.unique(network_at_event1["i"])) in network_at_event2:
                    print("Event order has been violated.")
                    self.violated = True
        return


    def add_self_edges(self):
        """
        Generates self edges on the temporal network.
        """
        for unique_event_iterator in np.unique(self.network["event_id"]):
            network_at_event = self.network[self.network["event_id"] == unique_event_iterator]
            time_at_event = network_at_event["t"].max()
            weight_at_event = network_at_event["weight"].max()
            agents_at_event = np.unique(network_at_event["i"])
            for i in agents_at_event:
                self.add_edge([i, i, time_at_event, weight_at_event, unique_event_iterator])
        self.edges = True
        self.network = self.network.sort_values("t")
        self._update_network()
      
        
    def plot_delay_propagation_with_gaussian_smoothening(self, sigma, show=True):
        """
        Generate a plot of the normalized average agent delay over time.
        
        Args:
            show (bool, optional): toggle to show the plot
        """
        self.network = self.network.sort_values(by=["t", "event_id"])
        x = self.network["t"]
        all_agent_delays = np.sum(self.network[self.all_agent_delays_array], axis = 1) / (self.N)
        
        plt.figure(figsize=(10,10))
        plt.plot(x, all_agent_delays, label="Average agent delay")
        plt.plot(x, np.ones((x.shape)) * self.delay_magnitude, linestyle="dotted", label="Delay magnitude", color="grey", alpha = 0.7)
        plt.title(f"Average agent delay per unit time with delay magnitude: {self.delay_magnitude}")
        plt.legend()
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("Delay")
        if show:
            plt.show()


    def plot_delay_variance(self, show=True):
        """
        Plots delay variance over time.

        Args:
            show (bool, optional): toggle to show the plot
        """
        x = self.network["t"]
        all_agent_std_dev = np.std(self.network[self.all_agent_delays_array], axis = 1) / (self.delay_magnitude)
        
        fig = plt.figure(figsize=(10,10))
        plt.plot(x, all_agent_std_dev, label="Agent delay standard deviation")
        plt.title(f"Agent delay standard deviation over time, divided by delay magnitude: {self.delay_magnitude}")
        plt.grid()
        plt.xlabel("Time (s)")
        plt.ylabel("Standard deviation of agent delays divided by delay magnitude")
        if show:
            plt.show()

      
    def DBN_graph(self, delta_t: float, lookback = False, bar = False) -> nx.DiGraph():
        """Returns a network representation of the event structure of a self.

        Args:
            dbn (DelayBufferNetwork): Input dbn
            lookback (bool): boolean to set if you want nodes to point to previous nodes instead of next ones.
        Returns:
            G (nx.DiGraph): A graph which has all the events as nodes, and the edges are edges between the nodes. Directedness is given by the time dependence.
            Directedness can be flipped to have a network where all nodes point to previous nodes.
        """
        dbn_graph = {}
        if bar:
            pbar = tqdm(total=len(np.unique(self.network.event_id)))
            
        # For all events.
        for event_iterator in np.unique(self.network.event_id):
            G = nx.DiGraph()
            events_connected = []
            # Gather data.
            network_at_event = self.network[self.network["event_id"]==event_iterator]
            agents_at_event = np.unique(np.concatenate((network_at_event["i"], network_at_event["j"])))
            buffer_in_event = network_at_event["buffer"].iloc[0]
            event_time = network_at_event["t"].max()
            
            df_within_dt = (self.network['t'] < event_time + delta_t) & (self.network['t'] > event_time - delta_t)
            # For all agents at event, find events within time window, add to list.
            for agent in agents_at_event:
                mask = (self.network['i'] == agent) & df_within_dt
                df_events_in_dt = self.network.loc[mask]
                if len(df_events_in_dt != 0):
                    unique_events_in_time_window = np.unique(df_events_in_dt.event_id)
                    for connected_event_id in unique_events_in_time_window:
                        if connected_event_id != event_iterator:
                            events_connected.append(connected_event_id)

            # Add all connection of this event as nodes and edges connected to this event.
            G.add_node(event_iterator)
            G.nodes[event_iterator]["buffer"] = buffer_in_event
            for event_id in events_connected:
                if lookback:
                    G.add_edge(event_iterator, event_id)
                else:
                    G.add_edge(event_id, event_iterator)
            dbn_graph[event_iterator] = G
            if bar:
                pbar.update(1)
                pbar.refresh()
        return dbn_graph


    def event_centralities(self, delta_t: float, event_graphs=None, lookback = False, bar = False) -> dict:
        """Calculates each nodes' centrality by just counting how many nodes there are in their own connection graph.

        Args:
            self (DelayBufferNetwork): _description_
            delta_t (float): _description_
            lookback (bool, optional): _description_. Defaults to False.
            bar (bool, optional): _description_. Defaults to False.

        Returns:
            dict: dictionary filled with centrality values for each node.
        """
        centralities = {}
        if event_graphs == None:
            event_graphs = self.DBN_graph(delta_t, lookback, bar)
        for k,v in event_graphs.items():
            cent = len(event_graphs[k].nodes)
            centralities[k] = cent
        self.centralities = centralities
      
                    
    def distribute_buffers_according_to_centrality(self, buffer_budget: float, delta_t: float, higher_than_mean: float = 0, using_event_dict=False):
        """Distributes buffers according to temporal degree centrality.

        Args:
            buffer_budget (float): buffer budget to distribute
            delta_t (float): delta t within which the temporal centrality is defined.
        """
        self.wipe_buffers()
        if self.centralities is None:
            self.event_centralities(delta_t)
        centralities = self.centralities
        
        if (higher_than_mean)>0:
            avg_centrality = np.average([*centralities.values()]) * higher_than_mean
            centralities = {key:0 if val < avg_centrality else val for key, val in centralities.items()}

        total_conn = sum(centralities.values())
        buffer_per_conn = buffer_budget / total_conn
        for event_id in np.unique(self.network.event_id):
            buffer_assigned_to_event = centralities[event_id] * buffer_per_conn
            self.network["buffer"] = np.where(self.network["event_id"] == event_id, buffer_assigned_to_event, self.network["buffer"])
        
        if using_event_dict:
            for event_id in np.unique(self.network.event_id):
                self.event_dict[event_id]["buffer"] = centralities[event_id] * buffer_per_conn
                self.event_buffer_array[event_id] = centralities[event_id] * buffer_per_conn

        pass