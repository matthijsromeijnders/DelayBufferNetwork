# Author: Matthijs Romeijnders ~ 2023. ~95% attribution to University of Utrecht, ~5% independent.

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import pandas as pd
from tqdm import tqdm
import warnings
import networkx as nx
import pickle
warnings.filterwarnings("ignore")

class DelayBufferNetwork():
    def __init__(self, from_df = None, random_event_dict=False, load=False, path="", system="dict",
                 dont_build_df=False, del_df=False, uniform_time_range=False):
        """Init

        Args:
            from_df (pandas DF, optional): init from schedule dataframe. Defaults to None.
            random_event_dict (bool, optional): generate random event dict, dont initialize from data. Defaults to False.
            load (bool, optional): load DBN from path. Defaults to False.
            path (_type_, optional): path to load DBN from. Defaults to "".
            system(str, optional): Select which system you want to use. Options are "dict", "df", and "array"
            dont_build_df (bool, optional): Set to not copy DF, saves memory for large DF. Defaults to False.
            del_df (bool, optional): Del DF after init, saves memory for large DF. Defaults to False.
            uniform_time_range (bool, optional): Set True if time steps are of a constant size, makes measures quicker. Defaults to False.
        """
        if not random_event_dict:
            if load:
                self.load_event_arrays(path)
                self.load_dicts(path)
            if from_df is not None:
                self.network = from_df.copy()
            else:
                self.network = None  
            self.dict = True
            self.df = False
            self.array = False
            
            self.system = system
            if system == "df":
                self.df = True
                self.dict = False
            if system == "array":
                self.array = True 
                self.dict = False   
                
            self.unique_agents = self.get_unique_agents()
            self.last_event_list = self.unique_agents
            self.N_agents = len(self.unique_agents)
            self.propagator_matrices = []
            self.centralities = None
            self.uniform_sampling = uniform_time_range
            self.time_range = None
            self.buffer_budget = 0
            self.event_dict = {}
            self.time_event_dict = {}
            self.total_events = None
            self.agent_delays = None
            self.event_time_array = None
            self.event_added_delay_array = None
            self.event_buffer_array = None
            self.delay_magnitude = 0
            
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
        self.set_time_range()
    
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
        with open(path+"_time_dict.pkl", "rb") as f:
            self.time_event_dict = pickle.load(f)

        
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

        for _, row in self.network.iterrows():
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
        """Build a random dictionary, and event time array.
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
    
    
    def set_time_range(self):
        """
        Finds the time range between min and max time. Can only work if uniform sampling is true.
        """
        if self.uniform_sampling:
            time_step_array = np.diff(np.unique(self.network["t"]))
            
            values, counts = np.unique(time_step_array, return_counts=True)
            most_frequent_timestep = values[np.argmax(counts)]
            
            time_step = most_frequent_timestep
            self.time_range = np.arange(self.network["t"].min(), self.network["t"].max() + time_step, time_step)
        else:
            if self.df:
                self.time_range = np.unique(self.network.t)
            else:
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
        except KeyError:
            self.network["delay"] = 0
            self.network["current_event_delay"] = 0
    
    
    def add_delay(self, tau=1):
        """Adds a delay to a delay column in the temporal network Pandas.DataFrame format. Multiple options are possible to add delays.

        Args:
            tau (float) : lambda variable of exponential distribution.
            using_dict(bool, optional): Set true to add the delays to the dict-based graph system. Defaults to True.
            using_df(bool, optional): Set true to add the delays to the array-based system. Defaults to False.
            using_df(bool, optional): Set true to add the delays to the DF system. Defaults to False.
        """
        # Add delays to dict-based graph system.
        if self.dict:
            delay = scipy.stats.expon.rvs(scale=1, loc=0, size = self.total_events)
            for i in range(self.total_events):
                self.event_dict[i]["delay"] = delay[i]
        
        # Add delays to array-based system.
        if self.array:
            delay = scipy.stats.expon.rvs(scale=1, loc=0, size = self.total_events)
            self.event_added_delay_array = delay

        # Add delays to the DF.
        if self.df:
            self.prepare_delays()
            for event_iterator in self.network.event_id:
                delay = scipy.stats.expon.rvs(scale=tau, loc=0, size = 1)
                self.network["delay"] = np.where(self.network["event_id"] == event_iterator, delay, self.network["delay"])


    def add_agent_delay(self, agent_delay_array, event_id=0):
        """Add delays to agents only. Requires an array or list to be passed which holds the specific delays for each agent. 

        Args:
            agent_delay_array (list/np.array, optional): agent delay list.
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


    def add_event_buffer(self, buffer, epsilon = 0):
        """Adds a buffer to a delay column in the temporal network Pandas.DataFrame format. Multiple options are possible to add buffers.

        Args:
        buffer (float) : Buffer size.
        epsilon (float) : (DOES NOT WORK FOR DF SYSTEM) Scale of uniform distribution around which buffers are sampled, such that buffers are sampled from a uniform distribution between B-epsilon and B+epsilon.
        using_dict(bool, optional): Set true to add the delays to the dict-based graph system. Defaults to True.
        using_df(bool, optional): Set true to add the delays to the array-based system. Defaults to False.
        using_df(bool, optional): Set true to add the delays to the DF system. Defaults to False.
        """
        self.buffer_budget = 0
        if self.dict:
            for i in range(self.total_events):
                buffer_per_event = max(0, np.random.uniform(buffer-epsilon, buffer+epsilon))
                self.event_dict[i]["buffer"] = buffer
                self.buffer_budget += buffer_per_event
                self.event_buffer_array[i] = buffer_per_event
     
        if self.array:
            if epsilon != 0:
                for i in range(self.total_events):
                    buffer_per_event = max(0, np.random.uniform(buffer-epsilon, buffer+epsilon, 1))
                    
                    self.event_buffer_array[i] = buffer_per_event
                    self.buffer_budget += buffer_per_event
            else:
                self.event_buffer_array = np.ones((self.event_buffer_array.shape)) * buffer
                self.buffer_budget = np.sum(self.event_buffer_array)
 
        if self.df:
            np.nan_to_num(self.network["buffer"], 0)  
            self.network["buffer"] = buffer
    
    
    def process_delays_df(self, dynamic_topology=True, return_current_delays=False, bar=False):
        """
        Processes / propagates delays and buffers throughout the network.
        """
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
            if dynamic_topology:
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
            for _, row in self.network.iterrows():
                i_agent = self.all_agent_delays_array_realtime[int(row['i'])]
                j_agent = self.all_agent_delays_array_realtime[int(row['j'])]
                self.network.loc[self.network["t"] >= row["t"], i_agent] = row['current_event_delay']
                self.network.loc[self.network["t"] >= row["t"], j_agent] = row['current_event_delay']

        self.set_time_range()
    
    
    def process_delays_dict(self, dynamic_topology=True, bar=False):
        """
        Processes / propagates delays and buffers throughout the network, uses the event dictionaries for faster processing.
        """
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

            if dynamic_topology:
                self.event_dict[unique_event_iterator]["t"] += max(0, delay_minus_buffer) + added_delay
                self.event_time_array[self.event_time_array[:, 0] == unique_event_iterator, 1] += max(0, delay_minus_buffer) + added_delay

            # Change agent delays for the future according to the current max delay at the event.
            #network_at_event_after_delay_propagation = self.network[self.network["event_id"]==unique_event_iterator]
            
            if dynamic_topology:
                events_later_than_current_event = np.nonzero(self.event_time_array[:,1] >= event_time)
                self.event_time_array = self.event_time_array[self.event_time_array[:, 1].argsort()] # Could be quicker by using something that sorts only one row of the array I think.
                self.agent_delays[events_later_than_current_event[0][:, np.newaxis], unique_agents_at_event] = max(0, delay_minus_buffer) + added_delay
            else:
                self.agent_delays[unique_event_iterator:,unique_agents_at_event] = max(0, delay_minus_buffer) + added_delay
                
            if bar:
                pbar.update(1)
                pbar.refresh()

    
    def process_delays_arrays(self, dynamic_topology=True, bar=False):
        """
        Processes / propagates delays and buffers throughout the network. Uses event arrays for faster processing.
        """
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

            if dynamic_topology:
                self.event_dict[unique_event_iterator]["t"] += max(0, delay_minus_buffer) + added_delay
                self.event_time_array[self.event_time_array[:, 0] == unique_event_iterator, 1] += max(0, delay_minus_buffer) + added_delay

            # Change agent delays for the future according to the current max delay at the event.
            #network_at_event_after_delay_propagation = self.network[self.network["event_id"]==unique_event_iterator]
            
            if dynamic_topology:
                events_later_than_current_event = np.nonzero(self.event_time_array[:,1] >= event_time)
                self.event_time_array = self.event_time_array[self.event_time_array[:, 1].argsort()] # Could be quicker by using something that sorts only one row of the array I think.
                self.agent_delays[events_later_than_current_event[0][:, np.newaxis], unique_agents_at_event] = max(0, delay_minus_buffer) + added_delay
            else:
                self.agent_delays[unique_event_iterator:,unique_agents_at_event] = max(0, delay_minus_buffer) + added_delay 
            if bar:
                pbar.update(1)
                pbar.refresh()

    
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
    
    
    def process_delays(self, dynamic_topology=False, loading_bar=False):
        """Selects a delay processing function that corresponds with the system that is being used.
        """
        if self.dict:
            self.process_delays_dict(dynamic_topology, bar=loading_bar)
        elif self.array:
            self.process_delays_arrays(dynamic_topology, bar=loading_bar)
        elif self.df:
            self.process_delays_df(dynamic_topology, bar=loading_bar)
    
    
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
      
        
    def plot_delay_propagation(self):
        """
        Generate a plot of the normalized average agent delay over time.
        
        Args:
            show (bool, optional): toggle to show the plot
        """
        self.network = self.network.sort_values(by=["t", "event_id"])
        x = self.network["t"]
        all_agent_delays = np.sum(self.network[self.all_agent_delays_array], axis = 1) / len(self.unique_agents)
        
        plt.figure(figsize=(10,10))
        plt.plot(x, all_agent_delays, label="Average agent delay")
        plt.plot(x, np.ones((x.shape)) * self.delay_magnitude, linestyle="dotted", label="Delay magnitude", color="grey", alpha = 0.7)
        plt.title(f"Average agent delay per unit time with delay magnitude: {self.delay_magnitude}")
        plt.legend()
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("Delay")
        plt.show()


    def plot_delay_variance(self, show=True):
        """
        Plots delay variance over time.

        Args:
            show (bool, optional): toggle to show the plot
        """
        x = self.network["t"]
        all_agent_std_dev = np.std(self.network[self.all_agent_delays_array], axis = 1) / (self.delay_magnitude)
        
        _ = plt.figure(figsize=(10,10))
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
        for k, _ in event_graphs.items():
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