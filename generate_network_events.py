import teneto as tn
from teneto import TemporalNetwork
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys
import warnings
import scipy.optimize
import os
import configparser
from DelayBufferNetwork import DelayBufferNetwork
from DelayBufferNetworkMeasures import *
import sys
import random
import scipy.stats as stats
import networkx as nx
import warnings
import scipy
warnings.filterwarnings("ignore")

def generate_network_events(agents: list,agent_types:list, time_range: tuple, time_step: float, event_rules: dict, events_per_timestep: int, random_events: bool=False, random_rule: bool =False):
    
    # agents list should hold tuples of (agent id, agent type)
    # event rules list should hold {agent type: number of agent types per event}
    
    time_range = np.arange(time_range[0], time_range[1], time_step)
    df = pd.DataFrame()
    events = 0
    edges = []
    
    
    events_at_t = int(events_per_timestep)
    for t in time_range:
        if random_events:
            events_at_t = scipy.stats.poisson.rvs(int(events_per_timestep))
            if events_at_t > len(agents)/event_rules[agent_types[0]]:
                events_at_t=int(len(agents)/event_rules[agent_types[0]])
            elif events_at_t == 0:
                events_at_t = 1
        agents_not_yet_picked = agents.copy()
        for event in range(events_at_t):
            agents_at_event = np.array([])
   
            for agent_type in agent_types:
                # Find how many agents we need of this type
                rule = event_rules[agent_type]
                if random_rule:
                    rule = np.random.choice(np.arange(2, rule))
                    
                # Select agents that have the correct type
                agents_of_type = np.array([agent[0] for agent in agents if agent[1] == agent_type and agent in agents_not_yet_picked])
                
                # Random choice the required agents from the agents of correct type
                # rule is the number of elements chosen, we cannot replace -> duplicate agents.
                # We use a uniform probability distribution of agent probabilities.
                agents_at_event_of_type = np.random.choice(agents_of_type, rule, replace=False)
                
                agents_at_event = np.append(agents_at_event, agents_at_event_of_type)
                for agent_at_event in agents_at_event_of_type:
                    agents_not_yet_picked.remove((agent_at_event, agent_type))

            weight = 1/(len(agents_at_event))    
            for agent1 in agents_at_event:
                for agent2 in agents_at_event:
                    if agent1 != agent2:
                        edges.append([int(agent1), int(agent2), t, weight, events])

            events += 1
    df = pd.DataFrame(edges, columns=["i", "j", "t", "weight", "event_id"])
    return df


def generate_network_events_meanfield_mimic(agents: list, time_range: tuple, time_step: float, k_neighbours: int):
    # agents list should hold tuples of (agent id, agent type)
    
    time_range = np.arange(time_range[0], time_range[1], time_step)
    
    events = 0
    weight = 1/(k_neighbours+1)
    
    edges, primary_agents = [], []
    
    for t in tqdm(time_range):


        for primary_agent in range(len(agents)):
            # At each timestep we make one event for every agent, with k other random agents
            primary_agent_array = np.array([int(primary_agent)])

            rule = k_neighbours
            
            
            # Random choice the required agents
            # rule is the number of agents chosen
            # We use a uniform probability distribution of agent probabilities.
            agents_at_event = np.random.choice(agents, rule, replace=False)
            agents_at_event = np.append(agents_at_event, primary_agent_array)
            
            for agent1 in agents_at_event:
                if agent1 != primary_agent:
                    edges.append([agent1, primary_agent, t, weight, events])
                    primary_agents.append([primary_agent, events])
                    
            events += 1
            
    df_edges = pd.DataFrame(edges, columns=["i", "j", "t", "weight", "event_id"])
    df_prims = pd.DataFrame(primary_agents, columns=["primary_agent", "event_id"])
    return df_edges, df_prims


def generate_network_events_meanfield_mimic_sametime_evaluation(agents: list, time_range: tuple, time_step: float, k_neighbours: int):
    # agents list should hold tuples of (agent id, agent type)
    
    time_range = np.arange(time_range[0], time_range[1], time_step)
    
    events = 0
    weight = 1/(k_neighbours+1)
    
    edges = []
    
    # Make sure every agent has at least 1 connection
    for agent1 in agents:
        edges.append([agent1, 0, 0, weight, events])
    
    
    for t in tqdm(time_range):
        # Random choice the required agents
        # rule is the number of agents chosen
        # We use a uniform probability distribution of agent probabilities.
        agents_at_event = agents
        
        # for agent1 in agents_at_event:
        #     for agent2 in agents_at_event:
        #         if agent1 != agent2:
        #             edges.append([agent1, agent2, t, weight, events])
                    
        # DUMMY EDGE
        edges.append([0, 1, t, weight, events])
                    
        events += 1
            
    df_edges = pd.DataFrame(edges, columns=["i", "j", "t", "weight", "event_id"])
    return df_edges

