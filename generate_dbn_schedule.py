import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import warnings
warnings.filterwarnings("ignore")




def generate_dbn_schedule(N: int, agent_types:list, time_range: tuple, time_step: float, event_rules: list, events_per_timestep: int, agent_numbers_of_type : dict = {},  randomness_events: bool=False, randomness_rule: bool =False):
    """Generates a schedule for a DBN. 

    Args:
        N (int): Number of agents
        agent_types (list): Different types of agents, eg ["conductor", "train", "train_driver"]\
        event_rule (list): Rule for how many agents of each type have to be present at an event, eg [3, 1, 1], idx should correspond with agent_types list.
        time_range (tuple): Range that time passes on the schedule, eg (0, 100)
        time_step (float): Time step size
        events_per_timestep (int): Number of events per time step. If randomness events is selected this will be treated as the scale of a Poisson distribution.
        agent_numbers_of_type (dict, optional): You can manually set the number of agents of each type that you would like with this dictionary, should be like {agent_type: N_type}
        randomness_events (bool, optional): If True every time step will have a random number of events, sampled from a Poisson distribution with scale = events_per_timestep. Defaults to False.
        randomness_rule (bool, optional): If True every event will have a random number of agents of each type, uniformly sampled between 1 and event_rule[i], except if there is only one type. Defaults to False.

    Returns:
        df (Pandas DataFrame): schedule of events and agents which can be processed with delaybuffernetwork.py.
    """
    # Construct agents list, which contains tuples of (agent id, type) (could have been a dict now that i think about it).
    if agent_numbers_of_type == {}:
        agents = []
        for i in range(N):
            agent_type = np.random.choice(agent_types, replace=False)
            agents.append((i, agent_type))
    else:
        agents = []
        for agent_type in agent_types:
            for i in range(agent_numbers_of_type[agent_type]):
                agents.append((i, agent_type))
    
    # Construct agent type - rule dictionary.
    event_rule_dict = {}
    for i, agent_type in enumerate(agent_types):
        event_rule_dict[agent_type] = event_rules[i]
    
    time_range = np.arange(time_range[0], time_range[1], time_step)
    df = pd.DataFrame()
    events = 0
    edges = []
    events_at_t = int(events_per_timestep)
    
    for t in time_range:
        if randomness_events:
            events_at_t = scipy.stats.poisson.rvs(int(events_per_timestep))
            if events_at_t > len(agents)/event_rule_dict[agent_types[0]]:
                events_at_t=int(len(agents)/event_rule_dict[agent_types[0]])
            elif events_at_t == 0:
                events_at_t = 1
        agents_not_yet_picked = agents.copy()
        for _ in range(events_at_t):
            agents_at_event = np.array([])
   
            for agent_type, rule in event_rule_dict.items():
                # Find how many agents we need of this type
                
                if randomness_rule:
                    if len(event_rule_dict > 1):
                        rule = np.random.choice(np.arange(1, rule))
                    else:
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
