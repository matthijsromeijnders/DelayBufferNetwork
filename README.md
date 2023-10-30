# DelayBufferNetwork
REQUIRES PYTHON 3.10 OR NEWER

DelayBufferNetworks are objects which allow the user to calculate delay propagation on a temporal network, based on a schedule.

The code in delaybuffernetwork.py contains the main DelayBufferNetwork object. 
Temporal network/Dynamic graph/Evolutionary graph data can be used to initialize a network, so long as it is in ijt-event format (Agent i is connected to agent j at time t at event event_id).
This data should be in a pandas DataFrame.

After initialization, one can add delays and buffers, and then process the delays.
The DataFrame system is highly interpretable and easy to use, but very slow for large networks. 
Two optimizations are included which are faster but make it harder to find what exactly happened at a singular event.
One is a dictionary-based graph system and the other is an array-based system. They provide less insight but are faster.
The default system is the dictionary-based graph system

Delays can be processed with or without dynamic temporal topology, where delayed events will actually take place later.
This has repercussions for general graph reachability as well as the speed at which delays can spread to other delays. 
Large delays have to "wait" before they can infect other events. Therefore it stretches the time scale of the schedule.
Note that this is effectively a doubly dynamic network, since a temporal network already has a dynamic topology over time.
Small buffers have very little effect in this system such that the "\mathcal{v} vs B" is nonlinear.
Dynamic topology delay propagation is untested for the array approach, could be incorrect.

Once an appropriate dataset is available (one can be generated with generate_data.py), A DBN can be initialized. 
Without data, a random schedule can also be simulated with DBN.build_random_event_dict(N, T, k, B), note that this only works with the graph system, and is not as free with data parameters as generating data.
Delays can be added to events individually, or exponential noise can be given to all. Similarly, buffers can be added individually to events, or a uniform buffer can be set with an optional uniform deviation (B+ \eps). Then, process_delays/process_delays_graph/process_delays_arrays can be called on the DBN. 

Afterwards, the effects can be inspected by analyzing DBN through variables associated with the system that was used.
For the dict-based graph system, this would be the the DBN.event_dict, which holds all the information on each event.
For the array system, this would be several arrays, such as event_time_array (holds the time that each event takes place), event_agent_array (agent ids for each event), and agent_delays (delay of each agent at each time step).

The DelayBufferNetMeasures.py file contains functions to calculate the entanglement entropy over the DBN. Entropy convergence time and entropy difference can also be calculated. ~ https://doi.org/10.1103/PhysRevE.105.054301

Event rules combined with agent swapping is not possible in this release. Event rules in general can be set in the data generation, and this works perfectly, agents of the same type just cannot be swapped if one is delayed in this version.
