# DelayBufferNetwork
DelayBufferNetworks are objects which allow the user to calcute delay propagation on a temporal network

The code in delaybuffernetwork.py contains the main DelayBufferNetwork object. 
Temporal network/Dynamic graph/Evolutionary graph data can be used to initialize a network, so long as it is in ijt-event format (Agent i is connected to agent j at time t at event event_id).
This data should be in a pandas DataFrame.

After initialization, one can add delays and buffers, and then process the delays.
The library contains two optimizations which remove the DataDrame after initialization and turn it into either dictionary form or numpy array form.

The delays can be processed with or without interactive topology (the delays displace events).
