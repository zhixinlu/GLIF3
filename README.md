# GLIF3
In this repo, we created 3 glif models using PyTorch

The traditional glif3 model, using 1st order Euler forward method. The time of spike is calculated within each time bin acurately. The evolution in each time bin is based on the actual spike-window-free time during each time bin. (see the following diagram)

...|_______|____----|--------|--------|----__--|--------|--------|------__|________|___-----|--------|--------|---_____| ... 
For each time bin, we determin when the previous spike, if exists, ends, and where a new spike begins. It is the time interval in between that is used for the 1st order Eular forward method. All the reset of Voltage, the spike-induced sudden change in after spike current and  the adaptive threshold induced by s, was done at the begining of each detected spike. In this method, we need to choose a time bin dt that is < than the least spike_window_length for all neurons.



Similarly, we added the nonlinear version glif3 and the nonlinear version glif5 model.
