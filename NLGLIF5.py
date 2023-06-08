import torch
from torch import nn
import matplotlib.pyplot as plt

class NLGLIF5(nn.Module):
    def __init__(self,
                 neuronal_model_paras,  # dictionary of parameters {'K_V': [K_V,K_V,....,K_V], 'R_V': ..., ...}
                 dt=0.002,  # length of a time bin (seconds). dt should be large than the shortest spike_len.
                 device='cuda'):

        super().__init__()
        self.n_neurons = len(neuronal_model_paras['C_inv'])
        self.n_ascs = 2 # two after spike current channels
        assert self.n_ascs == len(neuronal_model_paras['K_Ij'][0])
        assert self.n_neurons == len(neuronal_model_paras['K_Ij'])
        self.dt = dt # (seconds)
        self.device = device
         
        self.C_inv = nn.Parameter(torch.tensor(neuronal_model_paras['C_inv']), requires_grad=False)
        
        self.alpha = nn.Parameter(torch.tensor(neuronal_model_paras['alpha']), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(neuronal_model_paras['beta']), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(neuronal_model_paras['gamma']), requires_grad=False)
        
        self.V_rest = nn.Parameter(torch.tensor(neuronal_model_paras['V_rest']), requires_grad=False)
        
        self.V_reset_slope = nn.Parameter(torch.tensor(neuronal_model_paras['V_reset_slope']), requires_grad=False)
        self.V_reset_intcp = nn.Parameter(torch.tensor(neuronal_model_paras['V_reset_intcp']), requires_grad=False)
        
        self.V_threshold = nn.Parameter(torch.tensor(neuronal_model_paras['V_threshold']), requires_grad=False)
        self.K_theta_s = nn.Parameter(torch.tensor(neuronal_model_paras['K_theta_s']), requires_grad=False)
        self.A_theta_s = nn.Parameter(torch.tensor(neuronal_model_paras['A_theta_s']), requires_grad=False)
        self.K_theta_v = nn.Parameter(torch.tensor(neuronal_model_paras['K_theta_v']), requires_grad=False)
        self.A_theta_v = nn.Parameter(torch.tensor(neuronal_model_paras['A_theta_v']), requires_grad=False)
        
        self.spike_len = nn.Parameter(0.001*torch.tensor(neuronal_model_paras['spike_len']), requires_grad=False)
        
        self.K_Ij = nn.Parameter(torch.tensor(neuronal_model_paras['K_Ij']).T, requires_grad=False)
        self.R_Ij = nn.Parameter(torch.ones_like(torch.tensor(neuronal_model_paras['K_Ij'])).T, requires_grad=False)
        self.A_Ij = nn.Parameter(torch.tensor(neuronal_model_paras['A_Ij']).T, requires_grad=False)
        
        self.to(device)
        if torch.min(self.spike_len) <= self.dt:
            raise Exception('Error: dt is too small. Choose dt to be greater than the shortest spike window length.')
    
    def V_reset(self):
        return self.V_rest + self.V_reset_intcp + self.V_reset_slope*(self.V_spike_begin-self.V_rest)
    def R_inv(self,V):
        return self.alpha + self.beta*(V-self.V_rest) + self.gamma*(V-self.V_rest)**2.0
        
    def syn_cur(self,S,W=None):
        '''calculate the synaptic current based on the connectivity matrix and the predicted spikes.
        input:
            S: tensor of shape [batch,n_neurons];
            W: tensor of shape [n_neurons,n_neurons]
        output:
            I_int: tenror of shape [batch,n_neurons]
        '''
        I_int = torch.zeros_like(S) # this should be modified based on given synapse model.
        return I_int
    
    def forward(self, I_ext):
        '''
        calculate the V and S from given external stimulus I_ext using the nonlinear version of the glif5 mdoel.
        Input: 
            I_ext: (list of tensors) external stimulus current. The shape of I_ext[t] is (batch,n_neurons) where t=0,1,...,T-1.
        
        Output: 
            S_pred: (torch tensor) prediction of binary spike train with shape (batch,T,n_neurons)
            V_pred: (torch tensor) prediction of voltages with shape (batch,T,n_neurons)
        '''
        assert type(I_ext) is list
        assert I_ext[0].shape[1] == self.n_neurons
        #I_ext : list of tensors with each tensor's shape=[batch,n_neurons], and len(I) = duration
        V_pred = [torch.ones_like(I_ext[0])*self.V_rest]
        S_pred = [torch.zeros_like(I_ext[0])]

        asc = torch.zeros((I_ext[0].shape[0], self.n_ascs, self.n_neurons)).to(self.device) # shape:[batch,2,n_neurons]
        theta_s = torch.zeros_like(I_ext[0])
        theta_v = torch.zeros_like(I_ext[0])
        
        #define a non-positive remaining_spike_window which increase to zero linearly in time but decrease by spike_len when a neuron spikes. 
        remaining_spike_window = torch.zeros_like(I_ext[0])
        #define a spike_portion_dt which denotes the length of time in each dt time bin where the neuron is in a spiek window. 
        spike_portion_of_dt = torch.zeros_like(I_ext[0])
        
        
        for cur in I_ext:
            #prepare the dynamic threshold based on the theta_s and theta_v from previous time bin:
            self.dynamic_V_threshold = self.V_threshold + theta_v + theta_s
            
            #remove from dt the spike windows from previously detected spikes.
            dt_minus_remaining_spike_window = (self.dt-remaining_spike_window).clamp(min=0.0) 
            
            # as we move into this new time bin, we substract the remaining_spike_window by dt. 
            #     If the neuron fires in this time bin, we will latter add spike_len to the remaining_spike_window.
            remaining_spike_window = (remaining_spike_window - self.dt).clamp(min=0.0)
            
            #calculate the prelinary voltage based on the asc and the I_int, and the dt_minus_remaining_spike_window.
            I_int = self.syn_cur(S_pred[-1])
            V_ = V_pred[-1] - dt_minus_remaining_spike_window*(V_pred[-1] - self.V_rest)*self.C_inv*self.R_inv(V_pred[-1]) + dt_minus_remaining_spike_window*(cur + I_int + asc.sum(1))*self.C_inv
            
            #====================================================================================
            #Given the preliminary voltage which may exceeds the V_threshold, we detect the spike and also adjust the actual time in the time bin that is out of any spike window:
            #           1. Set S=1 for neurons that begin to fire during this time bin, and record the V at the begining of he spike:
            S = torch.heaviside(V_- self.dynamic_V_threshold,values=torch.zeros_like(V_))
            self.V_spike_begin = self.dynamic_V_threshold
            
            #           2. Calculate the amount of time where this time bin overlapes with the begining of the spike window:
            spike_window_head = dt_minus_remaining_spike_window *  (1.0 - ( (self.dynamic_V_threshold-V_pred[-1]) / (torch.max(V_,self.dynamic_V_threshold) - V_pred[-1]) ))
            
            #           3. Reset the voltage for neurons that begins to fire during this time bin to V_threshold, and denote this voltage by V:
            V = (S[:,:]==0)*V_ + (S[:,:]==1)*(self.V_reset())
            
            #           4. Remove the spike_window_head from the dt_minus_remaining_spike_window and obtain the actual usable time during this time bin:
            spike_free_dt = dt_minus_remaining_spike_window - spike_window_head
            
            #           5. based on the usable time bin, calculate the asc, theta_v, and theta_s:
            asc = asc*(1 - spike_free_dt[:,None,:]*self.K_Ij)
            theta_v = theta_v + spike_free_dt*(-self.K_theta_v*theta_v + self.A_theta_v*(V-self.V_rest))
            theta_s = theta_s - spike_free_dt*theta_s*self.K_theta_s

            #           6. for neurons that begin to spike during this time bin, modify the asc using the A_Ij and R_Ij, modify the theta_s by self.A_theta_s:
            asc = asc + (S[:,None,:]==1)*(asc*self.R_Ij + self.A_Ij)
            theta_s = theta_s + (S[:,:]==1)*self.A_theta_s

            #           7. for neurons that begin to fire at this time bin, add spike_len to their remaining_spike_window, but also substract the spike_window_head.
            remaining_spike_window = remaining_spike_window + (S==1)*self.spike_len - spike_window_head
            #====================================================================================
            
            #save to list
            S_pred.append(S)            
            V_pred.append(V)
        return torch.stack(S_pred[1:], 1), torch.stack(V_pred[1:], 1)

def example():
    nlglif5_paras = {'V_rest': [-75.0,-70.0,-65.0],
                   'C_inv': [11.0,12.0,13.0],
                   'gamma': [0.003,0.002,0.003],
                   'beta': [-0.06,-0.05,-0.06],
                   'alpha': [3.8,3.8,3.8],
                   'V_rest': [-75.0,-70.0,-65.0],
                   'V_reset_slope': [1.0,0.9,0.9],
                   'V_reset_intcp': [-30.0,-20.0,-20.0],
                   'V_threshold': [-40.0,-40.0,-50.0],
                   'K_theta_s': [0.0,0.0,0.0],
                   'K_theta_v': [0.0,0.0,0.0],
                   'A_theta_v': [0.0,0.0,0.0],
                   'A_theta_s': [0.0,0.0,0.0],
                   'K_Ij': [[ 10., 300.],[10.,300.],[10.,300.]],
                   'A_Ij': [[-18.2971913 , 194.31189362],[-21.239182 , 292.11231362],[-15.9021913 , 90.32856483]],
                   'spike_len': [2.5,4.6,3.3]}
    I_ext = [torch.zeros(5,3).to('cuda') for k in range(int(2/0.002))] \
            + [300*torch.ones(5,3).to('cuda') for k in range(int(2/0.002))] \
            + [torch.zeros(5,3).to('cuda') for k in range(int(2/0.002))] 
    
    nlglif5 = NLGLIF5(nlglif5_paras,  # dictionary of parameters {'K_V': [K_V,K_V,....,K_V], 'R_V': ..., ...}
                 dt=0.002,  # this should be the length of a time bin in units of seconds.
                 device='cuda')
    
    S_pred,V_pred = nlglif5(I_ext)
    plt.figure()
    plt.plot(V_pred[0,:,0].to('cpu'),'r-',alpha=0.5)
    plt.plot(V_pred[0,:,1].to('cpu'),'g-',alpha=0.5)
    plt.plot(V_pred[0,:,2].to('cpu'),'b-',alpha=0.5)
    plt.show()
if __name__ == '__main__':
    example()
