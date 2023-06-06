import torch
from torch import nn
import matplotlib.pyplot as plt

class GLIF3(nn.Module):
    def __init__(self,
                 neuronal_model_paras,  # dictionary of parameters {'K_V': [K_V,K_V,....,K_V], 'R_V': ..., ...}
                 dt=0.002,  # length of a time bin (seconds).
                 device='cuda'):

        super().__init__()
        self.n_neurons = len(neuronal_model_paras['K_V'])
        self.n_ascs = 2 # two after spike current channels
        assert self.n_ascs == len(neuronal_model_paras['K_Ij'][0])
        assert self.n_neurons == len(neuronal_model_paras['K_Ij'])
        self.dt = dt # (seconds)
        self.device = device

        self.K_V = nn.Parameter(torch.tensor(neuronal_model_paras['K_V']), requires_grad=False)
        self.R_V_K_V = nn.Parameter(torch.tensor(neuronal_model_paras['K_V'])*torch.tensor(neuronal_model_paras['R_V']), requires_grad=False)
        self.V_rest = nn.Parameter(torch.tensor(neuronal_model_paras['V_rest']), requires_grad=False)
        self.V_reset = nn.Parameter(torch.tensor(neuronal_model_paras['V_reset']), requires_grad=False)
        self.V_threshold = nn.Parameter(torch.tensor(neuronal_model_paras['V_threshold']), requires_grad=False)
        self.spike_len = nn.Parameter(0.001*torch.tensor(neuronal_model_paras['spike_len']), requires_grad=False)
        self.K_Ij = nn.Parameter(torch.tensor(neuronal_model_paras['K_Ij']).T, requires_grad=False)
        self.R_Ij = nn.Parameter(torch.ones_like(torch.tensor(neuronal_model_paras['K_Ij'])).T, requires_grad=False)
        self.A_Ij = nn.Parameter(torch.tensor(neuronal_model_paras['A_Ij']).T, requires_grad=False)
        self.to(device)
        
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
        calculate the V and S from given external stimulus I_ext using the glif3 mdoel.
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
        
        #define a non-positive frozen_dt which increase to zero linearly in time but decrease by spike_length when a neuron spikes. 
        frozen_dt = torch.zeros_like(I_ext[0])
        
        for cur in I_ext:
            #calculate the actual usable dt in each time bin for each neuron by excluding the spike_window_len.
            spike_free_dt = nn.functional.relu(self.dt+frozen_dt) # frozen_dt non-possitive. 
            #both the asc and the I_int at the present time bin is calculated based on the spike status of the previous time bin.
            asc = asc*(1 - spike_free_dt[:,None,:]*self.K_Ij)
            I_int = self.syn_cur(S_pred[-1],W=None)
            
            #calculate the prelinary voltage based on the asc and the I_int
            V_ = V_pred[-1] - spike_free_dt*self.K_V*(V_pred[-1] - self.V_rest) + spike_free_dt*(cur + I_int + asc.sum(1))*self.R_V_K_V
            # calculate the S based on the preliminary voltage
            S = torch.heaviside(V_- self.V_threshold,values=torch.zeros_like(V_))*(frozen_dt==0.0)
            
            # increase asc by A_Ij for firing neurons:
            asc = asc + (S[:,None,:]==1)*(asc*self.R_Ij + self.A_Ij)
            
            # decrease the frozen_dt by -1.0*spike_len for neurons that fire in this time bin, and increase frozen_dt by self.dt for all neurons, then clamp frozen_dt to 0.
            frozen_dt = (frozen_dt - (S[:,:]==1)*self.spike_len + self.dt).clamp(max=0.0)

            S_pred.append(S)
            #reset the preliminary voltage V_ to V_reset for firing neurons
            V = (S[:,:]==0)*V_ + (S[:,:]==1)*(self.V_reset)
            
            V_pred.append(V)
        return torch.stack(S_pred[1:], 1), torch.stack(V_pred[1:], 1)

def example():
    glif3_paras = {'V_rest': [-75.0,-70.0,-65.0],
            'V_reset': [-75.0,-70.0,-65.0],
            'V_threshold': [-30.0,-40.0,-50.0],
            'K_V': [110.0,100.0,115.0],
            'R_V': [0.104,0.106,0.096],
            'K_Ij': [[ 10., 300.],[10.,300.],[10.,300.]],
            'A_Ij': [[-18.2971913 , 194.31189362],[-21.239182 , 292.11231362],[-15.9021913 , 90.32856483]],
            'spike_len': [2.5,4.6,3.3]}
    I_ext = [torch.zeros(5,3).to('cuda') for k in range(int(2/0.002))] \
            + [300*torch.ones(5,3).to('cuda') for k in range(int(2/0.002))] \
            + [torch.zeros(5,3).to('cuda') for k in range(int(2/0.002))] 
    
    S_pred,V_pred = glif3(I_ext)
    plt.figure()
    plt.plot(V_pred[0,:,0].to('cpu'),'r-',alpha=0.5)
    plt.plot(V_pred[0,:,1].to('cpu'),'g-',alpha=0.5)
    plt.plot(V_pred[0,:,2].to('cpu'),'b-',alpha=0.5)
    plt.show()
if __name__ == '__main__':
    example()
