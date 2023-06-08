import numpy as np
import json
import requests
import pickle
import os, random
import pandas as pd
import matplotlib.pyplot as plt
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.api.queries.glif_api import GlifApi
ctc = CellTypesCache(manifest_file='../cell_types/manifest.json')
import subprocess

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

runcmd('echo "Hello, World!"', verbose = True)

def get_glif3_para(neuron_id):
    '''
    This function extracts the glif3 parameters from the cell-type database.
    
    Input: specimen_id (cell_id)
    
    Output: None if glif3 model is not available
            or
            {'V_rest':V_rest, 'V_reset':V_reset, 'V_threshold':V_threshold, 'K_V':K_V, 'R_V':R_V, 'K_Ij':K_Ij, 'A_Ij':A_Ij, 'spike_len':spike_len}
            where 
            V_rest: resting voltage (mV)
            V_reset: should be the same as the resting voltage (mV)
            V_threshold: the spiking voltage threshold (mV)
            spike_len: the time length of the spike cut (ms)
            
            K_V: exp decay rate of voltage (mV/s)
            R_V: neuron's resistance (10^9 Ohm)
            K_Ij: exp decay rate of after spike currents, 2D vector (pA/s)
            A_Ij: increase of after spike current at the end of a spike, 2D vector (pA)
            
            The dynamical equation of the glif3 between spike cuts is:
                dV/dt = -K_V*(V-V_rest) + K_V*R_V*(I_ext + I_j)
                dI_j/dt = -K_Ij*I_j
            The model ignore the dynamics during the spike cut windows.
            At the end of each spike cut window, 
                V(t_spike_end) = V_reset
                I_j(t_spike_end) = 1.0*I_j(t_spike_begin) + A_Ij
            For the above equations, K_Ij, A_Ij and I_j are 2-d vectors, representing two independent ion channels.            
    '''
    try:
        glif_api = GlifApi()
        #use api to obtain glif3 neuronal_model_id from neuron_id
        api_url = "http://api.brain-map.org/api/v2/data/query.xml?criteria=model::NeuronalModel,rma::critera,[specimen_id$eq" + str(neuron_id) + "],neuronal_model_template[name$il'*3 LIF*']"
        glif3_metadata_by_specimen_id = requests.get(api_url).text
        sub1 = "<id>"
        sub2 = "</id>"
        # getting index of substrings
        idx1 = glif3_metadata_by_specimen_id.index(sub1)
        idx2 = glif3_metadata_by_specimen_id.index(sub2)
        model_id_string = ''
        # getting elements in between
        for idx in range(idx1 + len(sub1), idx2):
            model_id_string = model_id_string + glif3_metadata_by_specimen_id[idx]
        neuronal_model_id = int(model_id_string)

        neuron_config = glif_api.get_neuron_configs([neuronal_model_id])

        El_reference = neuron_config[neuronal_model_id]['El_reference']
        El = neuron_config[neuronal_model_id]['El']
        V_rest = 1e3 * (El_reference + El)
        V_reset = V_rest

        C_val = neuron_config[neuronal_model_id]['C']
        coeff_C = neuron_config[neuronal_model_id]['coeffs']['C']
        C = coeff_C * C_val
        R_V_K_V = 1e-9*(1.0 / C)

        th_inf = neuron_config[neuronal_model_id]['th_inf']
        coeff_th_inf = neuron_config[neuronal_model_id]['coeffs']['th_inf']
        V_threshold = 1e3 * th_inf * np.array(coeff_th_inf) + V_rest

        R_input = neuron_config[neuronal_model_id]['R_input']
        coeff_G = neuron_config[neuronal_model_id]['coeffs']['G']
        G = coeff_G * 1.0 / R_input
        K_V= G / C

        R_V = R_V_K_V / K_V

        asc_tau_array = neuron_config[neuronal_model_id]['asc_tau_array']
        K_Ij = 1.0 / np.array(asc_tau_array)

        asc_amp_array = neuron_config[neuronal_model_id]['asc_amp_array']
        coeff_asc_amp_array = neuron_config[neuronal_model_id]['coeffs']['asc_amp_array']
        A_Ij = 1e12 * np.array(coeff_asc_amp_array) * np.array(asc_amp_array)

        spike_len = neuron_config[neuronal_model_id]['dt'] * neuron_config[neuronal_model_id]['spike_cut_length']*1000


        # Downloading the model metadata JSON file
        !mkdir temp
        command_str = 'wget http://api.brain-map.org/neuronal_model/download/' + str(neuronal_model_id) + ' -O temp/temp.zip'
        runcmd(command_str)
        !unzip temp/temp -d temp/
        
        # Opening and reading the JSON file
        with open('temp/model_metadata.json') as json_file:
            model_metadata = json.load(json_file)
        explained_var_ratio = model_metadata['neuronal_model_runs'][0]['explained_variance_ratio']
        # Cleaning Up the downloaded file
        !rm -r temp
        
        glif3_para = {'V_rest':V_rest, 'V_reset':V_reset, 'V_threshold':V_threshold, 'K_V':K_V, 'R_V':R_V, 'K_Ij':K_Ij, 'A_Ij':A_Ij, 'spike_len':spike_len}
    except:
        glif3_para = None
        explained_var_ratio = None
    return glif3_para, explained_var_ratio


# Download the two metadata files:
file_for_cre_line = 'cell_types_specimen_details.csv'#https://celltypes.brain-map.org/cell_types_specimen_details.csv
if not os.path.isfile(file_for_cre_line):
    !wget 'https://celltypes.brain-map.org/cell_types_specimen_details.csv' --no-check-certificate
file_for_m_type = '41593_2019_417_MOESM5_ESM.xlsx'#https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-019-0417-0/MediaObjects/41593_2019_417_MOESM5_ESM.xlsx
if not os.path.isfile(file_for_m_type):
    !wget 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-019-0417-0/MediaObjects/41593_2019_417_MOESM5_ESM.xlsx' --no-check-certificate

cell_ids = [x['id'] for x in ctc.get_cells(species=[CellTypesApi.MOUSE])]
creline_type_data = pd.read_csv(file_for_cre_line)
m_type_data = pd.read_excel(file_for_m_type)


MetaData_w_Glif3Paras = {}
for cell_id in cell_ids:
    glif3paras,explained_var_ratio = get_glif3_para(cell_id)
    try:
        creline_type = str(creline_type_data.loc[creline_type_data['specimen__id'] == cell_id]['line_name'].values[0])
    except:
        creline_type = ''
    try:
        area = str(creline_type_data.loc[creline_type_data['specimen__id'] == cell_id]['structure_parent__acronym'].values[0])
    except:
        area = ''
    try:
        layer = str(creline_type_data.loc[creline_type_data['specimen__id'] == cell_id]['structure__layer'].values[0])
    except:
        layer = ''
    try:
        m_type = str(m_type_data.loc[m_type_data['specimen_id'] == cell_id]['m-type'].values[0])
        if m_type == 'nan':
            m_type = ''
    except:
        m_type = ''
    MetaData_w_Glif3Paras[cell_id] = {'m_type': m_type, 'creline_type':creline_type, 'area':area, 'layer':layer, 'glif3paras':glif3paras, 'explained_var_ratio':explained_var_ratio} 

with open('glif3_para_with_meta_data.pickle', 'wb') as handle:
    pickle.dump(MetaData_w_Glif3Paras, handle, protocol=pickle.HIGHEST_PROTOCOL)
