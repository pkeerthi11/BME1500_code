import os, mne, scipy
import numpy as np
from plot_connectivity import plot_connectivity
import mne_connectivity
from copy import deepcopy
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

# Set the output plots folder
results_folder = '/home/bme1500trd/Desktop/connectivity_results'
if not os.path.exists(results_folder):
    os.system('mkdir %s' % results_folder)

# Create a subfolder for dbs-on data
results_folder_on = '/home/bme1500trd/Desktop/connectivity_results/dbs-on'
if not os.path.exists(results_folder_on):
    os.system('mkdir %s' % results_folder_on)

# Create a subfolder for dbs-off data
results_folder_off = '/home/bme1500trd/Desktop/connectivity_results/dbs-off'
if not os.path.exists(results_folder_off):
    os.system('mkdir %s' % results_folder_off)

# Subject names and paths
subjects_dir = '/home/bme1500trd/freesurfer/subjects'
subjects = ['al0067a',
            'al0064a',
            'al0053a',
            'al0051a',
            'al0049a',
            'al0047a',
            'al0043a',
            'al0042a',
            'al0040a',
            'al0039a',
            'al0029a',
            'al0011a']

# DBS-OFF runs 
dbs_off = ['01_tsss_1_GB_OFF1',
           '01_tsss_1_OFF1',
           '01_tsss_1_OFF1',
           '01_tsss_1_DR_OFF1_L1',
           '01_tsss_1_OFF1',
           '01_tsss_1_HL_OFF1',
           '01_tsss_1_ID_OFF1',
           '01_tsss_1_BC_OFF1',
           '01_tsss_1_OFF1',
           '01_tsss_1_OFF1',
           '01_tsss_DBSSMOFF1',
           '01_tsss_DBSDBOFF1']

# DBS-ON runs 
dbs_on = ['11_tsss_11_GB_1M2P_15_91_130_BILATERAL',
          '11_tsss_11_1M2P_15_91_130_BILATERAL',
          '11_tsss_11_1M2P_15_91_130_BILATERAL',
          '11_tsss_11_DR_1M2P_15_91_130_BILATERAL',
          '11_tsss_11_1M2P_15_90_130_BILATERAL',
          '09_tsss_9_HL_2M3P_15_87_130_BILATERAL',
          '09_tsss_9_ID_1M2P_15_91_130_BILATERAL',
          '09_tsss_9_BC_1M2P_5M6P_15_90_130_BILATERAL',
          '10_tsss_10_1M2P_15_91_130_BIL',
          '05_tsss_5_1M2P_15_91_130_bilateral',
          '09_tsss_DBSSM1M2P15087130BI',
          '08_tsss_DBSDB_BILATERAL']

# Set some empty lists
dbs_off_theta_list = list()
dbs_off_alpha_list = list()
dbs_off_beta_list = list()
dbs_off_gamma_list = list()

dbs_on_theta_list = list()
dbs_on_alpha_list = list()
dbs_on_beta_list = list()
dbs_on_gamma_list = list()

# Run through subjects and load connectivity results for all frequency bands
for i in range(len(subjects)):
    
    dbs_off_theta_list.append(np.load(os.path.join(subjects_dir, subjects[i], dbs_off[i], 'connectivity_results','theta_connectivity.npy'), allow_pickle=True).item())
    dbs_off_alpha_list.append(np.load(os.path.join(subjects_dir, subjects[i], dbs_off[i], 'connectivity_results','alpha_connectivity.npy'), allow_pickle=True).item())
    dbs_off_beta_list.append(np.load(os.path.join(subjects_dir, subjects[i],  dbs_off[i], 'connectivity_results','beta_connectivity.npy'), allow_pickle=True).item())
    dbs_off_gamma_list.append(np.load(os.path.join(subjects_dir, subjects[i], dbs_off[i], 'connectivity_results','gamma_connectivity.npy'), allow_pickle=True).item())
    
    dbs_on_theta_list.append(np.load(os.path.join(subjects_dir, subjects[i],  dbs_on[i], 'connectivity_results','theta_connectivity.npy'), allow_pickle=True).item())
    dbs_on_alpha_list.append(np.load(os.path.join(subjects_dir, subjects[i],  dbs_on[i], 'connectivity_results','alpha_connectivity.npy'), allow_pickle=True).item())
    dbs_on_beta_list.append(np.load(os.path.join(subjects_dir, subjects[i],  dbs_on[i], 'connectivity_results','beta_connectivity.npy'), allow_pickle=True).item())
    dbs_on_gamma_list.append(np.load(os.path.join(subjects_dir, subjects[i], dbs_on[i], 'connectivity_results','gamma_connectivity.npy'), allow_pickle=True).item())

# Combine frequency lists in a parent list
dbs_list = [dbs_off_theta_list, dbs_off_alpha_list, dbs_off_beta_list, dbs_off_gamma_list,
            dbs_on_theta_list, dbs_on_alpha_list, dbs_on_beta_list, dbs_on_gamma_list]

# Loop through the parent list and average coherence values across subjects
for group in dbs_list:
    coh = np.zeros([68,68])
    group_length = len(group)
    for run in group:
        for model in run.keys():
            if model == 'coh':
                coh = coh + run[model]
                
    coh = coh / group_length   
    group.append({'coh_avg': coh})
                
# Load labels for one of the subjects. We just need the names for plotting,
# so no need to load subjectspecific labels
labels = mne.read_labels_from_annot('al0067a', parc='aparc')
labels_pseudo = labels[::2]

# Plot average connectivity
con_methods=['coh_avg']
vmin = [0.5]
vmax = [1]
plot_connectivity(dbs_list[0][12], 'theta', labels,  con_methods, vmin, vmax, results_folder_off, save_fig=True, both_hemi=True)
plot_connectivity(dbs_list[1][12], 'alpha', labels,  con_methods, vmin, vmax, results_folder_off, save_fig=True, both_hemi=True)
plot_connectivity(dbs_list[2][12], 'beta', labels,  con_methods, vmin, vmax, results_folder_off, save_fig=True, both_hemi=True)
plot_connectivity(dbs_list[3][12], 'gamma', labels,  con_methods, vmin, vmax, results_folder_off, save_fig=True, both_hemi=True)
plot_connectivity(dbs_list[4][12], 'theta', labels,  con_methods, vmin, vmax, results_folder_on, save_fig=True, both_hemi=True)
plot_connectivity(dbs_list[5][12], 'alpha', labels,  con_methods, vmin, vmax, results_folder_on, save_fig=True, both_hemi=True)
plot_connectivity(dbs_list[6][12], 'beta', labels,  con_methods, vmin, vmax, results_folder_on, save_fig=True, both_hemi=True)
plot_connectivity(dbs_list[7][12], 'gamma', labels,  con_methods, vmin, vmax, results_folder_on, save_fig=True, both_hemi=True)


# # Indices of the labels we are interested in. Labels are isthmus cingulate (ICC) 20-21,
# # parahippocampal gyrus (paraH) 34-35, posterior cingulate gyrus  (PCC) 46-47, the precuneus cortex (PCUN) 50-51, 
# # rosttral anterior cingulate (rACC) 52-53
indices = [20,21,34,35,46,47,50,51,52,53] # This is for both hemispheres
labels_indexed = [labels[i] for i in indices]

# Plot connectivity averages
beta_dbs_off_dmn_avg = {}
beta_dbs_on_dmn_avg = {}
beta_dbs_off_dmn_avg['coh_avg'] = dbs_list[2][12]['coh_avg'][np.ix_(indices,indices)]
beta_dbs_on_dmn_avg['coh_avg'] = dbs_list[6][12]['coh_avg'][np.ix_(indices,indices)]
plot_connectivity(beta_dbs_off_dmn_avg, 'beta_off', labels_indexed,  con_methods, [0.3], [0.7], results_folder_off, save_fig=False)
plot_connectivity(beta_dbs_on_dmn_avg, 'beta_on', labels_indexed,  con_methods, [0.3],[0.7], results_folder_off, save_fig=False)

# Stack coherence measures in a 3D matrix
beta_off_dmn = np.stack([dbs_list[2][0]['coh'][np.ix_(indices,indices)],dbs_list[2][1]['coh'][np.ix_(indices,indices)],dbs_list[2][2]['coh'][np.ix_(indices,indices)],dbs_list[2][3]['coh'][np.ix_(indices,indices)],dbs_list[2][4]['coh'][np.ix_(indices,indices)],dbs_list[2][5]['coh'][np.ix_(indices,indices)],
                      dbs_list[2][6]['coh'][np.ix_(indices,indices)],dbs_list[2][7]['coh'][np.ix_(indices,indices)],dbs_list[2][8]['coh'][np.ix_(indices,indices)],dbs_list[2][9]['coh'][np.ix_(indices,indices)],dbs_list[2][10]['coh'][np.ix_(indices,indices)],dbs_list[2][11]['coh'][np.ix_(indices,indices)]], axis=2)
    
beta_on_dmn = np.stack([dbs_list[6][0]['coh'][np.ix_(indices,indices)],dbs_list[6][1]['coh'][np.ix_(indices,indices)],dbs_list[6][2]['coh'][np.ix_(indices,indices)],dbs_list[6][3]['coh'][np.ix_(indices,indices)],dbs_list[6][4]['coh'][np.ix_(indices,indices)],dbs_list[6][5]['coh'][np.ix_(indices,indices)],
                      dbs_list[6][6]['coh'][np.ix_(indices,indices)],dbs_list[6][7]['coh'][np.ix_(indices,indices)],dbs_list[6][8]['coh'][np.ix_(indices,indices)],dbs_list[6][9]['coh'][np.ix_(indices,indices)],dbs_list[6][10]['coh'][np.ix_(indices,indices)],dbs_list[6][11]['coh'][np.ix_(indices,indices)]], axis=2)


# Do a t-test with off and on data. Plot matrix
t,p = ttest_rel(beta_off_dmn, beta_on_dmn, axis=2)
plt.imshow(t, cmap='coolwarm')
plt.colorbar()

# Do a multiple comparisons correction
pp = np.ndarray.flatten(p)
pp = pp[~np.isnan(pp)]
reject, pp, _, _ = multipletests(pp, method='fdr_bh', alpha=0.05)

# Hard code HAM-D scores 
scores = np.array([23,22,22,26,20,26,22,24,23,30,31,26])

# Do a correlation between HAM-D scores and dbs-off connectivity
res = np.apply_along_axis(lambda x: scipy.stats.pearsonr(x, scores.flatten()), axis=2, arr=beta_off_dmn)
plt.imshow(res[:,:,0], cmap='coolwarm')
plt.colorbar()