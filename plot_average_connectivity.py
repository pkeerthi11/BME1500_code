import os, mne
import numpy as np
from plot_connectivity import plot_connectivity

# Plot average connectivity 
results_folder_on = '/Users/ozenctaskin/Desktop/results/dbs-on'
if not os.path.exists(results_folder_on):
    os.system('mkdir %s' % results_folder_on)

results_folder_off = '/Users/ozenctaskin/Desktop/results/dbs-off'
if not os.path.exists(results_folder_off):
    os.system('mkdir %s' % results_folder_off)

subjects_dir = '/Users/ozenctaskin/Desktop/subjects/'
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

dbs_off_theta_list = []
dbs_off_alpha_list = []
dbs_off_beta_list = []
dbs_off_gamma_list = []

dbs_on_theta_list = []
dbs_on_alpha_list = []
dbs_on_beta_list = []
dbs_on_gamma_list =[]

for i in range(len(subjects)):
    
    dbs_off_theta_list.append(np.load(os.path.join(subjects_dir, subjects[i], dbs_off[i], 'connectivity_results','theta_connectivity.npy'), allow_pickle=True).item())
    dbs_off_alpha_list.append(np.load(os.path.join(subjects_dir, subjects[i], dbs_off[i], 'connectivity_results','alpha_connectivity.npy'), allow_pickle=True).item())
    dbs_off_beta_list.append(np.load(os.path.join(subjects_dir, subjects[i],  dbs_off[i], 'connectivity_results','beta_connectivity.npy'), allow_pickle=True).item())
    dbs_off_gamma_list.append(np.load(os.path.join(subjects_dir, subjects[i], dbs_off[i], 'connectivity_results','gamma_connectivity.npy'), allow_pickle=True).item())
    
    dbs_on_theta_list.append(np.load(os.path.join(subjects_dir, subjects[i],  dbs_on[i], 'connectivity_results','theta_connectivity.npy'), allow_pickle=True).item())
    dbs_on_alpha_list.append(np.load(os.path.join(subjects_dir, subjects[i],  dbs_on[i], 'connectivity_results','alpha_connectivity.npy'), allow_pickle=True).item())
    dbs_on_beta_list.append(np.load(os.path.join(subjects_dir, subjects[i],  dbs_on[i], 'connectivity_results','beta_connectivity.npy'), allow_pickle=True).item())
    dbs_on_gamma_list.append(np.load(os.path.join(subjects_dir, subjects[i], dbs_on[i], 'connectivity_results','gamma_connectivity.npy'), allow_pickle=True).item())

dbs_list = [dbs_off_theta_list, dbs_off_alpha_list, dbs_off_beta_list, dbs_off_gamma_list,
            dbs_on_theta_list, dbs_on_alpha_list, dbs_on_beta_list, dbs_on_gamma_list]

for group in dbs_list:
    coh = np.zeros([68,68])
    pli = np.zeros([68,68])
    wpli2_debiased = np.zeros([68,68])
    ciplv = np.zeros([68,68])
    group_length = len(group)
    for run in group:
        for model in run.keys():
            if model == 'coh':
                coh = coh + run[model]
            if model == 'pli':
                pli = pli + run[model]
            if model == 'wpli2_debiased':
                wpli2_debiased = wpli2_debiased + run[model]
            if model == 'ciplv':
                ciplv = ciplv + run[model]
      
    coh = coh / group_length
    pli = pli / group_length
    wpli2_debiased = wpli2_debiased / group_length
    ciplv = ciplv / group_length
    
        
    group.append({'coh_avg': coh, 'pli_avg': pli, 'wpli2_debiased_avg': wpli2_debiased, 'ciplv_avg': ciplv})

labels = mne.read_labels_from_annot('fsaverage', parc='aparc',
                                    subjects_dir='/Applications/freesurfer/7.4.1/subjects/')
labels = labels[:-1]
con_methods=['coh_avg', 'pli_avg', 'wpli2_debiased_avg', 'ciplv_avg']
plot_connectivity(dbs_list[0][12], 'theta', labels,  con_methods, subjects_dir, 'fsaverage', results_folder_off, save_fig=True)
plot_connectivity(dbs_list[1][12], 'alpha', labels,  con_methods, subjects_dir, 'fsaverage', results_folder_off, save_fig=True)
plot_connectivity(dbs_list[2][12], 'beta', labels,  con_methods, subjects_dir, 'fsaverage', results_folder_off, save_fig=True)
plot_connectivity(dbs_list[3][12], 'gamma', labels,  con_methods, subjects_dir, 'fsaverage', results_folder_off, save_fig=True)
plot_connectivity(dbs_list[4][12], 'theta', labels,  con_methods, subjects_dir, 'fsaverage', results_folder_on, save_fig=True)
plot_connectivity(dbs_list[5][12], 'alpha', labels,  con_methods, subjects_dir, 'fsaverage', results_folder_on, save_fig=True)
plot_connectivity(dbs_list[6][12], 'beta', labels,  con_methods, subjects_dir, 'fsaverage', results_folder_on, save_fig=True)
plot_connectivity(dbs_list[7][12], 'gamma', labels,  con_methods, subjects_dir, 'fsaverage', results_folder_on, save_fig=True)


    