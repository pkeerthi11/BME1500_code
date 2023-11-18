#Test 

# To run this script you need the following packages: mne, autoreject, nibabel,
# pyvistaqt, finnpy. All can be installed with pip. 

import os, mne
from preprocess_data import preprocess_data
from mne_source_reconstruction import mne_source_reconstruction
from calculate_connectivity import calculate_connectivity
from plot_connectivity import plot_connectivity
from morph_to_fsaverage import morph_to_fsaverage

# Subjects - Get bilateral, 130Hz
data_root = os.path.join('/home', 'bme1500trd', 'workspace', 'data', 'TRD', 'MEG')
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
image_root = os.path.join('raw', 'tsss')
dbs_off = ['01_tsss_1_GB_OFF1.fif',
           '01_tsss_1_OFF1.fif',
           '01_tsss_1_OFF1.fif',
           '01_tsss_1_DR_OFF1_L1.fif',
           '01_tsss_1_OFF1.fif',
           '01_tsss_1_HL_OFF1.fif',
           '01_tsss_1_ID_OFF1.fif',
           '01_tsss_1_BC_OFF1.fif',
           '01_tsss_1_OFF1.fif',
           '01_tsss_1_OFF1.fif',
           '01_tsss_DBSSMOFF1.fif',
           '01_tsss_DBSDBOFF1.fif']

# DBS-ON runs 
dbs_on = ['11_tsss_11_GB_1M2P_15_91_130_BILATERAL.fif',
          '11_tsss_11_1M2P_15_91_130_BILATERAL.fif',
          '11_tsss_11_1M2P_15_91_130_BILATERAL.fif',
          '11_tsss_11_DR_1M2P_15_91_130_BILATERAL.fif',
          '11_tsss_11_1M2P_15_90_130_BILATERAL.fif',
          '09_tsss_9_HL_2M3P_15_87_130_BILATERAL.fif',
          '09_tsss_9_ID_1M2P_15_91_130_BILATERAL.fif',
          '09_tsss_9_BC_1M2P_5M6P_15_90_130_BILATERAL.fif',
          '10_tsss_10_1M2P_15_91_130_BIL.fif',
          '05_tsss_5_1M2P_15_91_130_bilateral.fif',
          '09_tsss_DBSSM1M2P15087130BI.fif',
          '08_tsss_DBSDB_BILATERAL.fif']

# Set some initial run parameters 
subjects_dir = '/home/bme1500trd/freesurfer/subjects'
empty_room = '/home/bme1500trd/workspace/empty_room/selected/rec.fif'
n_jobs=-1
method = 'dSPM' # Source method
con_methods=['coh', 'pli', 'wpli2_debiased', 'ciplv'] # Connectivity metrics
sample_data_folder = mne.datasets.sample.data_path()
fsaverage_src = os.path.join(str(sample_data_folder), 'subjects', 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif')

# Preprocess empty room readings
(room_readings, _) = preprocess_data(empty_room, 120, 0, 'NA', 'NA', -1, False)

# Loop through subjects
for i in range(len(subjects)):
    
    # Get paths
    dbs_off_data = os.path.join(data_root,subjects[i],image_root,dbs_off[i])
    dbs_on_data = os.path.join(data_root,subjects[i],image_root,dbs_on[i]) 
    subject = subjects[i]
    
    ############### Process DBS-OFF
    
    # We first set hasCoreg to 'NA' for OFF data. We set path to it later for 
    # the ON data so that we don't do coregistration twice for same set of 
    # subjects
    hasCoreg = 'NA'
    
    # Preprocess OFF data
    (preprocessed_data, data_file_name) = preprocess_data(dbs_off_data, 120, 2, subjects_dir, subject, -1, True)
    
    # Run source reconstruction
    (stcs, stcs_psd, inverse_operator, intermediate_folder) = mne_source_reconstruction(preprocessed_data, room_readings, subjects_dir, 
                                                                                        subject, n_jobs, data_file_name,  method=method, 
                                                                                        hasT1=True, hasCoreg=hasCoreg)

    # Run this if you want to do connectivity analysis on fsaverage space 
    fsaverage_stcs = morph_to_fsaverage(stcs,fsaverage_src, True, str(subjects_dir), subject)
    (con_mat_theta, con_mat_alpha, con_mat_beta, con_mat_gamma, labels) = calculate_connectivity(preprocessed_data, fsaverage_stcs, fsaverage_src, subjects_dir, 'fsaverage', data_file_name, con_methods=['coh', 'pli', 'wpli2_debiased', 'ciplv'], n_jobs=-1)

    # Plot connectivity
    plot_connectivity(con_mat_theta, 'theta', labels, con_methods, subjects_dir, subject, data_file_name, save_fig=True)
    plot_connectivity(con_mat_alpha, 'alpha', labels, con_methods, subjects_dir, subject, data_file_name, save_fig=True)
    plot_connectivity(con_mat_beta, 'beta', labels, con_methods, subjects_dir, subject, data_file_name, save_fig=True)
    plot_connectivity(con_mat_gamma, 'gamma', labels, con_methods, subjects_dir, subject, data_file_name, save_fig=True)

    ################ Processs DBS-ON
    
    # Now set hasCoreg to where the Coreg folder would be. Would look better
    # in a loop but I am tired. 
    hasCoreg = intermediate_folder
    
    # Preprocess OFF data
    (preprocessed_data, data_file_name) = preprocess_data(dbs_on_data, 120, 2, subjects_dir, subject, -1, True)
    
    # Run source reconstruction
    (stcs, stcs_psd, inverse_operator, intermediate_folder) = mne_source_reconstruction(preprocessed_data, room_readings, subjects_dir, 
                                                                                        subject, n_jobs, data_file_name,  method=method, 
                                                                                        hasT1=True, hasCoreg=hasCoreg)

    # Run this if you want to do connectivity analysis on fsaverage space 
    fsaverage_stcs = morph_to_fsaverage(stcs,fsaverage_src, True, str(subjects_dir), subject)
    (con_mat_theta, con_mat_alpha, con_mat_beta, con_mat_gamma, labels) = calculate_connectivity(preprocessed_data, fsaverage_stcs, fsaverage_src, subjects_dir, 'fsaverage', data_file_name, con_methods=['coh', 'pli', 'wpli2_debiased', 'ciplv'], n_jobs=-1)

    # Plot connectivity
    plot_connectivity(con_mat_theta, 'theta', labels, con_methods, subjects_dir, subject, data_file_name, save_fig=True)
    plot_connectivity(con_mat_alpha, 'alpha', labels, con_methods, subjects_dir, subject, data_file_name, save_fig=True)
    plot_connectivity(con_mat_beta, 'beta', labels, con_methods, subjects_dir, subject, data_file_name, save_fig=True)
    plot_connectivity(con_mat_gamma, 'gamma', labels, con_methods, subjects_dir, subject, data_file_name, save_fig=True)    
