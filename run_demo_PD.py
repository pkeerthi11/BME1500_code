#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:31:02 2024

@author: prerana
"""

# To run this script you need the following packages: mne, autoreject, nibabel,
# pyvistaqt, finnpy. All can be installed with pip. 

import os, mne
from preprocess_data import preprocess_data
from mne_source_reconstruction import mne_source_reconstruction
from average_sources_in_label import average_sources_in_label
from finnpy.source_reconstruction.mri_anatomy import copy_fs_avg_anatomy
from calculate_connectivity import calculate_connectivity
from plot_connectivity import plot_connectivity
from morph_to_fsaverage import morph_to_fsaverage

######################## Set paths and variables ##############################

# Specify subject paths.
subjects_dir = '/storage/prerana/subjects/pd'
subject = 'al0001a'
data = os.path.join(subjects_dir, subject, '01_tsss_DBS_GA_OFF1', '01_tsss_DBS_GA_OFF1.fif')
room_noise = os.path.join('/mnt','sensor_covariance_file.fif')


# If this is set False, copy fsaverage. This is to test the no-T1 MRI case.
hasT1 = True 

# Set this to the sourceRecIntermediateFiles saved in the freesurfer subject
# folder if you already have the coregistration performed.
hasCoreg = 'NA'

# Copy fsaverage for noT1 case or copy sample subject for the standard case. 
if hasT1 == False:
    copy_fs_avg_anatomy(str(subjects_dir) + '/', subject)


    
# Number of cores to use. -1 uses all. 
n_jobs=4

# The method to use for source reconstruction and connectivity 
method = 'dSPM'
con_methods=['coh', 'pli', 'wpli2_debiased', 'ciplv']
######################### Run analysis functions ##############################

# Preprocess the experiment data and empty room measurements
(preprocessed_data, data_file_name) = preprocess_data(str(data), 120, 0, str(subjects_dir), subject, -1, True)
(room_readings, _) = preprocess_data(str(room_noise), 120, 0, 'NA', 'NA', -1, False) #Place outside 

# Run source reconstruction
(stcs, stcs_psd, inverse_operator, intermediate_folder) = mne_source_reconstruction(preprocessed_data, room_readings, str(subjects_dir), subject, n_jobs, data_file_name,  method=method, hasT1=hasT1, hasCoreg=hasCoreg)

# Connectivity
#(con_mat_theta, con_mat_alpha, con_mat_beta, con_mat_gamma, labels) = calculate_connectivity(preprocessed_data, stcs, inverse_operator['src'], str(subjects_dir), subject, data_file_name, con_methods=con_methods, n_jobs=n_jobs)

# # Run this if you want to do connectivity analysis on fsaverage space 
# fsaverage_src = os.path.join(sample_data_folder, 'subjects', 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif')
# fsaverage_stcs = morph_to_fsaverage(stcs,fsaverage_src, hasT1, str(subjects_dir), subject)
# (con_mat_theta, con_mat_alpha, con_mat_beta, con_mat_gamma, labels) = calculate_connectivity(preprocessed_data, fsaverage_stcs, fsaverage_src, subjects_dir, 'fsaverage', data_file_name, con_methods=['coh', 'pli', 'wpli2_debiased', 'ciplv'], n_jobs=-1)

# Plot connectivity
#plot_connectivity(con_mat_theta, 'theta', labels, con_methods, str(subjects_dir), subject, data_file_name, save_fig=True)
#plot_connectivity(con_mat_alpha, 'alpha', labels, con_methods, str(subjects_dir), subject, data_file_name, save_fig=True)
#plot_connectivity(con_mat_beta, 'beta', labels, con_methods, str(subjects_dir), subject, data_file_name, save_fig=True)
#plot_connectivity(con_mat_gamma, 'gamma', labels, con_methods, str(subjects_dir), subject, data_file_name, save_fig=True)
