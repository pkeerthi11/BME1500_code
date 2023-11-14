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

# Path to the sample data folder provided by MNE
sample_data_folder = mne.datasets.sample.data_path()
data = (sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif')
room_noise = (sample_data_folder / 'MEG' / 'sample' / 'ernoise_raw.fif')

# Specify subject paths.
subjects_dir = (sample_data_folder / 'subjects')

# If this is set False, copy fsaverage. This is to test the no-T1 MRI case.
hasT1 = True 

# Get a copy of the sample data folder for this demo so we don't mess with MNE
# sample data. If the sample_copy folder already exists from an earlier run, 
# delete it first.
if os.path.exists(os.path.join(subjects_dir, 'sample_copy')):
    os.system('rm -r %s' % os.path.join(subjects_dir, 'sample_copy'))
# Copy fsaverage for noT1 case or copy sample subject for the standard case. 
if hasT1 == False:
    copy_fs_avg_anatomy(str(subjects_dir) + '/', 'sample_copy', 'sample_copy')
else:      
    sample_folder = os.path.join(subjects_dir, 'sample')
    sample_copy = os.path.join(subjects_dir, 'sample_copy')
    os.system('cp -r %s %s' % (sample_folder, sample_copy))

# Set subject name to sample_copy
subject = 'sample_copy'
    
# Number of cores to use. -1 uses all. 
n_jobs=4

######################### Run analysis functions ##############################

# Preprocess the experiment data and empty room measurements
preprocessed_data = preprocess_data(str(data), 250, 3, str(subjects_dir), subject, -1, True)
room_readings = preprocess_data(str(data), 250, 0, 'NA', 'NA', -1, False)

# Run source reconstruction
(stcs, stcs_psd, inverse_operator) = mne_source_reconstruction(preprocessed_data, room_readings, str(subjects_dir), subject, n_jobs, hasT1=hasT1)

# Connectivity
(con_mat_theta, con_mat_alpha, con_mat_beta, con_mat_gamma, labels) = calculate_connectivity(preprocessed_data, stcs, str(subjects_dir), subject, inverse_operator, con_methods=['coh', 'pli', 'wpli2_debiased', 'ciplv'], n_jobs=n_jobs)

# Plot connectivity
con_methods=['coh', 'pli', 'wpli2_debiased', 'ciplv']
plot_connectivity(con_mat_theta, labels, con_methods, str(subjects_dir), subject, save_fig=True)
plot_connectivity(con_mat_alpha, labels, con_methods, str(subjects_dir), subject, save_fig=True)
plot_connectivity(con_mat_beta, labels, con_methods, str(subjects_dir), subject, save_fig=True)
plot_connectivity(con_mat_gamma, labels, con_methods, str(subjects_dir), subject, save_fig=True)

# Morph to fsaverage space 
fsaverage_stcs = morph_to_fsaverage(stcs, hasT1, str(subjects_dir), subject)
fsaverage_stcs_psd = morph_to_fsaverage(stcs_psd, hasT1, str(subjects_dir), subject)

# Run label averaging on fsaverage space. We use the aparc label
(label_epochs, times, label_epochs_psd, frequencies) = average_sources_in_label(str(subjects_dir), 'fsaverage', fsaverage_stcs, fsaverage_stcs_psd)
