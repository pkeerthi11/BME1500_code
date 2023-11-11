# To run this script you need the following packages: mne, autoreject, nibabel,
# pyvistaqt, finnpy. All can be installed with pip. 

import os, mne
from preprocessData import preprocessData
from mne_sourceReconstruction import mne_sourceReconstruction
from averageSourcesInLabel import averageSourcesInLabel
from finnpy.source_reconstruction.mri_anatomy import copy_fs_avg_anatomy

# path to the data
sample_data_folder = mne.datasets.sample.data_path()
data = (sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif')
room_noise = (sample_data_folder / 'MEG' / 'sample' / 'ernoise_raw.fif')

# Subject paths
subjects_dir = (sample_data_folder / 'subjects')

# If this is set, copy fsaverage. This is to test the no-T1 MRI case.
hasT1 = True 

# Get a copy of the sample data folder for this demo so we don't mess with MNE
# sample data. If the copy folder exists, delete it first
if os.path.exists(os.path.join(subjects_dir, 'sample_copy')):
    os.system('rm -r %s' % os.path.join(subjects_dir, 'sample_copy'))
# Copy fsaverage for noT1 case or copy sample subject for the standard case. 
if hasT1 == False:
    copy_fs_avg_anatomy(str(subjects_dir) + '/', 'sample_copy', 'sample_copy')
else:      
    sample_folder = os.path.join(subjects_dir, 'sample')
    sample_copy = os.path.join(subjects_dir, 'sample_copy')
    os.system('cp -r %s %s' % (sample_folder, sample_copy))

# Subject name
subject = 'sample_copy'

# Our functions do not overwrite anything. So we clear some of the folders and 
# files this script creates so that it can be run again without manual cleaning
if os.path.exists(os.path.join(subjects_dir, subject, 'plots')):
    os.system('rm -r %s' % os.path.join(subjects_dir, subject, 'plots'))

if os.path.exists(os.path.join(subjects_dir, subject, 'sourceRecIntermediateFiles')):
    os.system('rm -r %s' % os.path.join(subjects_dir, subject, 'sourceRecIntermediateFiles'))    

for i in os.listdir(os.path.join(subjects_dir, subject, 'bem')):
    if i != 'sample-fiducials.fif':
        os.system('rm %s' % os.path.join(subjects_dir, subject, 'bem', i))
    
# Core values. -1 uses all. 
n_jobs=4

# Preprocess the experiment data and empty room measurements
preprocessed_data = preprocessData(data, 250, 3, subjects_dir, subject, -1, True)
room_readings = preprocessData(data, 250, 0, 'NA', 'NA', -1, False)

# Run source reconstruction
(stcs, stcs_psd, fsaverage_stcs, fsaverage_stcs_psd) = mne_sourceReconstruction(preprocessed_data, room_readings, subjects_dir, subject, n_jobs, hasT1=hasT1)

# Run label averaging on fsaverage space. We use the aparc label
(label_epochs, times, label_epochs_psd, frequencies) = averageSourcesInLabel(subjects_dir, 'fsaverage', fsaverage_stcs, fsaverage_stcs_psd)