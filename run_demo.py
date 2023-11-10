from preprocessData import preprocessData
from mne_sourceReconstruction import mne_sourceReconstruction
from averageSourcesInLabel import averageSourcesInLabel
import mne

# path to data
data = '/Users/ozenctaskin/MNE_DATA/MNE-sample-data/MEG/sample/sample_audvis_raw.fif'
room_noise = '/Users/ozenctaskin/MNE_DATA/MNE-sample-data/MEG/sample/ernoise_raw.fif'

# Subject paths
subjects_dir = '/Applications/freesurfer/7.4.1/subjects/'
subject = "sample"

# Run values 
n_jobs=4

preprocessed_data = preprocessData(data, 250, 3, subjects_dir, subject, -1, True)
room_readings = preprocessData(data, 250, 0, 'NA', 'NA', -1, False)

(stcs, stcs_psd, fsaverage_stcs, fsaverage_stcs_psd) = mne_sourceReconstruction(preprocessed_data, room_readings, subjects_dir, subject, n_jobs)

(label_epochs, times, label_epochs_psd, frequencies) = averageSourcesInLabel(subjects_dir, 'fsaverage', fsaverage_stcs, fsaverage_stcs_psd)