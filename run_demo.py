from preprocessData import preprocessData
from mne_sourceReconstruction import mne_sourceReconstruction

# path to data
data = '/Users/ozenctaskin/MNE_DATA/MNE-sample-data/MEG/sample/sample_audvis_raw.fif'
room_noise = '/Users/ozenctaskin/MNE_DATA/MNE-sample-data/MEG/sample/ernoise_raw.fif'

# Subject paths
subjects_dir = '/Applications/freesurfer/7.4.1/subjects/'
subject = "sample"

preprocessed_data = preprocessData(data, 250, 3, plot_diagnostic=False)
room_readings = preprocessData(data, 250, 0, plot_diagnostic=False)

(stcs, fsaverage_stcs) = mne_sourceReconstruction(preprocessed_data, room_readings, subjects_dir, subject)
