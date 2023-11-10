import mne 
import numpy as np

def averageSourcesInLabel(subjects_dir, subject, stcs, stcs_psd):
    
    # Read in aparc
    labels = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir)
    
    # Drop the last label as it is unknown
    labels = labels[:-1]
    
    # Create two empty label dictionaries for standard epochs and psd epochs
    label_epochs = {};
    label_epochs_psd = {}
    
    # Loop through each label and each source list and save. We also get the
    # mean of all channels that falls into each label. 
    for i in labels:
        label_epochs[i.name] = []
        label_epochs_psd[i.name] = []
        for stc, psd in zip(stcs,stcs_psd):
            label_epochs[i.name].append(np.mean(stc.in_label(i).data, axis=0))
            label_epochs_psd[i.name].append(np.mean(psd.in_label(i).data, axis=0))

    # Get the times on the x axis of the stcs and frequencies of the PSDs. 
    # Note that both of these are stored in the same variable (times) in MNE.
    times = stcs[0].times
    frequencies = stcs_psd[0].times

    return (label_epochs, times, label_epochs_psd, frequencies)