import mne 

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
        for stc, stc_psd in zip(stcs,stcs_psd):
            label_epochs[i.name].append(stc.in_label(i))
            label_epochs_psd[i.name].append(stc_psd.in_label(i))
