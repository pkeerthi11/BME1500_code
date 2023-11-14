import mne

def morph_to_fsaverage(stcs, fsaverage_scr_path, hasT1, subjects_dir, subject):
    
    # Calculate morph to fsaverage with the first epoch and then warp all epoches 
    # with this calculation. If the subject didn't have T1, we use a bit of a
    # hacky solution by warping fsaverage to fsavarage to have same amount of 
    # vertices across subjects at the end by subjecting an interpolation. 
    src_to = mne.read_source_spaces(fsaverage_scr_path)
    
    if hasT1 == True:
        morph = mne.compute_source_morph(stcs[0], subject_from=subject, subject_to='fsaverage', subjects_dir=subjects_dir, src_to=src_to)  
    else:
        # Here we set source subject to fsaverage and morph fsaverage to itself
        # just to do the interpolation.
        for ver in stcs:
            ver.subject = 'fsaverage'
        morph = mne.compute_source_morph(stcs[0], subject_from='fsaverage', subject_to='fsaverage', subjects_dir=subjects_dir, src_to=src_to) 
    morphed_stcs = []
    for i in range(len(stcs)):
        print('Applying fsaverage transformation to epoch number: %s' % i)
        morphed_stcs.append(morph.apply(stcs[i]))
        
    return (morphed_stcs)