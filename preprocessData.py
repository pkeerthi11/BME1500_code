import mne, os
import numpy as np
import matplotlib.pyplot as plt

def preprocessData(raw_data_path, downsample_to, epoch_size_sec, plot_diagnostic=True):
    
    # This function performs preprocessing on raw MEG data. 
    #
    # raw_data_path:   Path to MEG file
    # downsample_to:   Target frequency for downsampling. Initial sampling freq
    #                  is obtained from metadata automatically.
    # epoch_size_sec:  Epoch size in seconds. If set to 0, do not epoch
    # plot_diagnostic: Plot EEG and MEG channel PSDs before and after 
    #                  preprocessing
    #
    # This function produces epoched_data and saves a new fif file. If you want
    # to reject some bad epochs, first run epoched_data.plot() and then, select
    # the sections you want to remove. Once you close the plot, the selected 
    # sections are removed automatically. 
    
    # Read raw data
    raw_data = mne.io.read_raw_fif(raw_data_path, preload=True)
    
    # We downsample the raw data here, but we won't use this version. It's just 
    # for comparing plots against the preprocessed data at the end.
    if plot_diagnostic == True:
        raw_downsampled_for_plotting = raw_data.copy().resample(downsample_to)
    
    # Band pass filter data and get rid of frequencies below 0.5Hz and above 
    # 250Hz
    bandpassed_data = raw_data.copy().filter(0.5, 250, 
                                             phase='zero-double')

    # Notch filtering for the band pass filtered data. This time we get rid of 
    # frequencies between 20 and 120Hz with 20Hz intervals. 
    freqs_to_filter = np.arange(20,120,20)
    notched_data = bandpassed_data.copy().notch_filter(freqs_to_filter,
                                                       phase='zero-double')
    
    # Downsample the filtered data to specified frequency
    downsampled_data = notched_data.copy().resample(downsample_to)
    
    # Epoch downsampled and filtered data with the specified epoch sizes
    if epoch_size_sec != 0:
        preprocessed_data = mne.make_fixed_length_epochs(downsampled_data, 
                                                    epoch_size_sec)
    else:   
        preprocessed_data = downsampled_data
    
    # Plot PSD of raw and preprocessed data to make sure everything worked OK.
    if plot_diagnostic == True:
        raw_downsampled_for_plotting.plot_psd()
        downsampled_data.plot_psd()
    
    # Save processed data as fif
    if raw_data_path[-1] == '/' or raw_data_path[-1] == '\\':
        splitted_input_path = os.path.split(raw_data_path[:-1])
    else:
        splitted_input_path = os.path.split(raw_data_path)
    save_path = splitted_input_path[0]
    if epoch_size_sec != 0:
        new_file_path = os.path.join(save_path, 'preprocessed_' + splitted_input_path[1][:-3] + '-epo.fif')
    else:
        new_file_path = os.path.join(save_path, 'preprocessed_' + splitted_input_path[1][:-3] + '-raw.fif')
    preprocessed_data.save(new_file_path,overwrite=True)
    
    return preprocessed_data
    
    
    
    
    
    