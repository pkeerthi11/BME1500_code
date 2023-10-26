import mne
import numpy as np
import matplotlib.pyplot as plt

def preprocessData(raw_data, downsample_to, epoch_size_sec, plot_diagnostic=True):
    
    # Read raw data
    raw_data = mne.io.read_raw_ctf(raw_data, preload=True)
    
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
    epoched_data = mne.make_fixed_length_epochs(downsampled_data, 
                                                epoch_size_sec)
    
    # Plot PSD of raw and preprocessed data to make sure everything worked OK.
    if plot_diagnostic == True:
        fig, axs = plt.subplots(2,2)
        fig.suptitle('Raw vs. preprocessed data PSDs')
        raw_downsampled_for_plotting.plot_psd(ax=[axs[0,0], axs[0,1]])
        downsampled_data.plot_psd(ax=[axs[1,0], axs[1,1]])
    
    return epoched_data
    
    
    
    
    
    