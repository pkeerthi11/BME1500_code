import mne 
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# Channel to plot 
channel_to_plot = 40

# Read data
sampling_rate = 2000
downsampling_factor = 8
datapath = '/Users/ozenctaskin/Downloads/SubjectRest/SubjectRest.ds/'
raw_data_container = data = mne.io.read_raw_ctf(datapath, preload=True)
raw_data = raw_data_container.get_data()
data_shape = np.shape(raw_data)
data_size = data_shape[1] 

# Bandpass to get rid of everything below 0.5 and above 250Hz
low_cut = 0.5
high_cut = 250
bandpass_filtered_data = mne.filter.filter_data(raw_data,sampling_rate,low_cut,
                                                high_cut, phase='zero-double')

# # Check out psd of two channels

# f1, Pxx_spec1 = signal.welch(bandpass_filtered_data[channel_to_plot,:], sampling_rate,
#                              nfft=sampling_rate, nperseg=sampling_rate,
#                              noverlap=sampling_rate//2) 
#
# plt.semilogy(f1, Pxx_spec1)

# Notch filters ####### Do we set the phase here as well??
freqs = np.arange(20,120,20)
notched_data = mne.filter.notch_filter(bandpass_filtered_data, sampling_rate, 
                                       freqs, phase='zero-double')

# Downsample by factor of 8
downsampled_data = mne.filter.resample(bandpass_filtered_data, down=downsampling_factor)
sampling_rate = sampling_rate//downsampling_factor
new_data_size = data_size//downsampling_factor

# Epoch the data
epoch_length_insec = 2 # in secs
new_data_shape = np.shape(downsampled_data)
new_data_size = new_data_shape[1]
bins = np.arange(0,new_data_size,epoch_length_insec*sampling_rate)
epoched_data = np.split(downsampled_data[45,:], bins)




