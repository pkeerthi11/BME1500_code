# Need autoreject: pip install autoreject

import mne, os
import numpy as np
import matplotlib.pyplot as plt
from autoreject import AutoReject, set_matplotlib_defaults
mne.utils.check_random_state(42)
set_matplotlib_defaults(plt)

def preprocessData(raw_data_path, downsample_to, epoch_size_sec, subjects_dir, subject, n_jobs=-1, auto_reject=False):
    
    # This function performs preprocessing on raw MEG data. 
    #
    # raw_data_path :  Path to MEG file
    # downsample_to :  Target frequency for downsampling. Initial sampling freq
    #                  is obtained from metadata automatically.
    # epoch_size_sec:  Epoch size in seconds. If set to 0, do not epoch
    # subjects_dir  :  Freesurfer subject directory. Just needed for plots. If
    #                  NA, do not plot anything.
    # subject       :  Subject name in freesurfer dir. Just for plotting. If NA
    #                  do not plot anything.
    # n_jobs        :  Number of cores. -1 uses all.
    # auto_reject   :  Run autorejection algorithm to fix epochs. Only works if
    #                  epoch_size_sec is not 0.
    #
    # This function produces preprocessed_data and saves a new fif file.
    # To reject some bad epochs manually, first run epoched_data.plot() and
    # then select the sections you want to remove. Once you close the plot, 
    # the selected sections are removed automatically. 
    
    # Create the plot folder if it doesn't exist
    if subjects_dir != 'NA' or subject != 'NA':
        plots_folder = os.path.join(subjects_dir,subject,'plots')
        if not os.path.exists(plots_folder):
            os.system('mkdir %s' % plots_folder)  
        
        preprocessed_plots = os.path.join(plots_folder, 'preprocessing')
        if not os.path.exists(preprocessed_plots):
            os.system('mkdir %s' % preprocessed_plots)  
    
    # Read raw data
    raw_data = mne.io.read_raw_fif(raw_data_path, preload=True)
    
    # Just get the MEG channels 
    picks = mne.pick_types(raw_data.info, meg=True)
    raw_data = raw_data.pick(picks)
    
    # We downsample the raw data here, but we won't use this version. It's just 
    # for plotting at the end.
    raw_downsampled_for_plotting = raw_data.copy().resample(downsample_to)
    
    # Low pass filter data and get rid of frequencies above 250Hz. We don't do
    # a high-pass to remove drifts. We detrend at the epoching level instead.
    bandpassed_data = raw_data.copy().filter(None, 250, 
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
        epoched_data = mne.make_fixed_length_epochs(downsampled_data.copy(), 
                                                    epoch_size_sec,
                                                    preload=False)
        
        # Detrend the epochs to get rid of drifts.
        epoched_data.detrend=0;
        epoched_data.load_data()
        
    else:   
        epoched_data = downsampled_data.copy()
    
    # Plot PSD of raw and preprocessed data to make sure everything worked OK.
    if subject != 'NA':
        fig = raw_downsampled_for_plotting.plot_psd(show=False)
        fig.savefig(os.path.join(preprocessed_plots, 'raw_psd.png'))
        fig = downsampled_data.plot_psd(show=False)
        fig.savefig(os.path.join(preprocessed_plots, 'preprocessed_psd.png'))    
    
    # Auto detect bad epochs. We can only do this if the data is epoched, so 
    # add this as an "if" condition as well.
    if auto_reject == True and epoch_size_sec != 0:
        n_interpolates = np.array([1, 4, 32])
        consensus_percs = np.linspace(0, 1.0, 11)
        ar = AutoReject(n_interpolates, consensus_percs,
                        thresh_method='random_search', random_state=42, n_jobs=n_jobs)
        ar.fit(epoched_data)
        preprocessed_data, reject_log = ar.transform(epoched_data, return_log=True)
        
        # Plot cleaning results
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        
        for ax in axes:
            ax.tick_params(axis='x', which='both', bottom='off', top='off')
            ax.tick_params(axis='y', which='both', left='off', right='off')
        
        evoked = epoched_data.copy().average()
        cleaned_evoked = preprocessed_data.copy().average()
        
        ylim = dict(grad=(-170, 200))
        evoked.pick_types(meg='grad', exclude=[])
        evoked.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
        axes[0].set_title('Before autoreject')
        cleaned_evoked.pick_types(meg='grad', exclude=[])
        cleaned_evoked.plot(exclude=[], axes=axes[1], ylim=ylim)
        axes[1].set_title('After autoreject')
        plt.tight_layout()
        fig.savefig(os.path.join(preprocessed_plots, 'autoreject_evoked.png'))
         
        fig = ar.get_reject_log(epoched_data).plot()
        fig.savefig(os.path.join(preprocessed_plots, 'autoreject_log.png'))
    else:
        preprocessed_data = epoched_data.copy()
    
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
     
    # Close all plots so we don't overwhelm the memory in bulk runs.
    plt.close('all')
    
    return preprocessed_data
    
    
    
    
    
    