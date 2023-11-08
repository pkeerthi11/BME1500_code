# Need pyvistaqt

import matplotlib.pyplot as plt
import numpy as np
import mne, os

def mne_sourceReconstruction(preprocessed_epoched_data, preprocessed_room_readings, subjects_dir, subject):

########################## Setup paths ########################################

    # Create a folder for diagnostic plots
    plots_folder = os.path.join(subjects_dir,subject,'plots')
    if not os.path.exists(plots_folder):
        os.system('mkdir %s' % plots_folder)    

    # Create a folder for intermediate files produced at each step in this 
    # script. We want this folder in the freesurfer path like the plots folder.
    intermediate_folder = os.path.join(subjects_dir,subject,'sourceRecIntermediateFiles')
    if not os.path.exists(intermediate_folder):
        os.system('mkdir %s' % intermediate_folder)        
    
############################ Watershed BEM ####################################

    # BEM
    mne.bem.make_watershed_bem(subject,subjects_dir,overwrite=True)
    plot_bem_kwargs = dict(
        subject=subject,
        subjects_dir=subjects_dir,
        brain_surfaces="white",
        orientation="coronal",
        slices=[50, 100, 150, 200],
        show=False
    )
    
    fig = mne.viz.plot_bem(**plot_bem_kwargs)
    fig.savefig(os.path.join(plots_folder, 'watershed_results.png'))

######################### Registration ########################################

    # Make scalp surfaces for visualization and registration 
    mne.bem.make_scalp_surfaces(subject,subjects_dir)

    # Prepare registration and plotting arguments. FIducials are set to auto
    coreg = mne.coreg.Coregistration(preprocessed_epoched_data.info, subject, subjects_dir, fiducials='auto')
    plot_kwargs = dict(
        subject=subject,
        subjects_dir=subjects_dir,
        surfaces="head-dense",
        dig=True,
        eeg=[],
        meg="sensors",
        show_axes=True,
        coord_frame="meg"
    )
    view_kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))

    # Initial fit
    coreg.fit_fiducials(verbose=True)

    # Refining with ICP
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)

    # Final fit 
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
    
    # Plot 
    fig = mne.viz.plot_alignment(preprocessed_epoched_data.info, trans=coreg.trans, **plot_kwargs)
    mne.viz.set_3d_view(fig, **view_kwargs)
    screenshot = fig.plotter.screenshot()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(screenshot, origin='upper')
    ax.set_axis_off()  # Disable axis labels and ticks
    fig.tight_layout()
    fig.savefig(os.path.join(plots_folder, 'sensor_registration_results.png'), dpi=150)
    
    dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
    print(
        f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
        f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
    )

    # Save coregistration 
    transformation = os.path.join(intermediate_folder, '%s-trans.fif' % subject)
    mne.write_trans(transformation, coreg.trans)

########################## Source Space ######################################

    src = mne.setup_source_space(subject, spacing="oct6", n_jobs=-1, subjects_dir=subjects_dir)
    print(src)
    fig = mne.viz.plot_alignment(
        subject=subject,
        subjects_dir=subjects_dir,
        surfaces="white",
        coord_frame="mri",
        src=src
    )
    screenshot = fig.plotter.screenshot()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(screenshot, origin='upper')
    ax.set_axis_off()  # Disable axis labels and ticks
    fig.tight_layout()
    fig.savefig(os.path.join(plots_folder, 'source_space.png'), dpi=150)    

########################## Forward model #####################################
    
    # BEM solution
    conductivity = (0.3,)  # for single layer
    # conductivity = (0.3, 0.006, 0.3)  # for three layers
    model = mne.make_bem_model(
        subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(model)
    mne.write_bem_solution(os.path.join(intermediate_folder, '%s-bem-sol.fif' % subject), bem)
    
    # Forward model
    fwd = mne.make_forward_solution(
        preprocessed_epoched_data.info,
        trans=coreg.trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=-1,
        verbose=True,
    )
    print(fwd)
    mne.write_forward_solution(os.path.join(intermediate_folder, '%s-fwd.fif' % subject), bem)

############################ Noise Covariance #################################

    noise_cov = mne.compute_raw_covariance(preprocessed_room_readings, method=["shrunk", "empirical"], rank=None, verbose=True)
    fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, preprocessed_epoched_data.info, show=False)
    fig_cov.savefig(os.path.join(plots_folder, 'noise_cov.png'))
    fig_spectra.savefig(os.path.join(plots_folder, 'noise_spec.png'))
    
########################### Inverse Model #####################################

    # Build inverse operator 
    inverse_operator = mne.minimum_norm.make_inverse_operator(preprocessed_epoched_data.info, 
                                                              fwd, noise_cov, 
                                                              loose=0.2, depth=0.8)
    
    mne.minimum_norm.write_inverse_operator(os.path.join(intermediate_folder, '%s-inv.fif' % subject), inverse_operator)

    # Apply inverse - dSPM
    method = "dSPM"
    snr = 3.0
    lambda2 = 1.0 / snr**2
    stcs = mne.minimum_norm.apply_inverse_epochs(preprocessed_epoched_data,
                                                 inverse_operator,
                                                 lambda2,
                                                 method=method,
                                                 pick_ori=None,
                                                 verbose=True)

    # Calculate morph to fsaverage with the first epoch and then warp all epoches 
    # with this calculation.
    morph = mne.compute_source_morph(stcs[0], subject_from=subject, subject_to='fsaverage')  
    fsaverage_stcs = []
    for i in range(len(stcs)):
        print('Applying fsaverage transformation to epoch number: %s' % i)
        fsaverage_stcs.append(morph.apply(stcs[i]))
        
    return (stcs, fsaverage_stcs)
