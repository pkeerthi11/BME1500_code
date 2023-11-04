import mne, scipy, os
from finnpy.source_reconstruction.utils import init_fs_paths, read_cortical_models, get_mri_subj_to_fs_avg_trans_mat, apply_mri_subj_to_fs_avg_trans_mat
from finnpy.source_reconstruction.coregistration_meg_mri import calc_coreg, get_rigid_transform, plot_coregistration
from finnpy.source_reconstruction.mri_anatomy import scale_anatomy
from finnpy.source_reconstruction.bem_model import calc_skull_and_skin_models, read_skull_and_skin_models, plot_skull_and_skin_models, calc_bem_model_linear_basis
from finnpy.source_reconstruction.source_mesh_model import create_source_mesh_model, match_source_mesh_model
from finnpy.source_reconstruction.forward_model import calc_forward_model, optimize_fwd_model
from finnpy.source_reconstruction.sensor_covariance import get_sensor_covariance
from finnpy.source_reconstruction.inverse_model import calc_inverse_model, apply_inverse_model

def finnpy_sourceReconstruction(meg_data_path, fs_subject_folder, subj_path, subj_name, empty_room_meas, fsaverage_path):
    
    # Loading data and initiating paths
    rec_meta_info = mne.io.read_info(meg_data_path)
    rec_meta_info['dig'][84:] = []  #### Remove EEG
    init_fs_paths(fs_subject_folder)
    
    # Calculate coregistration between MEG sensors and MRI 
    (coreg_rotors, meg_pts) = calc_coreg(subj_name, subj_path, rec_meta_info, registration_scale_type = "free")
    rigid_mri_to_head_trans = get_rigid_transform(coreg_rotors)
    
    # Reslice anatomy in sensor coordinates 
    scale_anatomy(subj_path, subj_name, coreg_rotors[6:9])
    
    # Get rigid and inverse transforms 
    rigid_mri_to_meg_trans = get_rigid_transform(coreg_rotors)
    rigid_meg_to_mri_trans = scipy.linalg.inv(rigid_mri_to_meg_trans)
    
    # Calculate sull and skin models with watershed and read them in
    calc_skull_and_skin_models(subj_path, subj_name, preflood_height = 25, overwrite = False)
    (ws_in_skull_vert, ws_in_skull_faces,
     ws_out_skull_vert, ws_out_skull_faces,
     ws_out_skin_vect, ws_out_skin_faces) = read_skull_and_skin_models(subj_path, subj_name)
    
    # Calculate BEM 
    print("Calculating BEM model")
    (in_skull_reduced_vert, in_skull_faces,
     in_skull_faces_area, in_skull_faces_normal,
     bem_solution) = calc_bem_model_linear_basis(ws_in_skull_vert, ws_in_skull_faces)
    
    # Calculate cortical surface model 
    (lh_white_vert, lh_white_faces,
    rh_white_vert, rh_white_faces,
    lh_sphere_vert,
    rh_sphere_vert) = read_cortical_models(subj_path)
    (octa_model_vert, octa_model_faces) = create_source_mesh_model()
    (lh_white_valid_vert, rh_white_valid_vert) = match_source_mesh_model(lh_sphere_vert, 
                                                                         rh_sphere_vert, 
                                                                         octa_model_vert)
    
    # Forward model 
    (fwd_sol, lh_white_valid_vert, rh_white_valid_vert) = calc_forward_model(lh_white_vert, rh_white_vert, 
                                                                             rigid_meg_to_mri_trans, rigid_mri_to_meg_trans, 
                                                                             rec_meta_info, in_skull_reduced_vert, 
                                                                             in_skull_faces, in_skull_faces_normal, 
                                                                             in_skull_faces_area, bem_solution, 
                                                                             lh_white_valid_vert, rh_white_valid_vert)
    
    # Optimize forward model 
    print("Optimizing forward model orientation")
    optimized_fwd_sol = optimize_fwd_model(lh_white_vert, lh_white_faces, lh_white_valid_vert, rh_white_vert, rh_white_faces, rh_white_valid_vert, fwd_sol, rigid_mri_to_meg_trans)
    
    # Sensor covariance
    print("Calculating sensor covariance")
    cov_folder = os.path.join(subj_path, 'cov_data')
    os.system('mkdir %s' % cov_folder)
    cov_path = os.path.join(cov_folder, 'covariates')
    (sensor_cov_eigen_val, sensor_cov_eigen_vec, sensor_cov_names) = get_sensor_covariance(empty_room_meas, cov_path, overwrite=False)
    
    # Inverse model 
    print("Calculating inverse model")
    (inv_trans, noise_norm) = calc_inverse_model(sensor_cov_eigen_val, sensor_cov_eigen_vec, sensor_cov_names, optimized_fwd_sol, rec_meta_info)
    
    # transformation to fsaverage
    print("Calculating transformation to fs-average")
    (fs_avg_trans_mat, src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert) = get_mri_subj_to_fs_avg_trans_mat(lh_white_valid_vert, rh_white_valid_vert, octa_model_vert, subj_path, fsaverage_path , overwrite=False)
    
    # Load raw data and convert to source and fsaverage spaces 
    raw_data = mne.io.read_raw_fif(meg_data_path)
    sensor_data = raw_data.get_data()
    sensor_data = sensor_data[0:306, :]
    source_data = apply_inverse_model(sensor_data, inv_trans, noise_norm)
    fsaverage_space = apply_mri_subj_to_fs_avg_trans_mat(fs_avg_trans_mat, source_data)
    
    return (source_data, fsaverage_space)
    
    # # Visualize everything 
    # plot_coregistration(rigid_mri_to_head_trans, rec_meta_info, meg_pts, subj_path)
    # plot_skull_and_skin_models(ws_in_skull_vert, ws_in_skull_faces,
    #                             ws_out_skull_vert, ws_out_skull_faces,
    #                             ws_out_skin_vect, ws_out_skin_faces,
    #                             subj_path)