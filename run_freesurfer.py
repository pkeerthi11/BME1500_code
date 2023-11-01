# This script runs freesurfer recon-all for all subjects that have T1

from finnpy.source_reconstruction.utils import init_fs_paths   
from finnpy.source_reconstruction.mri_anatomy import extract_anatomy_from_mri_using_fs
import os    

# Set FS paths
init_fs_paths('/home/bme1500trd/freesurfer/subjects')

# Loop through subjects, get niftis and run recon-all 
mri_folder = '/home/bme1500trd/workspace/data/TRD/MRI'
for i in os.listdir(mri_folder):
    subject_folder = os.path.join(mri_folder, i, 'dcm')
    for sub in os.listdir(subject_folder):
        if sub[-3:] == 'nii':
            mri_image = os.path.join(subject_folder, sub)
            extract_anatomy_from_mri_using_fs(i, mri_image, overwrite = True)
