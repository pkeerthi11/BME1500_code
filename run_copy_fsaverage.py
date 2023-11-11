# This script copies fsaverage for the subjects who do not have T1 

from finnpy.source_reconstruction.mri_anatomy import copy_fs_avg_anatomy
import os

subject_folder = '/home/bme1500trd/workspace/data/HC'
fs_subj_path = '/home/bme1500trd/freesurfer/subjects/'
for i in os.listdir(subject_folder):
    copy_fs_avg_anatomy(fs_subj_path,i,i)