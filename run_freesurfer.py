# This script runs freesurfer recon-all for all subjects that have T1
import os    

# Subject list
subjects = ['al0067a',
            'al0064a',
            'al0053a',
            'al0051a',
            'al0049a',
            'al0047a',
            'al0043a',
            'al0042a',
            'al0040a',
            'al0039a',
            'al0029a',
            'al0011a']

# Loop through subjects, get niftis and run recon-all 
mri_folder = '/home/bme1500trd/workspace/data/TRD/MRI'
for i in subjects:
    subject_folder = os.path.join(mri_folder, i, 'dcm')
    for sub in os.listdir(subject_folder):
        if sub[-3:] == 'nii':
            mri_image = os.path.join(subject_folder, sub)
            os.system('recon-all -s %s -i %s -all -threads 8' % (i, mri_image))