#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:59:27 2024

@author: prerana
"""

import os
import glob
import subprocess

pd_harddrive_path = '/mnt/patient_data/pd'
pd_storage_path = '/storage/prerana/pd'

pd_harddrive_files = os.listdir(pd_harddrive_path)
pd_storage_files = os.listdir(pd_storage_path)

missing_folders = list(set(pd_harddrive_files).difference(pd_storage_files))

for folder in missing_folders:
    if folder[-1] == 'b':
        if folder[:-1]+['a'] in pd_storage_files:
            fullpath_a = os.path.join(pd_storage_path, folder)
            fullpath_b = os.path.join(pd_harddrive_path, folder[:-1]+['a'])
            subprocess.run(['cp','-r', fullpath_a, fullpath_b])
            
            missing_folders.remove(folder)
            
print('Remaining missing folders')

#al0013a
#al0021a
#al0021b
#Nothing in al0060a
#al0065a