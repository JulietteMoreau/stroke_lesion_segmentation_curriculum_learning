#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:12:27 2024

@author: moreau
"""

import nibabel as nib
import numpy as np
import os
import scipy
import math
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib.pyplot as plt


def calcul_fisher(image, masque):
    lesion=[]
    cerveau=[]
    
    CT = image.copy()
    ·# if modality is CT: threshold HU values
    CT = ((CT<80) & (CT>15))*image
        
    # normalize        
    CT = (CT-np.min(CT))/(np.max(CT)-np.min(CT))
    
    # calculate only if there is a lesion according to the mask
    if np.count_nonzero(masque) != 0:  
        # calculate the center of mass of the brain on the slice
        com = scipy.ndimage.center_of_mass(CT)
        
        # calculate the angle of the longitudinal fissure and the vertical of the image
        angle=0
        X = [0,CT.shape[0]]
        Y = [0,0]
        
        ecart = np.arctan(angle)*CT.shape[0]/2
        Y[0] = com[0]-ecart 
        Y[1] = com[0]+ecart
        
        coef = (Y[1]-Y[0])/(X[1]-X[0])
        oo = com[0]-coef*com[1]
        
        # find the side of the lesion
        cote_droit = True
        if np.count_nonzero(masque[:masque.shape[0]//2,:])<np.count_nonzero(masque[masque.shape[0]//2:,:]):
            cote_droit = False
        
        droite = []
        for i in range(CT.shape[0]):
            droite.append(coef*i+oo)
    
        # remove all information in the opposite hemisphere
        if cote_droit:
            for i in range(CT.shape[0]):
                for j in range(CT.shape[1]):
                    if i > droite[j]:
                        CT[i,j]=0
        else:
             for i in range(CT.shape[0]):
                for j in range(CT.shape[1]):
                    if i < droite[j]:
                        CT[i,j]=0

        # keep pixel values in the lesion and in the healthy tissue
        for i in range(CT.shape[0]):
            for j in range(CT.shape[1]):
                if CT[i,j]!=0:
                    if masque[i,j]!=0:
                        lesion.append(CT[i,j])
                    else:
                            cerveau.append(CT[i,j])
        
        # calculate fisher ratio
        moy_lesion = np.mean(lesion)
        ecart_type_lesion = np.std(lesion)
        moy_oppose = np.mean(cerveau)
        ecart_type_oppose = np.std(cerveau)
        return (moy_oppose - moy_lesion)**2/(ecart_type_oppose + ecart_type_lesion)


    
directory_CT = "/path/to/3D/volumes/"
directory_GT = "/path/to/3D//masks/"

pat = os.listdir(directory_CT)

volume = {}
area = {}
fisher = {}
    
for p in pat:
    # load image
    scan = nib.load(os.path.join(directory_CT, p))
    scan = scan.get_fdata()
    # load mask
    ref = nib.load(os.path.join(directory_GT, p))
    voxel_dims = (ref.header["pixdim"])[1:4] # get voxel size
    ref = ref.get_fdata()
    
    volume = np.count_nonzero(ref)*np.prod(voxel_dims)
    volume[p] = v
    
    for c in range(scan.shape[2]):
        
        no_slice = str(c)
        if len(no_slice)==1:
            no_slice = '00'+str(c)
        elif len(no_slice)==2:
            no_slice = '0'+str(c)
        slice_name = p[:-7]+'-slice'+no_slice
            
        ref2D = ref[:,:,c]
        a = np.count_nonzero(ref2D)*np.prod(voxel_dims[:2])
        a[slice_name] = a
        
        scan2D = scan[:,:,c]
        fisher[slice_name]=calcul_fisher(scan2D, ref2D)
            
        
df = pd.DataFrame({
    'slice': area.keys(),
    'area': area.values(),
    'fisher': [fisher[k] for k in area.keys()]
})
        
df.to_excel('/path/to/output/2D_features.xlsx', index=False)


df = pd.DataFrame({
    'patient': volume.keys(),
    'volume': area.values()
})
        
df.to_excel('/path/to/output/3D_volume.xlsx', index=False)
