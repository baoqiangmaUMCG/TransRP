# The code is created by Baoqiang Ma to extract a bouding box including primary tumor and lymph nodes for each patient, as well as obtain PET nii without brain
# This code refered to the official bouding box extraction code in HECKTOR 2021 (https://github.com/voreille/hecktor/blob/master/src/data/bounding_box.py) 


import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.ndimage.measurements import label

import pandas as pd


def bbox_auto(sitk_pt, pID, output_shape=(144, 144, 144), th=3):
    """Find a bounding box automatically based on the SUV

    Arguments:
        vol_pt {numpy array} -- The PET volume on which to compute the bounding box
        px_spacing_pt {tuple of float} -- The spatial resolution of the PET volume 
        px_origin_pt {tuple of float} -- The spatial position of the first voxel in the PET volume 

    Keyword Arguments:
        shape {tuple} -- The ouput size of the bounding box in millimeters
        th {float} -- [description] (default: {3})

    Returns:
        [type] -- [description]
    """
    np_pt = np.transpose(sitk.GetArrayFromImage(sitk_pt), (2, 1, 0))
    px_spacing_pt = sitk_pt.GetSpacing()
    px_origin_pt = sitk_pt.GetOrigin()

    output_shape_pt = tuple(e1 / e2
                            for e1, e2 in zip(output_shape, px_spacing_pt))
    # Gaussian smooth
    np_pt_gauss = gaussian_filter(np_pt, sigma=3)
    # auto_th: based on max SUV value in the top of the PET scan
    #auto_th = np.max(np_pt[:, :, np.int(np_pt.shape[2] * 2 // 3):]) / 4
    #print('auto_th = ', auto_th)
    # OR fixed threshold
    np_pt_thgauss = np.where(np_pt_gauss > th, 1, 0)
    # Find brain as biggest blob AND not in lowest third of the scan
    labeled_array, _ = label(np_pt_thgauss)
    try:
        np_pt_brain = labeled_array == np.argmax(
            np.bincount(labeled_array[:, :,
                                      np_pt.shape[2] * 2 // 3:].flat)[1:]) + 1
    except:
        print('th too high?')
        # Quick fix just to pass for all cases
        th = 0.1
        np_pt_thgauss = np.where(np_pt_gauss > th, 1, 0)
        labeled_array, _ = label(np_pt_thgauss)
        np_pt_brain = labeled_array == np.argmax(
            np.bincount(labeled_array[:, :,
                                      np_pt.shape[2] * 2 // 3:].flat)[1:]) + 1
                                      


    # Find lowest voxel of the brain and box containing the brain
    z = np.min(np.argwhere(np.sum(np_pt_brain, axis=(0, 1))))
    y1 = np.min(np.argwhere(np.sum(np_pt_brain, axis=(0, 2))))
    y2 = np.max(np.argwhere(np.sum(np_pt_brain, axis=(0, 2))))
    x1 = np.min(np.argwhere(np.sum(np_pt_brain, axis=(1, 2))))
    x2 = np.max(np.argwhere(np.sum(np_pt_brain, axis=(1, 2))))

    # Center bb based on this brain segmentation
    zshift = 30 // px_spacing_pt[2]
    if z - (output_shape_pt[2] - zshift) < 0:
        zbb = (0, output_shape_pt[2])
    elif z + zshift > np_pt.shape[2]:
        zbb = (np_pt.shape[2] - output_shape_pt[2], np_pt.shape[2])
    else:
        zbb = (z - (output_shape_pt[2] - zshift), z + zshift)

    yshift = 30 // px_spacing_pt[1]
    if np.int((y2 + y1) / 2 - yshift - np.int(output_shape_pt[1] / 2)) < 0:
        ybb = (0, output_shape_pt[1])
    elif np.int((y2 + y1) / 2 - yshift -
                np.int(output_shape_pt[1] / 2)) > np_pt.shape[1]:
        ybb = np_pt.shape[1] - output_shape_pt[1], np_pt.shape[1]
    else:
        ybb = ((y2 + y1) / 2 - yshift - output_shape_pt[1] / 2,
               (y2 + y1) / 2 - yshift + output_shape_pt[1] / 2)

    if np.int((x2 + x1) / 2 - np.int(output_shape_pt[0] / 2)) < 0:
        xbb = (0, output_shape_pt[0])
    elif np.int((x2 + x1) / 2 -
                np.int(output_shape_pt[0] / 2)) > np_pt.shape[0]:
        xbb = np_pt.shape[0] - output_shape_pt[0], np_pt.shape[0]
    else:
        xbb = ((x2 + x1) / 2 - output_shape_pt[0] / 2,
               (x2 + x1) / 2 + output_shape_pt[0] / 2)

    z_pt = np.asarray(zbb)
    y_pt = np.asarray(ybb)
    x_pt = np.asarray(xbb)

    # In the physical dimensions
    z_abs = z_pt * px_spacing_pt[2] + px_origin_pt[2]
    y_abs = y_pt * px_spacing_pt[1] + px_origin_pt[1]
    x_abs = x_pt * px_spacing_pt[0] + px_origin_pt[0]

    bb = np.asarray((x_abs, y_abs, z_abs)).flatten()
    
    
    # save image_withoutbrain    
    np_pt_brain =   binary_dilation(np_pt_brain,  iterations= 5 )
    np_pt_withoutbrain  =  np_pt * (1 - np_pt_brain)                                
    np_pt_withoutbrain = sitk.GetImageFromArray(np.transpose(np_pt_withoutbrain, (2 , 1, 0)))  
    np_pt_withoutbrain.CopyInformation(sitk_pt)
    
    # training set
    sitk.WriteImage(np_pt_withoutbrain, './Data/hecktor2022/imagesTr_PT_nobrain/' + str(pID) + "__PT_nobrain.nii.gz")
    # tetsing set
    #sitk.WriteImage(np_pt_withoutbrain, './Data/hecktor2022_testing/imagesTs_PT_nobrain/' + str(pID) + "__PT_nobrain.nii.gz")
    
    return bb


# training set
patient_info_path = './Data/hecktor2022_patient_info_training.csv'
pet_path = './Data/hecktor2022/imagesTr/'
bb_save_path = './Data/hecktor2022/bb_box/bb_box_training.csv'
'''
# testing set
patient_info_path = ./Data/hecktor2022_clinical_info_testing.csv'
pet_path = './Data/hecktor2022_testing/imagesTs/'
bb_save_path = './Data/hecktor2022_testing/bb_box/bb_box_testing.csv'
'''
bb_df = pd.DataFrame(columns=['PatientID', 'x1', 'x2', 'y1', 'y2', 'z1', 'z2'])

patinets_list = list (pd.read_csv(patient_info_path)['PatientID'])
for pID in patinets_list:
    bb  = bbox_auto(sitk.ReadImage(pet_path + str(pID)+'__PT.nii.gz'), pID)
    print (bb)
    bb_df = bb_df.append(
            {
            'PatientID': pID,
            'x1': bb[0],
            'x2': bb[1],
            'y1': bb[2],
            'y2': bb[3],
            'z1': bb[4],
            'z2': bb[5],
            },ignore_index=True)
bb_df.to_csv(bb_save_path, index = False)