import pandas as pd
import numpy as np
import torch
import os
import monai
import matplotlib.pyplot  as plt
import random
import torch.nn as nn

from models.ResNet18 import ResNet18_m3
#from models.DenseNet121 import DenseNet121_m3
from models.ViT import ViT, ViT_m1, ViT_m2, ViT_m3
from models.TransRP import TransRP_ResNet18, TransRP_ResNet18_m1, TransRP_ResNet18_m2, TransRP_ResNet18_m3, TransRP_DenseNet121, TransRP_DenseNet121_m1, TransRP_DenseNet121_m2, TransRP_DenseNet121_m3 ,clc

# sampler of oversampling the training set
def get_oversampler(endpoint_info, train_ID, event_name):
        label_raw_train = np.array(list(endpoint_info.loc[endpoint_info['PatientID'].isin(train_ID)][event_name]))
        weights = 1/ np.array([np.count_nonzero(1 - label_raw_train), np.count_nonzero(label_raw_train)]) # check event and no events samples numbers
        samples_weight = np.array([weights[t] for t in label_raw_train])
        samples_weight = torch.from_numpy(samples_weight) 
        sampler  = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        return sampler
    
def seed_torch(seed=42,deter=False):
    # `deter` means use deterministic algorithms for GPU training reproducibility, 
    #if set `deter=True`, please set the environment variable `CUBLAS_WORKSPACE_CONFIG` in advance
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    from monai.utils import set_determinism
    set_determinism(seed=seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def get_data_dict_withclc_hecktor(patientsID , opt, endpoint_info): 
    # Gender Age\tWeight\tHPV status\tSurgery\tChemotherapy\t
    # new
    endpoint_info['CenterID'] = endpoint_info['CenterID']/ 7.
    endpoint_info['Age'] = endpoint_info['Age'] / 100.
    endpoint_info['Gender'] = endpoint_info['Gender'] # MAN 1, FEMALE 0
    endpoint_info['Weight'] = endpoint_info['Weight'] / 150.
    endpoint_info['HPV status'] = endpoint_info['HPV status']  /2. 
    endpoint_info['Surgery'] = endpoint_info['Surgery'] /2.
    endpoint_info['Chemotherapy'] = endpoint_info['Chemotherapy'] 
    
    data_dict = []
    for i , pID in enumerate(patientsID):
        
        data_single_dict =  {str(image) : os.sep.join([opt.data_path, pID + '__' + str(image) +'.nii.gz']) for image in opt.input_modality  }      
        clc_data_sub = np.array(endpoint_info.loc[endpoint_info['PatientID'] == pID][['CenterID', 'Gender' ,'Age','Weight','HPV status','Surgery','Chemotherapy']])[0].astype(float)
        
        data_single_dict['clinical']=  torch.tensor(clc_data_sub,dtype = torch.float)
                
        event  = np.array(endpoint_info.loc[endpoint_info['PatientID'] == pID][opt.event_name])[0].astype(float)
        event_time  = np.array(endpoint_info.loc[endpoint_info['PatientID'] == pID][opt.event_time_name])[0].astype(float)
        
        data_single_dict[opt.event_name]=  torch.tensor(event,dtype = torch.float)
        data_single_dict[opt.event_time_name]=  torch.tensor(event_time,dtype = torch.float)
        
        data_dict.append(data_single_dict)
    return data_dict    
    
def get_model(opt): 
    
    input_channel =  len(opt.input_modality)
    
    if opt.model == 'clc': # DeepSurv model   
        return clc(in_channels = 512,in_channels_cnn = input_channel, img_size = (5,5,5),patch_size = (1,1,1),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None',add_clc =True)      
    
    if opt.model == 'ResNet18':
        return monai.networks.nets.resnet18(spatial_dims=3, n_input_channels = input_channel, num_classes=  1, conv1_t_stride = 2  ) 
    if opt.model == 'ResNet18_m3':
        return ResNet18_m3(spatial_dims=3, n_input_channels = input_channel, num_classes=  1, conv1_t_stride = 2)  
                                            
    if opt.model == 'DenseNet121':
        return monai.networks.nets.DenseNet121(spatial_dims=3, in_channels= input_channel, out_channels=  1) 
    if opt.model == 'DenseNet121_m3':
        from models.DenseNet121 import DenseNet121                                                               
        return DenseNet121(spatial_dims=3, in_channels= input_channel, out_channels=  1)     
                                            
    if opt.model == 'ViT':      
        return ViT(in_channels = input_channel, img_size = (96,96,96),patch_size = (16,16,16),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None')
    if opt.model == 'ViT_m1':      
        return ViT_m1(in_channels = input_channel, img_size = (96,96,96),patch_size = (16,16,16),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None',add_clc = 'True')    
    if opt.model == 'ViT_m2':      
        return ViT_m2(in_channels = input_channel, img_size = (96,96,96),patch_size = (16,16,16),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None',add_clc = 'True')                 
    if opt.model == 'ViT_m3':      
        return ViT_m3(in_channels = input_channel, img_size = (96,96,96),patch_size = (16,16,16),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None',add_clc = 'True')     

    if opt.model == 'TransRP_ResNet18':      
        #return cnn_ViT(in_channels = input_channel, img_size = (144,144,144),patch_size = (16,16,16),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None')       
        return TransRP_ResNet18(in_channels = 512,in_channels_cnn = input_channel, img_size = (6,6,6),patch_size = (1,1,1),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None')                  
    if opt.model == 'TransRP_ResNet18_m1':      
        return TransRP_ResNet18_m1(in_channels = 512,in_channels_cnn = input_channel, img_size = (6,6,6),patch_size = (1,1,1),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None',add_clc =True)          
    if opt.model == 'TransRP_ResNet18_m2':      
        return TransRP_ResNet18_m2(in_channels = 512,in_channels_cnn = input_channel, img_size = (6,6,6),patch_size = (1,1,1),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None',add_clc =True)            
    if opt.model == 'TransRP_ResNet18_m3':      
        return TransRP_ResNet18_m3(in_channels = 512,in_channels_cnn = input_channel, img_size = (6,6,6),patch_size = (1,1,1),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None',add_clc =True)                

    if opt.model == 'TransRP_DenseNet121':      
        return TransRP_DenseNet121(in_channels = 1024,in_channels_cnn = input_channel, img_size = (6,6,6),patch_size = (1,1,1),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None')         
    if opt.model == 'TransRP_DenseNet121_m1':      
        return TransRP_DenseNet121_m1(in_channels = 1024,in_channels_cnn = input_channel, img_size = (6,6,6),patch_size = (1,1,1),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None',add_clc =True)         
    if opt.model == 'TransRP_DenseNet121_m2':      
        return TransRP_DenseNet121_m2(in_channels = 1024,in_channels_cnn = input_channel, img_size = (6,6,6),patch_size = (1,1,1),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None',add_clc =True)             
    if opt.model == 'TransRP_DenseNet121_m3':      
        return TransRP_DenseNet121_m3(in_channels = 1024,in_channels_cnn = input_channel, img_size = (6,6,6),patch_size = (1,1,1),  pos_embed='conv', classification= False, num_classes=1, spatial_dims=3, post_activation ='None',add_clc =True)                

def plot_images(arr_list, nr_images, figsize, cmap_list, colorbar_title_list, filename, vmin_list, vmax_list):
    """
    Plot slices of multiple arrays. Each Numpy on a different row, e.g. CT (row 1), RTDOSE (row 2) and
    segmentation_map (row 3).
    """
    # Make sure that every input array has the same number of slices
    nr_slices = arr_list[0].shape[0]
    for i in range(1, len(arr_list)):
        assert nr_slices == arr_list[i].shape[0]

    # Initialize variables
    if nr_images is None:
        nr_images = nr_slices

    # Make sure that nr_images that we want to plot is greater than or equal to the number of slices available
    if nr_slices < nr_images:
        nr_images = nr_slices
    slice_indices = np.linspace(0, nr_slices - 1, num=nr_images)

    # Only consider unique values
    slice_indices = np.unique(slice_indices.astype(int))

    # Determine number of columns and rows
    num_cols = nr_images
    num_rows = len(arr_list)

    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=tuple(figsize))

    for row, arr in enumerate(arr_list):
        cmap = cmap_list[row]
        colorbar_title = colorbar_title_list[row]

        # Colormap
        # Data range
        vmin = vmin_list[row] if vmin_list[row] is not None else arr.min()
        vmax = vmax_list[row] if vmax_list[row] is not None else arr.max()
        # Label ticks
        # '+1' because end of interval is not included
        # ticks = np.arange(vmin, vmax + 1, ticks_steps_list[row]) if ticks_steps_list[row] is not None else None

        for i, idx in enumerate(slice_indices):
            # Consider the first and last slice
            if i == 0:
                idx = 0
            if i == nr_images - 1:
                idx = nr_slices - 1

            idx = int(idx)
            if num_rows >= 2:
              im = ax[row, i].imshow(arr[idx, ...], cmap=cmap, vmin=vmin, vmax=vmax)
              ax[row, i].axis('off')
            else:
              im = ax[i].imshow(arr[idx, ...], cmap=cmap, vmin=vmin, vmax=vmax)
              ax[i].axis('off')

        plt.tight_layout()

        # Add colorbar
        fig.subplots_adjust(right=0.8)
        max_height = 0.925
        min_height = 1 - max_height
        length = max_height - min_height
        length_per_input = length / num_rows
        epsilon = 0.05
        bottom = max_height - (row + 1) * length_per_input + epsilon / 2
        cbar = fig.add_axes(rect=[0.825, bottom, 0.01, length_per_input - epsilon])
        cbar.set_title(colorbar_title)
        fig.colorbar(im, cax=cbar) # , ticks=ticks)

    plt.savefig(filename)
    plt.close(fig)
