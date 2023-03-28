# -*- coding: utf-8 -*-
"""
Created on Monday March 27 15:50:49 2023
The main.py of TransRP
@author: MaB
"""
import logging
import os
import sys
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from utlis import  get_model, plot_images , get_data_dict_withclc_hecktor, seed_torch, get_oversampler
import torch.nn as nn
from para_opts import parse_opts
from torch.optim.lr_scheduler import MultiStepLR
import wandb
from test import test_hecktor

import losses
import random
import monai

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset, Dataset, DistributedWeightedRandomSampler
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    Compose,
    RandRotate90d,
    Resized,
    ScaleIntensityd,
    EnsureTyped,
    EnsureType,
    RandFlipd,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandRotated,
    RandZoomd,
    RandAffined,
    Rand3DElasticd,
    OneOf,
    CenterSpatialCropd
)

seed_torch(42)    

def main():
    
    # load settings
    opt = parse_opts()
    
    # observe in wandb
    if opt.resume_id != '': 
          wandb.init(project='TransRP', id = opt.resume_id, resume = 'must', entity='mbq1137723824')
    else: 
          wandb.init(project='TransRP', entity='mbq1137723824')
            
    # set endpoint 
    opt.event_name = 'Relapse'
    opt.event_time_name = 'RFS'
    
    # set device
    pin_memory = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set data directory
    opt.result_path = opt.result_path + str(opt.model) + '_input_' + str(opt.input_modality) + '_OS_' + str(opt.oversample) + '_fold' + str(opt.fold)
    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)       
        
    # save updated settings to wandb
    wandb.config.update(opt, allow_val_change=True)    
    
    # load patients ID and endpoint data
    endpoint_info = pd.read_csv(opt.endpoint_path) 

    # get patient IDs of different sets
    test_ID = list(endpoint_info.loc[endpoint_info['CVgroup_'+ opt.event_name] == 'test']['PatientID'])
    val_ID = list(endpoint_info.loc[endpoint_info['CVgroup_'+ opt.event_name] == 'CV' + str(opt.fold) ]['PatientID'])
    train_ID  = list(endpoint_info[~endpoint_info['CVgroup_'+ opt.event_name].isin(['test', 'CV' + str(opt.fold)])]['PatientID'])
    
    # generate data_dictionary
    train_data_dict = get_data_dict_withclc_hecktor(train_ID, opt, endpoint_info.copy()) 
    val_data_dict = get_data_dict_withclc_hecktor(val_ID, opt ,endpoint_info.copy())  
    test_data_dict = get_data_dict_withclc_hecktor(test_ID, opt, endpoint_info.copy())  

    # Define image transforms
    train_transforms = Compose([LoadImaged(keys = [image for image in opt.input_modality]),
                                AddChanneld(keys = [image for image in opt.input_modality]) , 
                                Resized(keys = ['CT','PT','gtv'], spatial_size = (96, 96 ,96),  mode = ('trilinear', 'trilinear','nearest'), align_corners= (True,True,None),  allow_missing_keys = True) , 
                                ScaleIntensityRanged( keys = 'CT', a_min = -200 , a_max = 200, b_min=0, b_max=1, clip= True, allow_missing_keys = True ),
                                ScaleIntensityRanged( keys = 'PT', a_min = 0 , a_max = 25, b_min=0, b_max=1, clip= True , allow_missing_keys = True),
                                RandFlipd(keys=[image for image in opt.input_modality], prob=0.5, spatial_axis=0),
                                RandFlipd(keys=[image for image in opt.input_modality], prob=0.5, spatial_axis=1),
                                RandFlipd(keys=[image for image in opt.input_modality], prob=0.5, spatial_axis=2), 
                                
                                # shape transforms
                                RandAffined(keys=['CT','PT','gtv'],
                                            prob= 0.5 ,
                                            translate_range=(7, 7, 7),  
                                            rotate_range=(np.pi / 24, np.pi / 24, np.pi / 24),
                                            scale_range=(0.07, 0.07, 0.07), padding_mode='border', mode = ('bilinear', 'bilinear','nearest'), allow_missing_keys = True), 
                                
                                Rand3DElasticd( keys=['CT','PT','gtv'],
                                                prob=0.2,
                                                sigma_range=(5, 8),
                                                magnitude_range=(100, 200),
                                                translate_range=(7, 7, 7),
                                                rotate_range=(np.pi / 24, np.pi / 24, np.pi / 24),
                                                scale_range=(0.07, 0.07, 0.07),
                                                padding_mode='border', mode = ('bilinear', 'bilinear','nearest'), allow_missing_keys = True),
                                # add noise
                                #RandGaussianNoised(keys =['ct','pt'], prob=0.2, mean=0.0, std=0.1, allow_missing_keys = True ),                     
                               ])
 
    print ([image for image in opt.input_modality])
    val_transforms = Compose([LoadImaged(keys = [image for image in opt.input_modality]),
                                AddChanneld(keys = [image for image in opt.input_modality]) , 
                                Resized(keys = ['CT','PT','gtv'], spatial_size = (96, 96 ,96),  mode = ('trilinear', 'trilinear','nearest'), align_corners= (True,True,None),  allow_missing_keys = True) , 
                                ScaleIntensityRanged( keys = 'CT', a_min = -200 , a_max = 200, b_min=0, b_max=1, clip= True, allow_missing_keys = True),
                                ScaleIntensityRanged( keys = 'PT', a_min = 0 , a_max = 25, b_min=0, b_max=1, clip= True, allow_missing_keys = True),                                              
                                ])
    
    # Define nifti dataset, data loader
    num_workers = 10
    train_ds = Dataset(data=train_data_dict, transform=train_transforms)
    
    # Oversampling the traning set
    if opt.oversample:
        sampler  =  get_oversampler(endpoint_info, train_ID, opt.event_name)
        train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=num_workers, pin_memory=pin_memory, sampler = sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    
    # training set used for test at every training epoch
    train_ds_test = Dataset(data=train_data_dict, transform=val_transforms)
    train_loader_test = DataLoader(train_ds_test, batch_size=opt.batch_size, num_workers=num_workers, pin_memory=pin_memory)
    
    # validation set
    val_ds = Dataset(data=val_data_dict, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, num_workers=num_workers, pin_memory=pin_memory)
    
    # test set
    test_ds = Dataset(data=test_data_dict, transform= val_transforms)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=num_workers, pin_memory=pin_memory) 
    
    # Create model
    model = get_model(opt).to(device)
    
    # loss function
    criterion = losses.NegativeLogLikelihood()
    
    # optimizer selection
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.learning_rate, weight_decay = opt.weight_decay)
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay = opt.weight_decay)
    scheduler  = MultiStepLR(optimizer , milestones = [200, 300], gamma = 0.2)
    
    # start a typical PyTorch training
    val_interval = 1
    best_metric = - 100000
    best_loss =  100000    
    best_cindex = - 1 
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    max_epochs = opt.n_epochs
    patience_number = opt.esn # for early stopping
    
    # train epoch
    if not opt.no_train:
        
        if wandb.run.resumed:  # reume a training 
            print (opt.result_path + '/' + opt.checkpoint_path)
            try:
                model_restore = wandb.restore(opt.checkpoint_path)
                checkpoint = torch.load(model_restore.name) 
            except:
                checkpoint = torch.load(opt.result_path + '/' + opt.checkpoint_path)    
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            opt.begin_epoch = checkpoint['epoch'] + 1   # traning straing from the next epoch of saved epoch
            patience_number = checkpoint['patience_number']
            best_metric = checkpoint['best_metric']
            best_metric_epoch =   checkpoint['best_metric_epoch']
            best_loss =  checkpoint['best_loss'], 
            best_cindex =  checkpoint['best_cindex']  
            best_loss =  np.array(best_loss)[0][0]
            #print ('best_loss ,best_cindex', best_loss ,best_cindex)

        for epoch in range(opt.begin_epoch, max_epochs):
            print('-' * 10)
            print(f'epoch {epoch}/{max_epochs}')
            model.train()
            epoch_loss = 0
            step = 0           
            for batch_data in train_loader:
                step += 1
                inputs = torch.Tensor().to(device)
                
                for image in opt.input_modality:
                    sub_data = batch_data[image].to(device)   
                    inputs = torch.cat((inputs,sub_data), 1)
                optimizer.zero_grad()
                if '_m' in opt.model: # clinical
                  outputs = model(inputs,batch_data['clinical'].to(device) )  
                else:
                  outputs = model(inputs)  
                
                loss,neglog_loss,l2_norm = criterion(outputs, batch_data[opt.event_time_name],batch_data[opt.event_name],  model)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                print(f'{step}/{epoch_len}, train_loss: {loss.item():.4f}')
                #writer.add_scalar('train_loss', loss.item(), epoch_len * epoch + step)
                wandb.log({'train_loss': loss.item()})
            
            # evaluate on the training set every epoch
            test_hecktor(model, train_loader_test, device, opt, endpoint_info, train_ID, mode ='train')  
            
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f'epoch {epoch} average loss: {epoch_loss:.4f}')
            wandb.log({'train_epoch_loss': epoch_loss})
            scheduler.step()
            
            # validation every epoch
            if not opt.no_val:
                if epoch % val_interval == 0:
                    
                    model.eval()
                    val_outputs_cat = torch.Tensor().to(device)
                    val_epoch_loss = 0
                    val_step = 0
                    
                    for val_data in val_loader:
                        
                        val_step += 1
                        val_images = torch.Tensor().to(device)
                        
                        for image in opt.input_modality:
                            sub_data = val_data[image].to(device)
                            val_images = torch.cat((val_images,sub_data), 1)

                        with torch.no_grad():
                            if '_m' in opt.model:
                              val_outputs =model(val_images, val_data['clinical'].to(device))
                            else: 
                              val_outputs =model(val_images)

                            val_outputs_cat = torch.cat((val_outputs_cat,val_outputs), 0)
                            val_loss,val_neglog_loss,val_l2_norm = criterion(val_outputs,val_data[opt.event_time_name],val_data[opt.event_name],  model)    
                            val_epoch_loss += val_neglog_loss.item()

                    val_epoch_loss /= val_step                   
                    wandb.log({'val_epoch_loss': val_epoch_loss})
                    val_outputs_cat = val_outputs_cat.detach().cpu().numpy()
                    
                    # label
                    epoch_time =  list(endpoint_info.loc[endpoint_info['PatientID'].isin(val_ID)][opt.event_time_name])
                    epoch_event = list(endpoint_info.loc[endpoint_info['PatientID'].isin(val_ID)][opt.event_name])
                   
                    # c-index calculation
                    metric = concordance_index(epoch_time, - val_outputs_cat[:, 0], epoch_event) # val_cindex
                    metric_values.append(metric)
                    
                    torch.save({ # Save our current model, optimizer status to local disk
                                     'epoch': epoch,
                                      'model_state_dict': model.state_dict(),
                                      'optimizer_state_dict': optimizer.state_dict(),
                                      'patience_number' : patience_number,
                                      'best_metric': best_metric,
                                      'best_metric_epoch' : best_metric_epoch,
                                      'best_loss' : best_loss,
                                      'best_cindex' : best_cindex
                                      }, opt.result_path + '/' + opt.checkpoint_path)
                    
                    torch.save({ # Save our current model, optimizer status to wandb cloud
                                     'epoch': epoch,
                                      'model_state_dict': model.state_dict(),
                                      'optimizer_state_dict': optimizer.state_dict(),
                                      'patience_number' : patience_number,
                                      'best_metric': best_metric,
                                      'best_metric_epoch' : best_metric_epoch,
                                      'best_loss' : best_loss,
                                      'best_cindex' : best_cindex
                                      }, os.path.join(wandb.run.dir, opt.checkpoint_path))
                    
                    # early stopping metric
                    es_metric = metric # C-index as the early stopping metric
                        
                    if epoch > 3:                    
                        if metric > best_cindex and val_epoch_loss < best_loss + 1.5:
                            best_loss = val_epoch_loss
                            best_cindex = metric
                            best_metric = es_metric
                            best_metric_epoch = epoch
                            torch.save(model.state_dict(), opt.result_path + '/best_metric_model.pth')
                            
                            print('saved new best metric model !')
                            wandb.run.summary['best_val-cindex'] = metric
                            wandb.run.summary['best_val-loss'] = val_epoch_loss
                            wandb.run.summary['best_epoch'] = best_metric_epoch
                            patience_number = opt.esn
                            
                        else:
                            patience_number -= 1
            
                    print(f'Current epoch: {epoch} current val_c-index: {metric:.4f} current val_loss: {val_epoch_loss:.4f} ')
                    print(f'Best val_loss: {best_metric:.4f} at epoch {best_metric_epoch}')
                    wandb.log({'val_c-index': metric, 'epoch' : epoch})       
                    
                    # test at every epoch 
                    test_hecktor(model, test_loader, device, opt, endpoint_info, test_ID, mode = 'test')
                    
            if patience_number  <  0:
                print ('Early stopping at epoch' + str(epoch))
                break
                
        print(f'Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}')
    
    # final test when training stop    
    if not opt.no_test:
        
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_workers, pin_memory=pin_memory)
        
        if wandb.run.resumed:
            # for read the history wandb saved values
            api = wandb.Api()
            run = api.run('mbq1137723824/Hecktor_outpred_DeepSurv/' + opt.resume_id)
            print (run.summary['Final test_c-index']) 
            history_test_cindex = run.history(keys=['Final test_c-index']).loc[0, 'Final test_c-index']
            print (history_test_cindex)
            
        model.load_state_dict(torch.load(opt.result_path + '/best_metric_model.pth'))

        model.eval()

        test_outputs_cat = torch.Tensor().to(device)
        with torch.no_grad():
            
            for test_data in test_loader:
    
                test_images = torch.Tensor().to(device)

                for image in opt.input_modality:
                    sub_data = test_data[image].to(device)
                    test_images = torch.cat((test_images,sub_data), 1)
                    
                with torch.no_grad():
                    
                    if '_m' in opt.model:
                      test_outputs =model(test_images, test_data['clinical'].to(device))
                    else:
                      test_outputs =model(test_images)

                    test_outputs_cat = torch.cat((test_outputs_cat,test_outputs), 0)
            
        
            test_outputs_cat = test_outputs_cat.detach().cpu().numpy()
     
            epoch_time =  list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_time_name])
            epoch_event = list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_name])
            
            metric = concordance_index(epoch_time, - test_outputs_cat[:, 0], epoch_event) # test_cindex          
            print (metric)

if __name__ == '__main__':
    main()

