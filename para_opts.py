# -*- coding: utf-8 -*-
'''
Created on Fri Jun 17 16:05:46 2022
@author: MaB
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--result_path',
        default='./result/',
        type=str,
        help='Result directory path')
    
    parser.add_argument(
        '--data_path',
        default='./data/Imaging_data',
        type=str,
        help='Data directory path (after resampled)')

    parser.add_argument(
        '--endpoint_path',
        default='./Data/hecktor2022_clinical_data_training_processed.csv',
        type=str,
        help='Endpoint information path')

    parser.add_argument(
        '--learning_rate',
        default=2e-4,
        type=float,
        help=
        'Initial learning rate ')

    parser.add_argument('--momentum', default=0.90, type=float, help='Momentum')

    parser.add_argument(
        '--weight_decay', default=2e-4, type=float, help='Weight Decay')

    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently only adam, sgd')

    parser.add_argument(
        '--batch_size', 
        default= 12, 
        type=int, 
        help='Batch Size')

    parser.add_argument(
        '--n_epochs',
        default= 400,
        type=int,
        help='Number of total epochs to run')

    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')

    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training, for example: save_100.pth')

    parser.add_argument(
        '--pretrain_path', default='', type=str, help='Pretrained model (.pth)')

    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)

    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)

    parser.add_argument(
        '--no_test',
        action='store_true',
        help='If true, test is not performed.')
    parser.set_defaults(no_test=False)

    parser.add_argument(
        '--checkpoint_path',
        default= 'current_model.pth',
        type=str,
        help='Trained model is saved at this name')

    parser.add_argument(
        '--model',
        default='Densenet121',
        type=str,
        help='(resnet/ denset and all nets from MONAI )')

    parser.add_argument(
        '--model_actfn',
        default='relu',
        type=str,
        help='activation function')

    parser.add_argument(
        '--input_modality',
        nargs='+',
        help='Different types of input modality selection: CT, PT, gtv')

    parser.add_argument(
        '--fold',
        default=1,
        type=int,
        help='fold number')

    parser.add_argument(
        '--esn',
        default= 25,
        type=int,
        help='early stopping patience number')

    parser.add_argument(
        '--resume_id',
        default='',
        type=str,
        help='The id of wandb runing for resume training')

    parser.add_argument(
        '--oversample',
        type= bool,
        default= False , 
        help='If true, oversample is performed.')

    parser.add_argument(
        '--es_metric',
        default='cindex',
        type=str,
        help='The metric for early stopping in validation set')   

    args = parser.parse_args()

    return args
