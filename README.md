# TransRP

This is the official code of MIDL 2023 paper: 'TransRP: Transformer-based PET/CT feature extraction incorporating clinical data for recurrence-free survival prediction in oropharyngeal cancer'. Its abstract is shown below.

Abstract:
The growing number of subtypes and treatment options for oropharyngeal squamous cell carcinoma (OPSCC), a common type of head and neck cancer (HNC), highlights the need for personalized therapies. Prognostic outcome prediction models can identify different risk groups for investigation of intensified or de-escalated treatment strategies. Convolution neural networks (CNNs) have been shown to have improved predictive performance compared to traditional clinical and radiomics models by extracting comprehensive and representative features. However, CNNs are limited in their ability to learn global features within an entire volume. In this study, we propose a Transformer-based model for predicting recurrence-free survival (RFS) in OPSCC patients, called TransRP. TransRP consists of a CNN encoder to extract rich PET/CT image features, a Transformer encoder to learn global context features, and a fully connected network to incorporate clinical data for RFS prediction. We investigated three different methods for combining clinical features into TransRP. The experiments were conducted using the public HECKTOR 2022 challenge dataset, which includes pretreatment PET/CT scans, Gross Tumor Volume masks, clinical data, and RFS for OPSCC patients. The dataset was split into a test set (n = 120) and a training set (n = 362) for five-fold cross-validation. The results show that TransRP achieved the highest test concordance index of 0.698 (an improvement > 2%) in RFS prediction compared to several state-of-the-art clinical and CNN-based methods. In addition, we found that incorporating clinical features with image features obtained from the Transformer encoder performed better than using the Transformer encoder to extract features from both clinical and image features. 

The TransRP models were developed based on packages:
PyTorch 1.8.0
MONAI 1.0.0
Lifelines 0.27.3


The architecture of TransRP:
![image](https://user-images.githubusercontent.com/86932526/228285207-3acf0560-a547-41dd-89b7-aba1c94bdf2c.png)

A command example of training TransRP models:

python main.py --optimizer sgd  --batch_size 12  --oversample True --input_modality CT PT gtv --model TransRP_DenseNet121_m3 --fold 1 --data_path './Data/images_processed'


A command example of testing TransRP models:

python main.py --optimizer sgd  --batch_size 12  --oversample True --input_modality CT PT gtv --model TransRP_DenseNet121_m3 --fold 1 --data_path './Data/images_processed' --no_train --no_val --resume_id jobID_saved_in_wandb
