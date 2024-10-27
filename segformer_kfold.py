from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from modules.segmentation_dataset import SegformerSegmentationDataset
from modules.train import train_and_validate_segformer
from modules.losses import FocalLoss
import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from datetime import datetime
import argparse
from copy import deepcopy

parser = argparse.ArgumentParser(description='Run k-fold cross-validation for Segformer model (or train the model with the whole dataset, if save_model is 1)')
parser.add_argument('--model_name', default='nvidia/mit-b0', help='Model name')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
parser.add_argument('--k_folds', type=int, default=7, help='Number of K folds')
parser.add_argument('--image_size', type=int, default=512, help='Image size')
parser.add_argument('--learning_rate', type=float, default=0.00006, help='Learning rate')
parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--content_dir', default='/input', help='Content directory')
parser.add_argument('--dataset_folder', default='/nikon_dataset', help='Dataset folder')
parser.add_argument('--save_model', type=int, default=0, help='Save model (0 or 1)')

parser.add_argument('--scheduler', default=None, help='Scheduler ("onecycle" or "poly" or None)')
parser.add_argument('--lr_multiplier', type=float, default=None, help='Learning Rate multiplier for OneCycleLR or power for PolynomialLR')
parser.add_argument('--total_iters_poly', type=int, default=None, help='Total iters for poly scheduler')

parser.add_argument('--apply_focal', type=int, default=1, help='Enable Focal Loss (instead of the usual CrossEntropy)')
parser.add_argument('--fl_alpha', type=float, default=0.25, help='Focal Loss alpha')
parser.add_argument('--fl_gama', type=float, default=2.0, help='Focal Loss gama')

model_name = parser.parse_args().model_name

batch_size = parser.parse_args().batch_size
epochs = parser.parse_args().epochs
k_folds = parser.parse_args().k_folds # Number of folds if using K-fold cross-validation or the proportion (1/K) of the dataset tha must be used if actually training the model
image_size = parser.parse_args().image_size

learning_rate = parser.parse_args().learning_rate
scheduler = parser.parse_args().scheduler
lr_multiplier = parser.parse_args().lr_multiplier # Only useful if scheduler is 1
total_iters_poly = parser.parse_args().total_iters_poly # Only useful if scheduler is poly
early_stopping_patience = parser.parse_args().early_stopping_patience

content_dir = parser.parse_args().content_dir # must come as /folder
dataset_folder = parser.parse_args().dataset_folder # must come as /folder
save_model = parser.parse_args().save_model == 1 # Useful for when

apply_focal = parser.parse_args().apply_focal == 1
fl_alpha = parser.parse_args().fl_alpha
fl_gama = parser.parse_args().fl_gama

# Dataset directory
dataset_dir = f'{content_dir}{dataset_folder}'

image_processor = SegformerImageProcessor(size={"height": image_size, "width": image_size})

# Class labels
id2label = {0: 'bg', 255: 'bunch'} # this repo is adapted for binary segmentation and assumes the class 255 (white) is the class of interest
label2id = {'bg': 0, 'bunch': 255}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=len(id2label.keys()), id2label=id2label, label2id=label2id)

# KFold cross-validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
dfs = []

now = datetime.now() 
date_time = now.strftime("%Y-%m-%d_%H:%M")

for fold, (train_index, test_index) in enumerate(kf.split(np.arange(len(os.listdir(f'{dataset_dir}/images'))))):
    print(f"Fold {fold+1}")

    # Create instances of the dataset for training and validation using the indices
    train_dataset = SegformerSegmentationDataset(dataset_dir, image_processor, indices=train_index)
    valid_dataset = SegformerSegmentationDataset(dataset_dir, image_processor, indices=test_index)

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    # Reset the model, optimizer, criterion, etc.
    model = deepcopy(model_)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if apply_focal:
        criterion = FocalLoss(alpha=fl_alpha, gamma=fl_gama)
    else:
        criterion = None

    if scheduler == 'onecycle':
        # Sets default value for OneCycleLR
        lr_multiplier = lr_multiplier if lr_multiplier else 10
        scheduler_ = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate * lr_multiplier
                                                        , steps_per_epoch=len(train_dataloader), epochs=epochs)
    elif scheduler == 'poly':
        # Sets default values for PolynomialLR
        lr_multiplier = lr_multiplier if lr_multiplier else 1
        total_iters_poly = total_iters_poly if total_iters_poly else epochs
        scheduler_ = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters_poly, power=lr_multiplier)
    else:
        scheduler_ = None

    # Calls the training function. best_model is only saved if save_model is True
    df, best_model = train_and_validate_segformer(train_dataloader, valid_dataloader, model
                                      , optimizer, criterion, device, epochs
                                      , scheduler_, early_stopping_patience, save_model)
    df['fold'] = fold + 1
    dfs.append(df)

    if save_model:
        torch.save(best_model, f'{content_dir}/output/segformer/{model_name.replace("/", ":")}_{date_time}_best_model_fold{fold+1:02d}.pth')

total_time = datetime.now() - now
total_seconds = total_time.total_seconds()
minutes = total_seconds // 60
seconds = total_seconds % 60

print(f'Total iteration time: {int(minutes)} minutes and {int(seconds)} seconds.')

df_all = pd.concat(dfs, ignore_index=True)
df_all.to_excel(f'{content_dir}/output/segformer/{model_name.replace("/", ":")}_{date_time}_metrics.xlsx', index=False)