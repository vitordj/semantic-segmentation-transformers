from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, OneFormerForUniversalSegmentation, OneFormerImageProcessor, OneFormerProcessor
from modules.segmentation_dataset import MaskFormerSegmentationDataset, OneFormerSegmentationDataset
from modules.train import train_and_validate_maskformer
import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from datetime import datetime
import argparse 
from copy import deepcopy

parser = argparse.ArgumentParser(description='Run k-fold cross-validation for MaskFormer or Mask2Former model (or train the model with the whole dataset, if save_model is 1)')
parser.add_argument('--model_name', default='facebook/maskformer-swin-base-ade', help='Model name')
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

# vamos criar alguns parsers para os argumentos de entrada para configurar os weights da loss
parser.add_argument('--no_object_weight', type=float, default=0.1, help='Weight for no object in the classification loss')
parser.add_argument('--dice_weight', type=float, default=1.0, help='Weight for the dice loss in the mask loss')
parser.add_argument('--cross_entropy_weight', type=float, default=1.0, help='Weight for the cross entropy loss in the classification loss')
parser.add_argument('--mask_weight', type=float, default=20.0, help='Weight for the focal loss in the mask loss')
parser.add_argument('--class_weight', type=float, default=2.0, help='Weight for the classification loss')

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

# Dataset directory
dataset_dir = f'{content_dir}{dataset_folder}'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if 'maskformer' in model_name.lower():
    model_ = MaskFormerForInstanceSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True)
    image_processor = MaskFormerImageProcessor(size={"height": image_size, "width": image_size}, ignore_index=255)
    _name = 'maskformer'
elif 'mask2former' in model_name.lower():
    model_ = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True)
    image_processor = Mask2FormerImageProcessor(size={"height": image_size, "width": image_size}, ignore_index=255)
    _name = 'mask2former'
elif 'oneformer' in model_name.lower():
    import json
    from transformers import AutoTokenizer
    
    model_ = OneFormerForUniversalSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True)
    json_obj = {'0':{'isthing':0, 'name':'bg'},
                '1':{'isthing':0, 'name':'bunch'}}

    with open(f"{content_dir}/class_info.json", "w") as f:
        json.dump(json_obj, f, indent=4)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    image_processor = OneFormerImageProcessor(size={"height": image_size, "width": image_size}, class_info_file = f'{content_dir}/class_info.json', repo_path='', ignore_index=255)
    image_processor = OneFormerProcessor(image_processor, tokenizer)
    image_processor.image_processor.num_text = model_.config.num_queries - model_.config.text_encoder_n_ctx
    model_.config.contrastive_temperature = None
    _name = 'oneformer'
else:
    raise ValueError("Model name must contain 'maskformer' or 'mask2former' or 'oneformer'")

# KFold cross-validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
dfs = []

for fold, (train_index, test_index) in enumerate(kf.split(np.arange(len(os.listdir(f'{dataset_dir}/images'))))):
    print(f"Fold {fold+1}")

    if _name == 'maskformer' or _name == 'mask2former':
        train_dataset = MaskFormerSegmentationDataset(dataset_dir, image_processor, indices=train_index)
        valid_dataset = MaskFormerSegmentationDataset(dataset_dir, image_processor, indices=test_index)
    elif _name == 'oneformer':
        train_dataset = OneFormerSegmentationDataset(dataset_dir, image_processor, indices=train_index)
        valid_dataset = OneFormerSegmentationDataset(dataset_dir, image_processor, indices=test_index)

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    # Reset the model, optimizer, criterion, etc.
    model = deepcopy(model_)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    df, best_model = train_and_validate_maskformer(train_dataloader, valid_dataloader, model
                                      , optimizer, image_processor, device, epochs
                                      , scheduler_, early_stopping_patience, save_model)
    df['fold'] = fold + 1
    dfs.append(df)

    # if it is to actually train the final model, runs only one fold
    if save_model:
        break

now = datetime.now() 
date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

df_all = pd.concat(dfs, ignore_index=True)
df_all.to_excel(f'{content_dir}/output/{_name}/{model_name.replace("/", ":")}_{date_time}_metrics.xlsx', index=False)

if best_model:
    torch.save(best_model, f'{content_dir}/output/{_name}/{model_name.replace("/", ":")}_{date_time}_best_model.pth')