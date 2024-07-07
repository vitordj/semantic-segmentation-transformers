# Purpose

This repository provides a framework for **fine-tuning semantic segmentation** transformer models specifically for **binary classification tasks**. It is designed to be flexible and easy to use, allowing you to customize various parameters such as the model name, batch size, number of epochs, and more through command-line args.

This makes it a powerful tool for experimenting with different configurations and optimizing your model's performance.

This repo is based on Hugging Face's Transformers lib, it is adapted for SegFormer, MaskFormer, Mask2Former and OneFormer.
It was strongly inspired by the Jupyter Notebooks made by @NielsRogge.

One of the main contribuitions of this repo is the implementation of a Focal Loss on SegFormer.
Althought not described in its paper, SegFormer works with a Cross-Entropy Loss, that can be harmful when working with unbalanced datasets.

IF THIS REPO HELP YOU WITH YOUR PROJECT, GIVE IT A STAR. :sunglasses:

# Default Folder Values and Build Configuration

The default folder values and the build configuration are set considering the requirements of the Information Technology Sector of UFSC (SeTIC) and to project's classes: background and (grape) bunch.

Please ensure that your setup meets these requirements before proceeding with the model training. If you need to customize these settings, you can do so by using command-line arguments as described in the 'Command-Line Arguments' section.

If you only need to train your model, you will need to use a `..._kfold.py` file with the `--save_model` arg set to 1.
Based on the `k_folds` arg 


# Command-Line Arguments (fine-tuning)

The project uses several command-line arguments to configure the model training. Here are the command-line arguments and their default values:

- `--model_name`: The name of the model to be used. The default value is "nvidia/mit-b0".
- `--batch_size`: The batch size in training. The default value is "4".
- `--epochs`: The number of epochs to be performed in training. The default value is "2".
- `--k_folds`: The number of folds if using K-fold cross-validation. The default value is "7".
- `--image_size`: The image size for training. The default value is "512".
- `--learning_rate`: The learning rate for training. The default value is "0.00006".
- `--scheduler`: Implemented for OneCycleLR and PolynomialLR. Accepts `onecycle` and `poly`, defaults to `None`.
- `--lr_multiplier`: If scheduler set to `onecycle`, default=10, it sets max_lr as learning_rate * lr_multiplier. If scheduler set to `poly`, default=1, it sets the power of polynomial function.
- `--total_iters_poly`: If scheduler set to `poly` it sets the total iterations. Defaults to number of epochs.  
- `--early_stopping_patience`: The default value is "10".
- `--content_dir`: The content directory. The default value is "/vitordj". 
- `--dataset_folder`: The dataset folder. The default value is "/nikon_dataset". 
- `--save_model`: Save model (0 for no, 1 for yes). The default value is "0".

SegFormer additionally have args for its FocalLoss implementation.

## How to Use Command-Line Arguments
When running the script, you can set the command-line arguments directly in the command line.

Here is an example of how you can do this:

```shell
python segformer_kfold.py --model_name 'nvidia/mit-b1' --batch_size 8 --epochs 4 --k_folds 5 --image_size 256 --learning_rate 0.00003 --scheduler 'onecycle' --lr_multiplier 5 --early_stopping_patience 5 --content_dir '/new_input' --dataset_folder '/new_dataset' --save_model 1
```


By default assumes the following folder structure in your container:
```shell
────bin
────home
.
.
.
────input (--content_dir arg)
    ├───your_dataset (--dataset_folder arg)
    │   ├───annotations
    │   └───images
    └───output
        ├───segformer
        ├───maskformer
        ├───mask2former
        └───oneformer
```
## Important

As MaskFormer API on Hugging Face is almost identical to Mask2Former's API, both can be executed from maskformer_kfold.
Its difference from segformer_kfold is in it Dataset class and not applying the FocalLoss.

In an intermediate step, it is checked if the model string contains 'maskformer' or 'mask2former'.


## Command-Line Arguments (evaluation)

This project also counts with evaluation scripts, with the suffix `_eval`.
To run the evaluation scripts, the following command-line arguments can be used: 
- `--model_path`: Path to the model file, a `.pth` file with the PyTorch model weights. Default is `model_weights.pth`. 
- `--image_size`: Image size. Default is `512`. - `--content_dir`: Content directory. Default is `/input`. 
- `--test_folder`: Dataset folder for running evaluations. Default is `/nikon_dataset_test`. Images and annotations folders must be inside this folder. 
- `--export_images`: Export images with the predicted binary masks. Set to `1` to enable, `0` to disable. Default is `1`. 
- `--export_human`: Export images with the predicted binary masks overlayed on the original images. Set to `1` to enable, `0` to disable. Default is `1`. 
- `--export_metrics`: Export metrics to an Excel file. Set to `1` to enable, `0` to disable. Default is `1`.

As an example, you can set --export_images and --export_human to zero if you want only the Precision, Recall, F1-Score and IoU of your model on the test size.
If you just need an PNG file with your binary segmentation and there's even no ground truth masks for your eval dataset, just set --export_human and --export_metrics to 0.

All exports are set to 1 by default.

## TODO
- Make Segformer convert 255 labels to 1 in its __getitem__ instead of using a function to this. Also need to update id2label & label2id dicts and class_list (is it really necessary?)
- Make Segformer and MaskFormer's Dataset class identical?
- Include condition on `segmentation_dataset.py` to work with val and train folders. 