from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, OneFormerForUniversalSegmentation, OneFormerImageProcessor, OneFormerProcessor
import os
import torch
import pandas as pd
import numpy as np
import argparse
from PIL import Image
from datetime import datetime
from modules.metrics import calculate_metrics_for_classes, adjust_labels_for_metrics

parser = argparse.ArgumentParser(description='Run k-fold cross-validation for Segformer model (or train the model with the whole dataset, if save_model is 1)')
parser.add_argument('--model_path', default='model_weights.pth', help='Path to the model file, a pth file with the pytorch model weights.')
parser.add_argument('--image_size', type=int, default=512, help='Image size')
parser.add_argument('--content_dir', default='/input', help='Content directory')
parser.add_argument('--test_folder', default='/nikon_dataset_test', help='Dataset folder for running evaluations') # images and annotations folders must be inside this folder

# Export masks or images + masks
parser.add_argument('--export_images', type=int, default=1, help='Export images with the predicted binary masks')
parser.add_argument('--export_human', type=int, default=1, help='Export images with the predicted binary masks overlayed on the original images')
parser.add_argument('--export_metrics', type=int, default=1, help='Export metrics to an Excel file')

model_path = parser.parse_args().model_path
image_size = parser.parse_args().image_size

content_dir = parser.parse_args().content_dir # must come as /folder
test_folder = parser.parse_args().test_folder # must come as /folder

export_images = parser.parse_args().export_images == 1
export_human = parser.parse_args().export_human == 1
export_metrics = parser.parse_args().export_metrics == 1

# Dataset directory
dataset_dir = f'{content_dir}{test_folder}'
results_dir = f'{dataset_dir}/results'

os.makedirs(results_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reescreve acima, primeiro pegando apenas o nome do arquivo se for passado o caminho completo, caso contrário, pega o nome do modelo
model_file = os.path.basename(os.path.normpath(model_path))
if 'oneformer' in model_file.lower():
    model_name = model_file.split('_')[0] + '/' + '_'.join(model_file.split('_')[1:5])
else:
    model_name = '/'.join(model_file.split('_')[:2])
if 'maskformer' in model_name.lower():
    model_ = MaskFormerForInstanceSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True)
    image_processor = MaskFormerImageProcessor(size={"height": image_size, "width": image_size}, ignore_index=255)
elif 'mask2former' in model_name.lower():
    model_ = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True)
    image_processor = Mask2FormerImageProcessor(size={"height": image_size, "width": image_size}, ignore_index=255)
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
else:
    raise ValueError("Model name must contain 'maskformer' or 'mask2former' or 'oneformer'")
model_.load_state_dict(torch.load(f'{content_dir}/{model_path}', map_location=torch.device(device)))

model_.to(device)
model_.eval()

files = [i for i in os.listdir(f'{dataset_dir}/images') if i.endswith('.jpg') or i.endswith('.JPG') or i.endswith('.png')]

metrics_list = []

now = datetime.now() 
date_time = now.strftime("%Y-%m-%d_%H_%M")
date_ = now.strftime("%Y-%m-%d")

class_list = [1] # Background is not relevant. This codebade is adapted for binary segmentation
model_.eval()
model_.model.is_training = False

for file in files:
    print('Processing file:', file)
    image = Image.open(f'{dataset_dir}/images/{file}')
    with torch.no_grad():
        if 'oneformer' in model_name.lower():
            encoded_inputs = image_processor(image, task_inputs=['semantic'], return_tensors="pt")
            outputs = model_(encoded_inputs.pixel_values.to(device), task_inputs=encoded_inputs.task_inputs.to(device))
        else:
            pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
            outputs = model_(pixel_values=pixel_values)
        predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
    png_file = file.replace('JPG', 'png').replace('jpg', 'png')
    if export_metrics:
        ground_truth = np.array(Image.open(f'{dataset_dir}/annotations/{png_file}'))
        adjusted_predicted, adjusted_labels = adjust_labels_for_metrics(predicted_segmentation_map, ground_truth)  # <<<<<<< atenção aqui
        metrics = calculate_metrics_for_classes(adjusted_predicted, adjusted_labels, [1])
        metrics['image'] = file
        metrics_list.append(metrics)
    else:
        ground_truth = np.zeros((predicted_segmentation_map.shape[0], predicted_segmentation_map.shape[1]))
        adjusted_predicted, adjusted_labels = adjust_labels_for_metrics(predicted_segmentation_map, ground_truth)

    if export_images:
        folder = f'{results_dir}/{model_name.replace("/", "_")}_{date_}'
        os.makedirs(folder, exist_ok=True)
        Image.fromarray((adjusted_predicted > 0).astype(np.uint8) * 255).save(f'{folder}/{png_file}')
        if export_human:
            color_seg = np.zeros((adjusted_predicted.shape[0],
                                adjusted_predicted.shape[1], 3), dtype=np.uint8)
            color_seg[adjusted_predicted == 1, :] = [226, 67, 205]
            final = np.array(image) + color_seg * 0.3
            final = final.astype(np.uint8)
            Image.fromarray(final).save(f'{folder}/{file.replace(".jpg", "_human.jpg").replace(".JPG", "_human.jpg").replace(".png", "_human.png").replace(".PNG", "_human.PNG")}', format='JPEG', quality=85)
if export_metrics:
    _metrics = list(metrics[class_list[0]].keys())

    columns = pd.MultiIndex.from_product([class_list, _metrics])
    df = pd.DataFrame(columns=columns)

    # Adicionar os dados ao DataFrame
    for entry in metrics_list:
        image = entry['image']
        metrics = entry[1]
        df.loc[image] = [metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['iou']]
        df.index.name = 'image'
    df.loc['AVG'] = df.mean(numeric_only=True, axis=0)

    if len(class_list) == 1: # if there is only one class, drop the first level of the columns
        df = df.droplevel(0, axis=1)

    df = df.reset_index()
    df.to_excel(f'{results_dir}/test_{model_name.replace("/", "_")}_{date_time}.xlsx', index=False)
    print('Execution completed, metrics saved to Excel file.')
else:
    print('Execution completed.')