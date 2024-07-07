from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch

# TODO - Implement the image processor for Mask2Former on __getitem__ method
# TODO - Create base class for segmentation datasets (their both share the same __init__ and __len__ methods)

class SegformerSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""
    def __init__(self, root_dir, image_processor, transform=None, indices=None, top_cut=0, bottom_cut=0):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.transform = transform
        self.indices = indices

        self.img_dir = os.path.join(self.root_dir, "images")
        self.ann_dir = os.path.join(self.root_dir, "annotations")
        image_file_names = [f for f in sorted(os.listdir(self.img_dir)) if os.path.isfile(os.path.join(self.img_dir, f))]
        annotation_file_names = [f for f in sorted(os.listdir(self.ann_dir)) if os.path.isfile(os.path.join(self.ann_dir, f))]

        if indices is not None:
            self.images = [image_file_names[i] for i in indices]
            self.annotations = [annotation_file_names[i] for i in indices]
        else:
            self.images = image_file_names
            self.annotations = annotation_file_names

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        if self.transform:
            image = self.transform(image)

        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.squeeze()

        return encoded_inputs

class MaskFormerSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset for MaskFormer and Mask2Former."""
    def __init__(self, root_dir, image_processor, transform=None, indices=None):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.transform = transform
        self.indices = indices

        self.img_dir = os.path.join(self.root_dir, "images")
        self.ann_dir = os.path.join(self.root_dir, "annotations")
        image_file_names = [f for f in sorted(os.listdir(self.img_dir)) if os.path.isfile(os.path.join(self.img_dir, f))]
        annotation_file_names = [f for f in sorted(os.listdir(self.ann_dir)) if os.path.isfile(os.path.join(self.ann_dir, f))]

        if indices is not None:
            self.images = [image_file_names[i] for i in indices]
            self.annotations = [annotation_file_names[i] for i in indices]
        else:
            self.images = image_file_names
            self.annotations = annotation_file_names

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")
        image = np.array(image)

        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))
        segmentation_map = np.array(segmentation_map)
        segmentation_map = np.where(segmentation_map == 255, 1, segmentation_map).astype(np.int32)

        if self.transform:
            transformed = self.transform(image=image, mask=segmentation_map)
            image, segmentation_map = transformed['image'], transformed['mask']

        encoded_inputs = self.image_processor(image,
                                              segmentation_maps=segmentation_map,
                                              return_tensors="pt",
                                          )
        for k,v in encoded_inputs.items():
          if isinstance(v, torch.Tensor):
            encoded_inputs[k].squeeze_() # remove batch dimension
          elif isinstance(v, list):
            encoded_inputs[k] = encoded_inputs[k][0]

        return encoded_inputs
    
class OneFormerSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset for MaskFormer and Mask2Former."""
    def __init__(self, root_dir, image_processor, transform=None, indices=None):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.transform = transform
        self.indices = indices

        self.img_dir = os.path.join(self.root_dir, "images")
        self.ann_dir = os.path.join(self.root_dir, "annotations")
        image_file_names = [f for f in sorted(os.listdir(self.img_dir)) if os.path.isfile(os.path.join(self.img_dir, f))]
        annotation_file_names = [f for f in sorted(os.listdir(self.ann_dir)) if os.path.isfile(os.path.join(self.ann_dir, f))]

        if indices is not None:
            self.images = [image_file_names[i] for i in indices]
            self.annotations = [annotation_file_names[i] for i in indices]
        else:
            self.images = image_file_names
            self.annotations = annotation_file_names

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        image = np.array(image)

        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))
        segmentation_map = np.array(segmentation_map)
        segmentation_map = np.where(segmentation_map == 255, 1, segmentation_map).astype(np.int32)

        if self.transform:
            transformed = self.transform(image=image, mask=segmentation_map)
            image, segmentation_map = transformed['image'], transformed['mask']
        encoded_inputs = self.image_processor(image,
                                              segmentation_maps=segmentation_map,
                                              task_inputs=['semantic'],
                                              return_tensors="pt",
                                          )
        for k,v in encoded_inputs.items():
            if isinstance(v, torch.Tensor):
                encoded_inputs[k].squeeze_() # remove batch dimension
            elif isinstance(v, list):
                encoded_inputs[k] = encoded_inputs[k][0]

        return encoded_inputs