import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import timm
import yaml
from typing import List


class SwinModelMultiTask(nn.Module):
    def __init__(self, dataset_info: dict, pretrained=False):
        super(SwinModelMultiTask, self).__init__()

        if not isinstance(dataset_info, dict):
            raise TypeError(f"Expected parameter to be a dict, but got {type(dataset_info).__name__}")
        
        self.dataset_info = dataset_info
        self.dataset_labels = list(self.dataset_info.keys())
        self.model = timm.create_model('swin_large_patch4_window7_224',  # swin_large_patch4_window12_384
                                       pretrained=pretrained, 
                                       num_classes=2).to("cpu") # num_classes is not important


        self.num_features = self.model.head.fc.in_features

        # Separate components of the image model head
        self.global_pool = self.model.head.global_pool
        self.dropout = self.model.head.drop

        # Removing the original head of the image model
        self.model.head = nn.Identity()
        self.fc = nn.ModuleDict()
        
        for dataset_name, dataset_args in self.dataset_info.items():
            fc = nn.Linear(self.num_features, dataset_args['num_classes'])
            self.fc[dataset_name] = fc
            

        for param in self.model.parameters():
            param.requires_grad = True

    def get_num_features(self):
        return self.num_features

    # Helper function to extract model features
    def extract_features(self, images):
        image_features = self.model(images) # torch.Size([32, 7, 7, 1536])
        image_features = self.global_pool(image_features) # torch.Size([32, 1536])
        image_features = self.dropout(image_features)
        return image_features

    def forward_classification_tasks(self, images):
        image_features = self.extract_features(images)
        outputs = {}
        for dataset_name, info in self.dataset_info.items():
            logits = self.fc[dataset_name](image_features)
            probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()

            outputs[dataset_name] = {
                'dataset': dataset_name,
                'predictions': probabilities[0],  
                'num_classes': info['num_classes']
            }
        return outputs
    def forward_explanation_tasks(self, images):
        image_features = self.extract_features(images)
        outputs = {}
        for dataset_name, info in self.dataset_info.items():
            logits = self.fc[dataset_name](image_features)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            outputs[dataset_name] = {
                'dataset': dataset_name,
                'predictions': probabilities,  
                'num_classes': info['num_classes']
            }
        return outputs

    
    def forward(self, images, data_label):
        if self.dataset_info[data_label]['type'] == "segmentation":
            image_features = self.model(images) # features before global_pool
        elif self.dataset_info[data_label]['type'] == "classification":
            image_features = self.extract_features(images)
        output = self.fc[data_label](image_features)
        return output