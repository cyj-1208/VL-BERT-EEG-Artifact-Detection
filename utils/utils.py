import typing
import time
import os
import json
import random
import torch
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from tqdm import tqdm
from sklearn import metrics
from PIL import Image
from torchvision import transforms

import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from pathlib import Path
import numpy as np

image_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
    
def load_images(image_list):
    images = []
    for img in image_list:
        if isinstance(img, Image.Image):
            images.append(image_transform(img))
        else:
            images.append(image_transform(Image.open(img).convert("RGB")))
    return torch.stack(images)   # [B, 3, 224, 224]

def build_grid_boxes(batch_size, img_size=224, grid=7):
    boxes = []
    step = img_size // grid
    for _ in range(batch_size):
        b = []
        for i in range(grid):
            for j in range(grid):
                x1 = j * step
                y1 = i * step
                x2 = (j + 1) * step
                y2 = (i + 1) * step
                b.append([x1, y1, x2, y2])
        boxes.append(b)
    return torch.tensor(boxes, dtype=torch.float)

class SpectrogramCNN(nn.Module):
    def __init__(self, out_channels=512, in_channels=1):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )

        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.out_channels = out_channels

    def forward(self, x):
        return self.feature_extractor(x)

class VisualFeatureExtractor(nn.Module):
    def __init__(self, num_tokens=16, in_channels=1):
        super().__init__()

        self.backbone = SpectrogramCNN(in_channels=in_channels)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.num_tokens = num_tokens

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, spectrogram):
        feat_map = self.backbone(spectrogram)
        pooled = self.pool(feat_map) 

        pooled = pooled.flatten(2) 
        pooled = pooled.transpose(1, 2)

        visual_attention_mask = torch.ones(
            pooled.size(0), pooled.size(1),
            device=pooled.device,
            dtype=torch.long
        )

        return pooled, visual_attention_mask


def collate_fn(batch):
    
    input_ids = torch.stack([item["input_ids"] for item in batch])         
    attention_mask = torch.stack([item["attention_mask"] for item in batch]) 
    images = torch.stack([item["image"] for item in batch])                  
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    files = [item["file"] for item in batch]                                 
    sources = [item["source"] for item in batch]                            

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image": images,
        "label": labels,
        "file": files,
        "source": sources
    }

def fit(model, train_loader, optimizer, num_epochs):
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Training", dynamic_ncols=True):
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            images = batch["images"].to(model.device)
            labels = batch["labels"].to(model.device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                labels=labels
            )

            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        tqdm.write(f"Epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.4f}")
        
def predict(model, dataloader):

    model.eval()
    pred_label = []
    true_label = []
    prob_list = []
    file_list = []
    source_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in tqdm(dataloader, desc="Predicting", leave=False, dynamic_ncols=True):    
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images
            )
            logits = outputs["logits"]

            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1)

            pos_probs = probs[:, 0] 

        pred_label.extend(prediction.cpu().numpy().tolist())
        true_label.extend(labels.cpu().numpy().tolist())
        prob_list.extend(pos_probs.cpu().numpy().tolist())

        file_list.extend(batch["file"])
        source_list.extend(batch["source"])

    return true_label, pred_label, prob_list, file_list, source_list


def save_model(model, tokenizer, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_path[-1] == "/":
        save_path += "/"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
def save_result(y_true : typing.List, y_pred : typing.List, y_prob : typing.Optional[typing.List]=None, file: typing.Optional[typing.List] = None, source: typing.Optional[typing.List] = None, save_path : str='./', save_report_name : str = 'result') -> str:

    result = pd.DataFrame({'True': y_true, 'Predicted': y_pred})
    if y_prob is not None:
        result['Probability'] = y_prob
    if file is not None:
        result['file'] = file
    if source is not None:
        result['source'] = source
    result.to_csv(f'{save_path}{save_report_name}_prediction.csv', index=False)
    return f'{save_path}{save_report_name}_prediction.csv'

def plot_confusion_matrix(y_true : typing.List, y_pred : typing.List, save_path : str, plot_name : typing.Optional[str]=None) -> None:

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    sns.heatmap(confusion_matrix, fmt='d', cmap='Blues')
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix)):
            color = 'white' if ((confusion_matrix[i, j] - confusion_matrix.min()) / (confusion_matrix.max() - confusion_matrix.min())) > 0.5 else 'black'
            plt.text(j+0.5, i+0.5, str(confusion_matrix[i, j]), ha='center', va='center', color=color, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    if plot_name is not None:
        plt.title(plot_name)
        plt.savefig(f'{save_path}/{plot_name}_confusion_matrix.png')
    else:
        plt.title('Confusion Matrix')
        plt.savefig(f'{save_path}/confusion_matrix.png')
    plt.close()

def compute_criterion(
    y_true: typing.List,
    y_pred: typing.List,
    y_prob: typing.Optional[typing.List] = None,
    criterion: typing.List[typing.Literal[
        'accuracy', 'precision', 'recall', 'f1',
        'weighted_accuracy', 'specificity', 'sensitivity', 'AUC'
    ]] = None,
    save_path: str = './',
    save_report_name: str = 'report'
) -> None:
    result = {}

    labels = np.unique(y_true)
    average = 'weighted' if len(labels) > 2 else 'binary'

    if 'accuracy' in criterion:
        result['accuracy'] = round(metrics.accuracy_score(y_true, y_pred), 4)
    if 'precision' in criterion:
        result['precision'] = round(metrics.precision_score(y_true, y_pred, average=average, zero_division=0), 4)
    if 'recall' in criterion:
        result['recall'] = round(metrics.recall_score(y_true, y_pred, average=average, zero_division=0), 4)
    if 'f1' in criterion:
        result['f1'] = round(metrics.f1_score(y_true, y_pred, average=average, zero_division=0), 4)
    if 'weighted_accuracy' in criterion:
        result['weighted_accuracy'] = round(metrics.balanced_accuracy_score(y_true, y_pred), 4)

    if 'specificity' in criterion or 'sensitivity' in criterion:
        cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
        if len(labels) == 2:
            tn, fp, fn, tp = cm.ravel()
            if 'specificity' in criterion:
                result['specificity'] = round(tn / (tn + fp), 4)
            if 'sensitivity' in criterion:
                result['sensitivity'] = round(tp / (tp + fn), 4)
        else:
            result['specificity'] = 'N/A'
            result['sensitivity'] = 'N/A'

    if 'AUC' in criterion:
        if y_prob is not None:
            try:
                if len(labels) == 2:
                    result['AUC'] = round(metrics.roc_auc_score(y_true, y_prob), 4)
                else:
                    result['AUC'] = round(metrics.roc_auc_score(y_true, y_prob, multi_class='ovr'), 4)
            except:
                result['AUC'] = -1
        else:
            result['AUC'] = -1

    df = pd.DataFrame(result, index=[0])
    df.to_csv(f'{save_path}/{save_report_name}.csv', index=False)
            
    # save to csv
    result = pd.DataFrame(result, index=[0])
    result.to_csv(f'{save_path}/{save_report_name}.csv', index=False)
    

def plot_roc_curve(y_true : typing.List, y_pred : typing.List, save_path : str, plot_name : typing.Optional[str]=None) -> None:

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {metrics.auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.1, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if plot_name is not None:
        plt.title(plot_name)
    else:
        plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.title('ROC Curve')
    plt.savefig(f'{save_path}/roc_curve.png')
    plt.close()
    
def execute_time(start_time: float) -> None:

    end_time = time.time()
    # show the execution time in HH:MM:SS
    print(f"Execution Time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    
def resize_data(json_file_path : str, sample_size : int) -> str:

    with open(json_file_path, 'r') as f:
        data = json.load(f)
    # positive_data from 0 to 39999
    pos_data = data[:40000]
    # negative_data from 40000 to 79999
    neg_data = data[40000:]
    neg_size, pos_size = sample_size // 2, sample_size // 2
    # randomly select the data
    pos_data = random.sample(pos_data, pos_size)
    neg_data = random.sample(neg_data, neg_size)
    data = pos_data + neg_data
    with open(f'./tmp_{sample_size}.json', 'w') as f:
        json.dump(data, f)
    return f'./tmp_{sample_size}.json'

def list_all_json_file(path: str) -> typing.List[str]:

    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.json' in file:
                files.append(os.path.join(r, file))
    return files