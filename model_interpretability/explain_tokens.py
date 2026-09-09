import os
import json
import csv
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from utils_vlbert.utils import load_images, VisualFeatureExtractor
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class VisualLinguisticBertForPretraining(nn.Module):
    def __init__(self, model_name, visual_dim=512, num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        self.visual_extractor = VisualFeatureExtractor(num_tokens=16, in_channels=1)
        self.visual_proj = nn.Linear(visual_dim, hidden)
        self.visual_ln = nn.LayerNorm(hidden)
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, images, labels=None):

        text_embeds = self.bert.embeddings(input_ids=input_ids)

        visual_feats, visual_mask = self.visual_extractor(images)
        visual_embeds = self.visual_proj(visual_feats)
        visual_embeds = self.visual_ln(visual_embeds)

        embeddings = torch.cat([text_embeds, visual_embeds], dim=1)
        combined_mask = torch.cat([attention_mask, visual_mask], dim=1)
        
        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=combined_mask, 
            return_dict=True
        )

        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(cls_output))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

def load_pt_model(model_dir, model_name):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VisualLinguisticBertForPretraining(
        model_name=model_name
    )

    state_dict = torch.load(
        os.path.join(model_dir, "vlbert.pt"),
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

class ExplainEEGDataset(Dataset):

    def __init__(self, json_path, tokenizer, img_root):

        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.img_root = img_root

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        max_text_tokens = 512 - 16

        encoding = self.tokenizer(
            item["prompt"],
            padding="max_length",
            truncation=True,
            max_length=max_text_tokens,
            return_tensors="pt"
        )

        img_path = os.path.join(self.img_root, item["spectrogram"])
        img = Image.open(img_path).convert("L")
        img = self.img_transform(img)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": img,
            "text": item["prompt"]
        }

def text_integrated_gradients(
    model,
    input_ids,
    attention_mask,
    images,
    steps=32
):

    device = next(model.parameters()).device

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    images = images.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images
        )
        logits = outputs["logits"]
        target_class = torch.argmax(logits, dim=-1).item()

    emb_layer = model.bert.embeddings
    input_emb = emb_layer(input_ids)

    baseline_ids = torch.zeros_like(input_ids)
    baseline_emb = emb_layer(baseline_ids)

    total_grads = torch.zeros_like(input_emb)

    for alpha in torch.linspace(0, 1, steps, device=device):

        interp_emb = baseline_emb + alpha * (input_emb - baseline_emb)
        interp_emb.requires_grad_(True)

        outputs = model(
            inputs_embeds=interp_emb,
            attention_mask=attention_mask,
            images=images
        )

        target_logit = outputs["logits"][0, target_class]

        grads = torch.autograd.grad(
            target_logit,
            interp_emb
        )[0]

        total_grads += grads

    avg_grads = total_grads / steps
    attributions = (input_emb - baseline_emb) * avg_grads

    token_scores = attributions.sum(dim=-1).squeeze(0)

    tokens = model.tokenizer.convert_ids_to_tokens(
        input_ids.squeeze(0)
    )

    return tokens, token_scores.detach().cpu().tolist(), target_class

def image_integrated_gradients(
    model,
    input_ids,
    attention_mask,
    images,
    steps=32
):

    device = next(model.parameters()).device

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    images = images.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images
        )
        logits = outputs["logits"]
        target_class = torch.argmax(logits, dim=-1).item()

    visual_feats, _ = model.visual_extractor(images)

    baseline = torch.zeros_like(visual_feats)

    total_grads = torch.zeros_like(visual_feats)

    for alpha in torch.linspace(0, 1, steps, device=device):

        interp = baseline + alpha * (visual_feats - baseline)
        interp.requires_grad_(True)

        visual_embeds = model.visual_proj(interp)
        visual_embeds = model.visual_ln(visual_embeds)

        text_embeds = model.bert.embeddings(input_ids)

        embeddings = torch.cat([text_embeds, visual_embeds], dim=1)

        combined_mask = torch.cat(
            [attention_mask,
             torch.ones(
                 (attention_mask.size(0), visual_embeds.size(1)),
                 device=device
             )],
            dim=1
        )

        outputs = model.bert(
            inputs_embeds=embeddings,
            attention_mask=combined_mask,
            return_dict=True
        )

        cls = outputs.last_hidden_state[:, 0]
        logits = model.classifier(model.dropout(cls))

        target_logit = logits[0, target_class]

        grads = torch.autograd.grad(
            target_logit,
            interp
        )[0]

        total_grads += grads

    avg_grads = total_grads / steps

    attributions = (visual_feats - baseline) * avg_grads

    return attributions.squeeze(0).detach().cpu(), target_class
    

def save_text_heatmap(tokens, attributions, save_path,
                      max_len=110):

    tokens = tokens[:max_len]
    attributions = attributions[:max_len]

    filtered = [
        (tok, attr)
        for tok, attr in zip(tokens, attributions)
        if tok not in ["[PAD]", "[CLS]", "[SEP]"]
    ]

    if len(filtered) == 0:
        return

    tokens, attributions = zip(*filtered)
    tokens = list(tokens)

    importances = np.abs(np.array(attributions))

    plt.figure(figsize=(8, 18))

    y_positions = np.arange(len(tokens))

    plt.barh(y_positions, importances, color="#1f77b4")

    plt.yticks(y_positions, tokens)
    plt.xlabel("Integrated Gradients")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_image_heatmap(image_tensor, out_path):

    if image_tensor.dim() == 1:
        image_tensor = image_tensor.unsqueeze(0)

    if image_tensor.dim() == 3:
        img = image_tensor.mean(dim=0)
    else:
        img = image_tensor

    img = img.numpy()

    if img.max() - img.min() < 1e-8:
        img = np.zeros_like(img)
    else:
        img = (img - img.min()) / (img.max() - img.min())

    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap="hot")
    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def aggregate_bands(tokens, scores):

    bands = {
        "delta": 0.0,
        "theta": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 0.0,
    }

    current = None

    for tok, sc in zip(tokens, scores):

        low = tok.lower()

        if "delta" in low:
            current = "delta"
        elif "theta" in low:
            current = "theta"
        elif "alpha" in low:
            current = "alpha"
        elif "beta" in low:
            current = "beta"
        elif "gamma" in low:
            current = "gamma"

        if current:
            bands[current] += sc

    return bands

def multimodal_integrated_gradients(
    model,
    input_ids,
    attention_mask,
    images,
    steps=32
):

    device = next(model.parameters()).device

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    images = images.to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images
        )

        logits = outputs["logits"]
        target_class = torch.argmax(logits, dim=-1).item()

    text_embeds = model.bert.embeddings(
        input_ids=input_ids
    )

    visual_feats, visual_mask = model.visual_extractor(images)

    visual_embeds = model.visual_proj(visual_feats)
    visual_embeds = model.visual_ln(visual_embeds)

    full_embeds = torch.cat(
        [text_embeds, visual_embeds],
        dim=1
    )

    baseline = torch.zeros_like(full_embeds)

    total_grads = torch.zeros_like(full_embeds)

    combined_mask = torch.cat(
        [attention_mask, visual_mask],
        dim=1
    )

    for alpha in torch.linspace(0, 1, steps, device=device):

        interp = baseline + alpha * (
            full_embeds - baseline
        )

        interp.requires_grad_(True)

        outputs = model.bert(
            inputs_embeds=interp,
            attention_mask=combined_mask,
            return_dict=True
        )

        cls = outputs.last_hidden_state[:, 0]

        logits = model.classifier(
            model.dropout(cls)
        )

        target_logit = logits[0, target_class]

        grads = torch.autograd.grad(
            target_logit,
            interp
        )[0]

        total_grads += grads

    avg_grads = total_grads / steps

    attributions = (
        full_embeds - baseline
    ) * avg_grads

    token_scores = attributions.sum(dim=-1)

    token_scores = token_scores.squeeze(0).detach().cpu()

    text_tokens = model.tokenizer.convert_ids_to_tokens(
        input_ids.squeeze(0)
    )

    image_tokens = [
        f"[IMG_{i}]"
        for i in range(visual_embeds.size(1))
    ]

    full_tokens = text_tokens + image_tokens

    return (
        full_tokens,
        token_scores.tolist(),
        target_class
    )
def remove_special_tokens(
    tokens,
    mean,
    std
):

    special_tokens = {
        "[PAD]",
        "[CLS]",
        "[SEP]",
        "<s>",
        "</s>",
        "<pad>"
    }

    new_tokens = []
    new_mean = []
    new_std = []

    for t, m, s in zip(tokens, mean, std):

        if t in special_tokens:
            continue

        if m < 1e-8:
            continue

        new_tokens.append(t)
        new_mean.append(m)
        new_std.append(s)

    return (
        new_tokens,
        np.array(new_mean),
        np.array(new_std)
    )

def explain_dataset_pipeline(
    model_dir,
    model_name,
    json_path,
    img_root,
    save_dir,
    steps=64, 
    test_mode=False
):

    os.makedirs(save_dir, exist_ok=True)

    model, tokenizer = load_pt_model(
        model_dir,
        model_name
    )

    model.tokenizer = tokenizer

    dataset = ExplainEEGDataset(
        json_path,
        tokenizer,
        img_root
    )

    import random
    random.seed(42)

    artifact_indices = []
    nonartifact_indices = []

    for idx in range(len(dataset)):

        completion = dataset.data[idx]["completion"].lower()

        if completion == "artifact":
            artifact_indices.append(idx)
        else:
            nonartifact_indices.append(idx)

    if test_mode:

        num_samples = 5

        selected_artifacts = random.sample(
            artifact_indices,
            min(num_samples, len(artifact_indices))
        )

        selected_nonartifacts = random.sample(
            nonartifact_indices,
            min(num_samples, len(nonartifact_indices))
        )

        selected_indices = (
            selected_artifacts +
            selected_nonartifacts
        )

        print("TEST MODE samples:")
        print(selected_indices)

    else:

        selected_indices = list(range(len(dataset)))
        
    artifact_scores = []
    nonartifact_scores = []

    token_reference = None

    for idx in tqdm(selected_indices):

        sample = dataset[idx]

        completion = dataset.data[idx]["completion"].lower()

        tokens, scores, pred = multimodal_integrated_gradients(
            model,
            sample["input_ids"].unsqueeze(0),
            sample["attention_mask"].unsqueeze(0),
            sample["image"].unsqueeze(0),
            steps=steps
        )

        scores = np.abs(np.array(scores))

        if token_reference is None:

            token_reference = tokens

        ref_len = len(token_reference)

        if len(scores) < ref_len:

            pad_len = ref_len - len(scores)

            scores = np.pad(
                scores,
                (0, pad_len),
                mode="constant"
            )

        elif len(scores) > ref_len:

            scores = scores[:ref_len]

        if completion == "artifact":

            artifact_scores.append(scores)

        else:

            nonartifact_scores.append(scores)

    artifact_scores = np.array(artifact_scores)
    nonartifact_scores = np.array(nonartifact_scores)

    print("artifact shape:", artifact_scores.shape)
    print("nonartifact shape:", nonartifact_scores.shape)

    if len(artifact_scores) == 0:
        print("ERROR: no artifact samples")
        return

    if len(nonartifact_scores) == 0:
        print("ERROR: no nonartifact samples")
        return

    artifact_mean = artifact_scores.mean(axis=0)
    artifact_std = artifact_scores.std(axis=0)

    nonartifact_mean = nonartifact_scores.mean(axis=0)
    nonartifact_std = nonartifact_scores.std(axis=0)

    tokens = token_reference
    def save_token_importance_csv(
        tokens,
        mean,
        std,
        save_path
    ):

        tokens, mean, std = remove_special_tokens(
            tokens,
            mean,
            std
        )
    
        df = pd.DataFrame({
            "token": tokens,
            "mean": mean,
            "error": std
        })
    
        df.to_csv(
            save_path,
            index=False
        )
    
        print(f"CSV saved: {save_path}")
    save_token_importance_csv(
        tokens,
        artifact_mean,
        artifact_std,
        os.path.join(
            save_dir,
            "artifact_token_importance.csv"
        )
    )
    
    save_token_importance_csv(
        tokens,
        nonartifact_mean,
        nonartifact_std,
        os.path.join(
            save_dir,
            "nonartifact_token_importance.csv"
        )
    )
 
    def plot_bar_with_std(
        tokens,
        mean,
        std,
        title,
        save_path
    ):
    
        tokens, mean, std = remove_special_tokens(
            tokens,
            mean,
            std
        )
    
        x = np.arange(len(tokens))
    
        plt.figure(figsize=(20, 6))
        
        plt.bar(
            x,
            mean,
            alpha=0.8,
            zorder=2
        )
        
        lower_err = np.minimum(std, mean)
        plt.errorbar(
            x,
            mean,
            yerr=[lower_err, std],
            fmt='none',
            ecolor='black',
            elinewidth=0.8,
            capsize=3,
            zorder=3
        )
        
        plt.ylim(bottom=0)
        
        plt.xticks(
            x,
            tokens,
            rotation=90
        )
        
        plt.ylabel("Importance")
        plt.xlabel("Token")
        plt.title(title)
        
        plt.tight_layout()
        
        plt.savefig(
            save_path,
            dpi=300
        )
        
        plt.close()

    plot_bar_with_std(
        tokens,
        artifact_mean,
        artifact_std,
        "Artifact Token Importance",
        os.path.join(
            save_dir,
            "artifact_token_errorbar.png"
        )
    )

    plot_bar_with_std(
        tokens,
        nonartifact_mean,
        nonartifact_std,
        "NonArtifact Token Importance",
        os.path.join(
            save_dir,
            "nonartifact_token_errorbar.png"
        )
    )

    print("Error bar plots saved.")