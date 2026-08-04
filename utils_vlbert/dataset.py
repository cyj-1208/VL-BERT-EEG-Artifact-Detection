import json
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import os

class EEGDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length, img_root):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
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
        LABEL_MAP = {
            "non_artifact": 0,
            "artifact": 1
        }

        img_path = os.path.join(self.img_root, item["spectrogram"])
        img = Image.open(img_path).convert("L")
        img = self.img_transform(img)

        label = torch.tensor(LABEL_MAP[item["completion"]], dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": img,
            "label": label,
            "file": item["file"], 
            "source": item["source"]   
        }