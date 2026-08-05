import ssl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils.dataset import EEGDataset
from utils.utils import collate_fn, fit, save_model
from utils.options import TrainingArguments
from transformers import BertModel
from utils.utils import load_images, VisualFeatureExtractor
from tqdm import tqdm
from transformers import AutoModel, AutoConfig

ssl._create_default_https_context = ssl._create_unverified_context

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

def train_llm(json_file_path, model_name, batch_size, max_length, num_epochs, lr, model_output_path, device, dataset, tokenizer):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VisualLinguisticBertForPretraining(
    model_name = model_name,
    visual_dim = 512
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(num_epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            images = batch["image"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                labels=labels
            )
            
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, loss={loss.item():.4f}")

    torch.save(model.state_dict(), f"{model_output_path}/vlbert.pt")
    tokenizer.save_pretrained(model_output_path)
    
if __name__ == "__main__":
    train_args = TrainingArguments()
    train_args.print_args()
    args = train_args.get_args()
    train_llm(args.json_file_path,
             args.model_name,
             args.batch_size,
             args.max_length,
             args.num_epochs,
             args.lr,
             args.model_output_path)