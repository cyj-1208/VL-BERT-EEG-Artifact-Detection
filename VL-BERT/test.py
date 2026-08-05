import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel, AutoModel, AutoConfig
from utils.dataset import EEGDataset
from utils.utils import collate_fn, predict, save_result, plot_confusion_matrix, plot_roc_curve, compute_criterion
from utils.options import TestingArguments
from torch.utils.data import DataLoader
from transformers import BertModel, AutoTokenizer
from PIL import Image
from torchvision import transforms
from utils.utils import load_images, VisualFeatureExtractor

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
        
def test_llm(
    json_file_path,
    model_path,
    model_name,
    tokenizer,
    batch_size,
    max_length,
    generate_report,
    save_path,
    save_report_name,
    criterion_name,
    img_root
):


    dataset = EEGDataset(json_file_path, tokenizer, max_length=max_length, img_root=img_root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisualLinguisticBertForPretraining(
    model_name=model_name,
    visual_dim=512
    )
    state_dict = torch.load(
        os.path.join(model_path, "vlbert.pt"),
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    y_true, y_pred, y_prob, file_list, source_list = predict(model, dataloader)

    os.makedirs(save_path, exist_ok=True)
    result_csv_path = save_result(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        file=file_list,
        source=source_list,
        save_path=save_path,
        save_report_name=save_report_name
    )

    if generate_report:
        df = pd.read_csv(result_csv_path)

        plot_confusion_matrix(
            df['True'].to_list(),
            df['Predicted'].to_list(),
            save_path,
            save_report_name
        )

        compute_criterion(
            y_true=df['True'].to_list(),
            y_pred=df['Predicted'].to_list(),
            criterion=criterion_name,
            save_path=save_path,
            save_report_name=save_report_name
        )

    print("Test finished.")


if __name__ == "__main__":
    test_args = TestingArguments()
    test_args.print_args()
    args = test_args.get_args()
    test_llm(json_file_path=args.json_file_path,
             model_name=args.model_name,
             batch_size=args.batch_size,
             max_length=args.max_length,
             generate_report=args.generate_report,
             save_path=args.save_report_path,
             save_report_name=args.save_report_name,
             criterion_name=args.criterion_name)