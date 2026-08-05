import time
import argparse
import os
from utils.utils import execute_time, resize_data
from train import train_llm
from test import test_llm
from explain_tokens import explain_dataset_pipeline
import random
import json
import torch
from utils.dataset import EEGDataset
from transformers import AutoTokenizer

MODEL_SAVE_PATH_TEMPLATE = "/home/user/code/bert/models_vlbert/{channel}/"
RESULT_SAVE_PATH_TEMPLATE = "/home/user/code/bert/results_vlbert/{channel}/"

MODEL_ID_LIST = [
    'google-bert/bert-base-uncased',
    'distilbert/distilbert-base-uncased',
    'FacebookAI/roberta-base',
    'FacebookAI/roberta-large'
]

FEATUE_TYPE_LIST = [
    'standard',
    'band_pass',
    'stft',
    'wavelet',
]

# cgmh no channel "A1-T3", "T4-A2" 
CHANNEL_LIST = [ "C3-CZ", "C3-P3", "C4-P4", "C4-T4", "CZ-C4",
    "F3-C3", "F4-C4", "F7-T3", "F8-T4", "FP1-F3", "FP1-F7",
    "FP2-F4", "FP2-F8", "P3-O1", "P4-O2", "T3-C3", "T3-T5",
    "T4-T6", "T5-O1", "T6-O2", "A1-T3", "T4-A2"
                
]


DATASET_CONFIG = {
    "tuar": {
        "img_root": "/home/user/code/bert/tuar_stft_spec",
    },
    "cgmh": {
        "img_root": "/home/user/code/bert/cgmh_stft_spec",
    },
}

def infer_dataset_name(json_path: str):
    if "tuar" in json_path.lower():
        return "tuar"
    elif "cgmh" in json_path.lower():
        return "cgmh"
    else:
        raise ValueError(f"Unknown dataset for json: {json_path}")

def infer_dataset_name(json_path: str) -> str:
    json_path = json_path.lower()
    if "tuar" in json_path:
        return "tuar"
    elif "cgmh" in json_path:
        return "cgmh"
    else:
        raise ValueError(f"[ERROR] Cannot infer dataset from json path: {json_path}")

def main(model_list, feature_list, repeat):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    for channel in CHANNEL_LIST:

        model_base_path = MODEL_SAVE_PATH_TEMPLATE.format(channel=channel)
        result_base_path = RESULT_SAVE_PATH_TEMPLATE.format(channel=channel)
    
        os.makedirs(model_base_path, exist_ok=True)
        os.makedirs(result_base_path, exist_ok=True)

        for feature_type in feature_list:

            train_json = f"../tuar_stft_cqt_spec_prompt/tuar_stft_cqt_spec_prompt/output_{channel}_train_{feature_type}.json"
            test_jsons = {
                "tuar": f"../tuar_stft_cqt_spec_prompt/tuar_stft_cqt_spec_prompt/output_{channel}_test_{feature_type}.json",
                "cgmh": f"../cgmh_stft_spec_prompt/output_{channel}_test_{feature_type}.json",
            }

            train_dataset_name = infer_dataset_name(train_json)
            train_img_root = DATASET_CONFIG[train_dataset_name]["img_root"]

            for model_id in model_list:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                train_dataset = EEGDataset(
                    json_path=train_json,
                    tokenizer=tokenizer,
                    max_length=512,
                    img_root=train_img_root
                )
                
                
                for r in range(repeat):
                    print(f"channel: {channel}, model_name: {model_id}, feature_type: {feature_type}, repeat: {r}")

                    model_save_path = f"{model_base_path}{model_id}_{feature_type}_{r}/"
                    result_save_path = f"{result_base_path}{model_id}_{feature_type}_{r}/"

                    os.makedirs(model_save_path, exist_ok=True)
                    os.makedirs(result_save_path, exist_ok=True)

                    
                    training_start_time = time.time()
                    train_llm(
                        json_file_path=train_json,
                        model_name=model_id,
                        batch_size=4,
                        max_length=512,
                        num_epochs=3,
                        lr=1e-6,
                        model_output_path=model_save_path,
                        device=device,
                        dataset=train_dataset,
                        tokenizer=tokenizer

                    )
                    execute_time(training_start_time)
                    
                    
                    for test_dataset_name, test_json in test_jsons.items():
                        test_img_root = DATASET_CONFIG[test_dataset_name]["img_root"]

                        print(
                            f"[TEST] train={train_dataset_name} → "
                            f"test={test_dataset_name}"
                        )

                        test_llm(
                            json_file_path=test_json,
                            model_path=model_save_path,
                            model_name=model_id,
                            batch_size=4,
                            max_length=512,
                            generate_report=True,
                            save_path=result_save_path,
                            tokenizer=tokenizer,
                            img_root=test_img_root,
                            save_report_name=f"report_{test_dataset_name}",
                            criterion_name=[
                                "weighted_accuracy",
                                "specificity",
                                "sensitivity",
                                "f1",
                                "AUC",
                                "precision"
                            ]
                        )
                    '''
                    explain_out_dir = os.path.join(result_save_path, "explanations")

                    explain_dataset_pipeline(
                        model_dir=model_save_path,
                        model_name=model_id,
                        json_path=test_json,
                        img_root=test_img_root,
                        save_dir=explain_out_dir,
                        steps=64, 
                        test_mode=False    
                    )
                    '''

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run the workflow")
    parser.add_argument("--feature", type=str, choices=FEATUE_TYPE_LIST + ["all"], required=True, help="Feature type to use")
    parser.add_argument("--model", type=str, choices=MODEL_ID_LIST + ["all"], required=True, help="Model to use")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeat")
    args = parser.parse_args()

    model_list = MODEL_ID_LIST if args.model == "all" else [args.model]
    feature_list = FEATUE_TYPE_LIST if args.feature == "all" else [args.feature]
    
    main(model_list, feature_list, args.repeat)
    execute_time(start_time)