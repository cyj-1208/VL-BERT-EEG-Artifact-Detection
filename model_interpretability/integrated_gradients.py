import argparse
import os
import time
from explain_tokens import explain_dataset_pipeline

MODEL_SAVE_PATH_TEMPLATE = "/home/user/code/bert/models_vlbert/{channel}/"
RESULT_SAVE_PATH_TEMPLATE = "/home/user/code/bert/results_vlbert/{channel}/"

MODEL_ID_LIST = [
    'google-bert/bert-base-uncased'
]

FEATUE_TYPE_LIST = [
    'stft'
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

def main(model_list, feature_list, repeat):

    for channel in CHANNEL_LIST:

        model_base_path = MODEL_SAVE_PATH_TEMPLATE.format(channel=channel)
        result_base_path = RESULT_SAVE_PATH_TEMPLATE.format(channel=channel)
    
        os.makedirs(model_base_path, exist_ok=True)
        os.makedirs(result_base_path, exist_ok=True)

        for feature_type in feature_list:

            test_jsons = {
                "tuar": f"../tuar_stft_spec_prompt/output_{channel}_test_{feature_type}.json",
                "cgmh": f"../cgmh_stft_spec_prompt/output_{channel}_test_{feature_type}.json",
            }

            for model_id in model_list:
                
                for r in range(repeat):

                    print(
                        f"channel: {channel}, "
                        f"model_name: {model_id}, "
                        f"feature_type: {feature_type}, "
                        f"repeat: {r}"
                    )

                    model_save_path = (
                        f"{model_base_path}"
                        f"{model_id}_{feature_type}_{r}/"
                    )

                    result_save_path = (
                        f"{result_base_path}"
                        f"{model_id}_{feature_type}_{r}/"
                    )

                    os.makedirs(model_save_path, exist_ok=True)
                    os.makedirs(result_save_path, exist_ok=True)

                    
                    for test_dataset_name, test_json in test_jsons.items():

                        test_img_root = DATASET_CONFIG[test_dataset_name]["img_root"]

                        print(
                            f"[EXPLAIN] "
                            f"dataset={test_dataset_name}"
                        )

                        explain_out_dir = os.path.join(
                            result_save_path,
                            "explanations",
                            test_dataset_name
                        )

                        explain_dataset_pipeline(
                            model_dir=model_save_path,
                            model_name=model_id,
                            json_path=test_json,
                            img_root=test_img_root,
                            save_dir=explain_out_dir,
                            steps=64, 
                            test_mode=False
                        )
                    

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