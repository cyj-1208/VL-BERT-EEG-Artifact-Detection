"""
Supported datasets:
    --dataset tuar
    --dataset cgmh
    --dataset all

Supported features:
    --feature stft
    --feature standard
    --feature wavelet
    --feature band_pass
    --feature all

Image generation:
    --image true
    --image false

"""
import argparse
import json
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt, stft, welch
from scipy.stats import kurtosis, skew
from tqdm import tqdm

SAMPLE_RATE = 100
TUAR_INPUT_PATH = "/Group16T/common/cyj/code/channel_tuar"
TUAR_SAVE_PATH = "/Group16T/common/cyj/code/tuar_stft_spec_prompt"

TUAR_TEMPLATE_PATHS = {
    "stft": "./prompt/channel_report_template_stft.txt",
    "standard": "./prompt/channel_report_template_standard.txt",
    "bandpass": "./prompt/channel_report_template_bandpass.txt",
    "wavelet": "./prompt/channel_report_template_wavelet.txt"
}

CGMH_INPUT_PATH = "/Group16T/common/cyj/code/channel_cgmh_with_seizure"
CGMH_SAVE_PATH = "/Group16T/common/cyj/code/cgmh_stft_spec_prompt"

CGMH_TEMPLATE_PATHS = {
    "stft": "./prompt/channel_report_template_stft.txt",
    "standard": "./prompt/channel_report_template_standard.txt",
    "bandpass": "./prompt/channel_report_template_bandpass.txt",
    "wavelet": "./prompt/channel_report_template_wavelet.txt"
}


TUAR_PRIOR_PATH = "channel_summary.csv"

TUAR_TARGET_LABELS = ["artifact", "non_artifact"]
TUAR_TARGET_FILES = {
    "artifact": 20000,
    "non_artifact": 20000,
}

TUAR_EXCLUDE_CHANNELS = []

CGMH_PRIOR_PATH = "channel_counts_cgmh.csv"

CGMH_TARGET_LABELS = ["artifact", "non_artifact"]

CGMH_EXCLUDE_CHANNELS = []

RANDOM_SEED = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def str2bool(value):
    """Convert command-line true/false text to bool."""
    value = value.lower()

    if value in ("true", "1", "yes", "y"):
        return True

    if value in ("false", "0", "no", "n"):
        return False

    raise argparse.ArgumentTypeError(
        "Expected true or false."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified TUAR/CGMH EEG prompt preprocessing."
    )

    parser.add_argument(
        "--dataset",
        choices=["cgmh", "tuar", "all"],
        required=True,
        help="Dataset to preprocess.",
    )

    parser.add_argument(
        "--feature",
        choices=["stft", "standard", "wavelet", "bandpass","all"],
        required=True,
        help="Feature type.",
    )

    parser.add_argument(
        "--image",
        type=str2bool,
        default=False,
        help="Generate STFT spectrogram images: true/false.",
    )

    return parser.parse_args()

def load_channel_priors(table_path, dataset):

    if not os.path.exists(table_path):
        logging.warning(
            "Prior CSV not found: %s. Continue without priors.",
            table_path,
        )
        return {}

    df = pd.read_csv(table_path)

    if dataset == "tuar":
        cols = [
            "artifact_eyem",
            "artifact_musc",
            "artifact_chew",
            "artifact_elec",
            "artifact_shiv",
            "non_artifact",
        ]

        for col in cols:
            if col not in df.columns:
                raise ValueError(
                    f"TUAR prior CSV is missing column: {col}"
                )

        total = df[cols].sum(axis=1).replace(0, np.nan)
        probs = df.copy()

        for col in cols:
            probs[col] = (df[col] / total).fillna(0.0)

        priors = {}

        for _, row in probs.iterrows():
            channel = str(row["channel"]).strip()

            priors[channel] = {
                "eyem": float(row["artifact_eyem"]),
                "musc": float(row["artifact_musc"]),
                "chew": float(row["artifact_chew"]),
                "elec": float(row["artifact_elec"]),
                "shiv": float(row["artifact_shiv"]),
                "non_artifact": float(row["non_artifact"]),
            }

        return priors

    if dataset == "cgmh":
        cols = [
            "eye",
            "muscle",
            "electrode",
            "non_artifact",
        ]

        for col in cols:
            if col not in df.columns:
                raise ValueError(
                    f"CGMH prior CSV is missing column: {col}"
                )

        total = df[cols].sum(axis=1).replace(0, np.nan)
        probs = df.copy()

        for col in cols:
            probs[col] = (df[col] / total).fillna(0.0)

        priors = {}

        for _, row in probs.iterrows():
            channel = str(row["channel"]).strip()

            priors[channel] = {
                "eyem": float(row["eye"]),
                "musc": float(row["muscle"]),
                "chew": 0.0,
                "elec": float(row["electrode"]),
                "shiv": 0.0,
                "non_artifact": float(row["non_artifact"]),
            }

        return priors

    return {}

def save_stft_spectrogram(
    data,
    save_path,
    sample_rate=SAMPLE_RATE,
    nperseg=100,
    noverlap=50,
    vmax=None,
):

    data = np.asarray(data).reshape(-1)

    f, t, sxx = stft(
        data,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )

    power = np.log1p(np.abs(sxx))

    plt.figure(figsize=(6, 4))
    plt.pcolormesh(
        t,
        f,
        power,
        shading="gouraud",
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Log Power")

    if vmax is not None:
        plt.clim(0, vmax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

class EEGPreprocessor:

    def __init__(
        self,
        input_path,
        save_path,
        feature_type,
        channel,
        template_path,
        target_labels,
        target_files=None,
        split_data=False,
        generate_image=False,
        channel_priors=None,
    ):
        self.input_path = input_path
        self.save_path = save_path
        self.feature_type = feature_type
        self.channel = channel
        self.template_path = template_path
        self.target_labels = target_labels
        self.target_files = target_files
        self.split_data = split_data
        self.generate_image = generate_image
        self.channel_priors = channel_priors

        os.makedirs(self.save_path, exist_ok=True)

        if self.feature_type not in {
            "stft",
            "standard",
            "wavelet",
            "bandpass"
        }:
            raise ValueError(
                f"Unsupported feature type: {self.feature_type}"
            )

        if self.generate_image and self.feature_type != "stft":
            logging.warning(
                "--image true is only meaningful for STFT. "
                "No images will be generated for %s.",
                self.feature_type,
            )

    def list_files(self, subfolders):
        """Find all .npy files under the specified subfolders."""
        files = []

        for folder in subfolders:
            full_path = os.path.join(self.input_path, folder)

            if not os.path.isdir(full_path):
                continue

            for filename in os.listdir(full_path):
                if filename.endswith(".npy"):
                    files.append(
                        {
                            "path": os.path.join(
                                full_path,
                                filename,
                            ),
                            "source": folder,
                        }
                    )

        random.shuffle(files)

        return files

    def get_file_boundaries(self, files):

        if self.split_data:
            boundaries = []

            for file_info in tqdm(
                files,
                desc="Processing files",
                unit="file",
                total=len(files),
                dynamic_ncols=True,
                leave=False,
            ):
                data = np.load(file_info["path"])
                boundaries.append(data.shape[0])

            return boundaries

        return [1] * len(files)

    @staticmethod
    def generate_bounded_sum_array(bounds, total_sum):

        if total_sum > sum(bounds):
            raise ValueError(
                "Target sum is larger than the total number "
                "of available segments."
            )

        result = [0] * len(bounds)
        remaining = total_sum
        indices = list(range(len(bounds)))

        while remaining > 0:
            i = random.choice(indices)

            if result[i] >= bounds[i]:
                continue

            result[i] += 1
            remaining -= 1

        return result

    def get_data(self, files, boundaries):

        selected_data = []
        sources = []
        file_names = []

        for file_info, count in tqdm(
            zip(files, boundaries),
            desc="Loading data",
            unit="file",
            total=len(files),
            dynamic_ncols=True,
            leave=False,
        ):
            if count == 0:
                continue

            data = np.load(file_info["path"])

            if self.split_data:
                indices = np.random.choice(
                    data.shape[0],
                    size=count,
                    replace=False,
                )

                for index in indices:
                    selected_data.append(
                        np.asarray(data[index]).reshape(-1)
                    )
                    sources.append(file_info["source"])
                    file_names.append(
                        os.path.basename(file_info["path"])
                    )

            else:
                data = np.squeeze(data)

                selected_data.append(
                    np.asarray(data).reshape(-1)
                )
                sources.append(file_info["source"])
                file_names.append(
                    os.path.basename(file_info["path"])
                )

        return (
            np.asarray(selected_data),
            sources,
            file_names,
        )

    @staticmethod
    def get_standard_features(data):
        data = np.asarray(data).reshape(-1)

        features = {}

        std = np.std(data)
        var = np.var(data)

        freq_bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
        }

        freqs, psd = welch(
            data,
            fs=SAMPLE_RATE,
            axis=0,
        )

        band_power = {
            band: np.sum(
                psd[
                    (freqs >= low)
                    & (freqs < high)
                ],
                axis=0,
            )
            for band, (low, high) in freq_bands.items()
        }

        features["CH_standard_deviation"] = float(std)
        features["CH_variance"] = float(var)
        features["CH_max"] = float(np.max(data))
        features["CH_min"] = float(np.min(data))
        features["CH_range"] = float(np.ptp(data))

        features["CH_alpha_delta_ratio"] = np.round(
            band_power["alpha"]
            / band_power["delta"],
            2,
        )

        features["CH_theta_alpha_ratio"] = np.round(
            band_power["theta"]
            / band_power["alpha"],
            2,
        )

        features["CH_delta_theta_ratio"] = np.round(
            band_power["delta"]
            / band_power["theta"],
            2,
        )

        if std < 1e-8:
            k = 0.0
            s = 0.0
        else:
            k = kurtosis(data)
            s = skew(data)

            if not np.isfinite(k):
                k = 0.0

            if not np.isfinite(s):
                s = 0.0

        features["CH_kurtosis"] = float(
            np.round(k, 2)
        )
        features["CH_skewness"] = float(
            np.round(s, 2)
        )

        return features

    @staticmethod
    def get_bandpass_features(data):
        features = {}
        freq_bands = {
            'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
            'beta': (13, 30), 'gamma': (30, 49)
        }
        for band, (low, high) in freq_bands.items():
            b, a = butter(3, [low, high], 'band', fs=SAMPLE_RATE)
            band_data = filtfilt(b, a, data)
            features[f'CH_{band}_power'] = np.sum(np.square(band_data)) / len(band_data)
        return features
        
    @staticmethod
    def get_wavelet_features(data):
        data = np.asarray(data).reshape(-1)

        features = {}

        coeffs = pywt.wavedec(
            data,
            "db4",
            level=5,
        )

        for level in range(1, 6):
            features[
                f"CH_level_{level}_power"
            ] = (
                np.sum(
                    np.square(coeffs[level])
                )
                / len(coeffs[level])
            )

        return features

    @staticmethod
    def get_stft_features(data):
        data = np.asarray(data).reshape(-1)

        features = {}

        f, _, sxx = stft(
            data,
            fs=SAMPLE_RATE,
            window="hann",
            nperseg=100,
            noverlap=50,
        )

        avg_psd = np.mean(
            np.log1p(np.abs(sxx)),
            axis=1,
        )

        freq_bands = {
            "delta": (0, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, len(f) + 1),
        }

        for band, (low, high) in freq_bands.items():
            features[
                f"CH_{band}_power"
            ] = np.mean(
                avg_psd[low:high]
            )

        return features

    def extract_features(self, data):
        if self.feature_type == "standard":
            return self.get_standard_features(data)

        if self.feature_type == "wavelet":
            return self.get_wavelet_features(data)

        if self.feature_type == "stft":
            return self.get_stft_features(data)
            
        if self.feature_type == "bandpass":
            return self.get_bandpass_features(data)

        raise ValueError(
            f"Unsupported feature type: {self.feature_type}"
        )

    def load_template(self):
        if not os.path.exists(self.template_path):
            raise FileNotFoundError(
                f"Template file not found: "
                f"{self.template_path}"
            )

        with open(
            self.template_path,
            "r",
            encoding="utf-8",
        ) as file:
            return file.read()

    @staticmethod
    def compose_report(features, template):
        
        for key, value in features.items():

            if isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):
                replacement = f"{float(value):.4e}"
            else:
                replacement = str(value)

            template = template.replace(
                key,
                replacement,
            )

        return template

    def generate_spectrogram(
        self,
        data,
        label_name,
        file_name,
        index,
        mode,
    ):

        if not self.generate_image:
            return None

        #if self.feature_type != "stft":
            #return None

        spec_dir = os.path.join(
            self.save_path,
            "spectrogram",
            self.channel,
            mode,
        )

        os.makedirs(
            spec_dir,
            exist_ok=True,
        )

        stem = os.path.splitext(
            os.path.basename(file_name)
        )[0]

        spec_name = (
            f"{label_name}_{stem}_{index}.png"
        )

        abs_spec_path = os.path.join(
            spec_dir,
            spec_name,
        )

        save_stft_spectrogram(
            data,
            save_path=abs_spec_path,
            sample_rate=SAMPLE_RATE,
        )

        return os.path.join(
            self.channel,
            mode,
            spec_name,
        )

    def process_label(
        self,
        label_name,
        subfolders,
        file_count=None,
    ):
        label_files = self.list_files(
            subfolders
        )

        logging.info(
            "%s files: %d",
            label_name,
            len(label_files),
        )

        if not label_files:
            logging.warning(
                "No .npy files found for label: %s",
                label_name,
            )
            return [], []

        boundaries = self.get_file_boundaries(
            label_files
        )

        if self.split_data:
            if file_count is None:
                raise ValueError(
                    "TUAR requires file_count."
                )

            boundaries = (
                self.generate_bounded_sum_array(
                    boundaries,
                    total_sum=file_count,
                )
            )

            logging.info(
                "Total selected segments for %s: %d",
                label_name,
                sum(boundaries),
            )

        else:
            logging.info(
                "Total segments for %s: %d",
                label_name,
                len(label_files),
            )

        all_data, all_sources, all_filenames = (
            self.get_data(
                label_files,
                boundaries,
            )
        )

        if len(all_data) == 0:
            return [], []

        if self.split_data:
            indices = np.arange(
                len(all_data)
            )
            np.random.shuffle(indices)

            split_index = int(
                len(all_data) * 0.8
            )

            train_indices = indices[
                :split_index
            ]
            test_indices = indices[
                split_index:
            ]

            train_output = self.build_output(
                label_name=label_name,
                data=all_data[train_indices],
                sources=[
                    all_sources[i]
                    for i in train_indices
                ],
                file_names=[
                    all_filenames[i]
                    for i in train_indices
                ],
                mode="train",
            )

            test_output = self.build_output(
                label_name=label_name,
                data=all_data[test_indices],
                sources=[
                    all_sources[i]
                    for i in test_indices
                ],
                file_names=[
                    all_filenames[i]
                    for i in test_indices
                ],
                mode="test",
            )

            return train_output, test_output
            
        test_output = self.build_output(
            label_name=label_name,
            data=all_data,
            sources=all_sources,
            file_names=all_filenames,
            mode="test",
        )

        return [], test_output

    def build_output(
        self,
        label_name,
        data,
        sources,
        file_names,
        mode,
    ):
        template = None

        if self.feature_type != "raw":
            template = self.load_template()

        output = []

        for i in tqdm(
            range(len(data)),
            desc=(
                f"{label_name} "
                f"{mode} - "
                f"{self.feature_type}"
            ),
            unit="file",
            dynamic_ncols=True,
        ):
            data_i = data[i]

            features = self.extract_features(
                data_i
            )

            source = os.path.basename(
                sources[i]
            )

            file_name = file_names[i]

            item = {
                "prompt": self.compose_report(
                    features,
                    template,
                ),
                "completion": label_name,
                "source": source,
                "file": os.path.basename(
                    file_name
                ),
            }

            spectrogram = (
                self.generate_spectrogram(
                    data=data_i,
                    label_name=label_name,
                    file_name=file_name,
                    index=i,
                    mode=mode,
                )
            )

            if spectrogram is not None:
                item["spectrogram"] = (
                    spectrogram
                )

            output.append(item)

        return output

    def get_label_subfolders(self):
        folder_groups = {}

        for label in self.target_labels:
            root = os.path.join(
                self.input_path,
                label,
            )

            if not os.path.exists(root):
                folder_groups[label] = []
                continue

            if self.split_data:
                if label == "artifact":
                    subfolders = [
                        os.path.join(
                            label,
                            folder,
                        )
                        for folder in os.listdir(root)
                        if os.path.isdir(
                            os.path.join(
                                root,
                                folder,
                            )
                        )
                    ]
                else:
                    subfolders = [label]

            else:
                subfolders = [
                    os.path.join(
                        label,
                        folder,
                    )
                    for folder in os.listdir(root)
                    if os.path.isdir(
                        os.path.join(
                            root,
                            folder,
                        )
                    )
                ]

            folder_groups[label] = subfolders

        return folder_groups

    def run(self):
        folder_groups = (
            self.get_label_subfolders()
        )

        logging.info(
            "Dataset: %s",
            "TUAR" if self.split_data else "CGMH",
        )

        logging.info(
            "Channel: %s",
            self.channel,
        )

        logging.info(
            "Feature: %s",
            self.feature_type,
        )

        logging.info(
            "Generate image: %s",
            self.generate_image,
        )

        logging.info(
            "Detected folders: %s",
            folder_groups,
        )

        merged_train = []
        merged_test = []

        for label in self.target_labels:
            if label not in folder_groups:
                continue

            file_count = None

            if self.split_data:
                file_count = self.target_files[
                    label
                ]

            train_output, test_output = (
                self.process_label(
                    label_name=label,
                    subfolders=folder_groups[
                        label
                    ],
                    file_count=file_count,
                )
            )

            merged_train.extend(
                train_output
            )
            merged_test.extend(
                test_output
            )

        if self.split_data:
            train_path = os.path.join(
                self.save_path,
                (
                    f"output_{self.channel}"
                    f"_train_{self.feature_type}.json"
                ),
            )

            test_path = os.path.join(
                self.save_path,
                (
                    f"output_{self.channel}"
                    f"_test_{self.feature_type}.json"
                ),
            )

            with open(
                train_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    merged_train,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            with open(
                test_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    merged_test,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            logging.info(
                "Saved TUAR train: %s",
                train_path,
            )

            logging.info(
                "Saved TUAR test: %s",
                test_path,
            )

        else:
            test_path = os.path.join(
                self.save_path,
                (
                    f"output_{self.channel}"
                    f"_test_{self.feature_type}.json"
                ),
            )

            with open(
                test_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    merged_test,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            logging.info(
                "Saved CGMH test: %s",
                test_path,
            )

        logging.info(
            "Preprocessing completed: %s",
            self.channel,
        )

def get_dataset_config(dataset, feature):
    if dataset == "tuar":
        return {
            "input_path": TUAR_INPUT_PATH,
            "save_path": TUAR_SAVE_PATH,
            "template_path": TUAR_TEMPLATE_PATHS[feature],
            "prior_path": TUAR_PRIOR_PATH,
            "target_labels": TUAR_TARGET_LABELS,
            "target_files": TUAR_TARGET_FILES,
            "exclude_channels": TUAR_EXCLUDE_CHANNELS,
            "split_data": True,
        }

    if dataset == "cgmh":
        return {
            "input_path": CGMH_INPUT_PATH,
            "save_path": CGMH_SAVE_PATH,
            "template_path": CGMH_TEMPLATE_PATHS[feature],
            "prior_path": CGMH_PRIOR_PATH,
            "target_labels": CGMH_TARGET_LABELS,
            "target_files": None,
            "exclude_channels": CGMH_EXCLUDE_CHANNELS,
            "split_data": False,
        }

    raise ValueError(f"Unsupported dataset: {dataset}")

def main():
    args = parse_args()

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
    if args.dataset == "all":
        datasets = ["cgmh", "tuar"]
    else:
        datasets = [args.dataset]
    
    if args.feature == "all":
        features = ["stft", "standard", "bandpass", "wavelet"]
    else:
        features = [args.feature]
    
    for dataset in datasets:
    
        for feature in features:
    
            print(f"\nProcessing dataset: {dataset}")
            print(f"Processing feature: {feature}")
    
            config = get_dataset_config(dataset, feature)
    
            logging.info("Starting preprocessing")
            logging.info("Dataset : %s", dataset)
            logging.info("Feature : %s", feature)
            logging.info("Image   : %s", args.image)
    
            priors = load_channel_priors(
                config["prior_path"],
                dataset,
            )
    
            base_input_path = config["input_path"]
    
            if not os.path.exists(base_input_path):
                raise FileNotFoundError(
                    f"Input path not found: {base_input_path}"
                )
    
            channel_dirs = sorted(
                [
                    d
                    for d in os.listdir(base_input_path)
                    if os.path.isdir(
                        os.path.join(base_input_path, d)
                    )
                    and not d.startswith(".")
                    and d not in config["exclude_channels"]
                    and d != "spectrogram_png"
                ]
            )
    
            if not channel_dirs:
                raise RuntimeError(
                    f"No channel directories found in {base_input_path}"
                )
    
            logging.info(
                "Channels to process: %d",
                len(channel_dirs),
            )
    
            for channel in channel_dirs:
    
                print(f"\nProcessing channel: {channel}")
    
                preprocessor = EEGPreprocessor(
                    input_path=os.path.join(
                        base_input_path,
                        channel,
                    ),
                    save_path=config["save_path"],
                    feature_type=feature,
                    channel=channel,
                    template_path=config["template_path"],
                    target_labels=config["target_labels"],
                    target_files=config["target_files"],
                    split_data=config["split_data"],
                    generate_image=args.image,
                    channel_priors=priors,
                )
    
                preprocessor.run()
    
            logging.info("All channels completed.")

if __name__ == "__main__":
    main()
