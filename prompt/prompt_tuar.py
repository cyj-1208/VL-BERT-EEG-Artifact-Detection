import os
import librosa
import librosa.display
import random
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import kurtosis, skew
from scipy.signal import welch, filtfilt, butter, stft, find_peaks, peak_widths
import pywt
import re
from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SAMPLE_RATE = 100

FEATURE_FUNCTIONS = {
    'standard': lambda data: EEGPreprocessor.get_standard_features(data),
    'wavelet': lambda data: EEGPreprocessor.get_wavelet_features(data),
    'band_pass': lambda data: EEGPreprocessor.get_band_pass_features(data),
    'stft': lambda data: EEGPreprocessor.get_stft_features(data),
    'stft_ICL': lambda data: EEGPreprocessor.get_stft_features(data),
    'wavelet_ICL': lambda data: EEGPreprocessor.get_wavelet_features(data),
}

def load_channel_priors(table_path: str):
    df = pd.read_csv(table_path)

    cols = ["artifact_eyem","artifact_musc","artifact_chew","artifact_elec","artifact_shiv","non_artifact"]
    s = df[cols].sum(axis=1).replace(0, np.nan)
    probs = df.copy()
    for c in cols:
        probs[c] = (df[c] / s).fillna(0.0)

    priors = {}
    for _, r in probs.iterrows():
        ch = str(r["channel"]).strip()
        priors[ch] = {
            "eyem": float(r["artifact_eyem"]),
            "musc": float(r["artifact_musc"]),
            "chew": float(r["artifact_chew"]),
            "elec": float(r["artifact_elec"]),
            "shiv": float(r["artifact_shiv"]),
            "non_artifact": float(r["non_artifact"]),
        }
    return priors
def _remove_trailing_icl(filename: str) -> str:
    return re.sub(r'_ICL(?=\.json$)', '', filename, flags=re.IGNORECASE)

    
def save_stft_spectrogram(
        data,
        save_path,
        sample_rate=200,
        nperseg=100,
        noverlap=50,
        vmax=None
    ):
        f, t, Sxx = stft(
            data,
            fs=sample_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap
        )
    
        power = np.log1p(np.abs(Sxx))
    
        plt.figure(figsize=(6, 4))
        plt.pcolormesh(t, f, power, shading='gouraud')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.colorbar(label='Log Power')
        if vmax is not None:
            plt.clim(0, vmax)
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
class EEGPreprocessor:
    def __init__(self, input_path, save_path, feature_type, target_files_list=None, target_labels=None, channel=None, channel_priors=None):
        self.input_path = input_path
        self.save_path = save_path
        self.feature_type = feature_type
        if len(target_files_list) != len(target_labels):
            raise ValueError("Length of target_files_list must match length of target_labels")
        self.target_files_list = target_files_list
        self.target_labels = target_labels or ['non_artifact', 'artifact']
        self.channel = channel or os.path.basename(input_path)
        self.channel_priors = channel_priors 

    def list_files(self, sub_folders):
        total_files = []
        for folder in sub_folders:
            full_path = os.path.join(self.input_path, folder)
            if not os.path.isdir(full_path):
                continue
            for file in os.listdir(full_path):
                if file.endswith('.npy'):
                    total_files.append({
                        'path': os.path.join(full_path, file),
                        'source': folder
                    })
        random.shuffle(total_files)
        return total_files

    def get_file_boundaries(self, files):
        boundaries = []
        for file in tqdm(files, desc='Processing files', unit='file', total=len(files), dynamic_ncols=True, leave=False, position=0):
            data = np.load(file['path'])
            boundaries.append(data.shape[0])
        return boundaries
    
    def generate_bounded_sum_array(self, bounds, total_sum=None):
        if total_sum > sum(bounds):
            raise ValueError("Target sum is larger than the total of bounds!")

        result = [0] * len(bounds)
        remaining_sum = total_sum
        indices = list(range(len(bounds)))

        while remaining_sum > 0:
            i = random.choice(indices)
            if result[i] >= bounds[i]:
                continue
            max_can_add = bounds[i] - result[i]
            to_add = min(remaining_sum, max_can_add, 1)
            result[i] += to_add
            remaining_sum -= to_add

        return result
    
 
    def get_data(self, files, boundaries):
        selected_data = []
        sources = []
        file_names = []
        for file, boundary in tqdm(zip(files, boundaries), desc='Loading data', unit='file', total=len(files), dynamic_ncols=True, leave=False, position=0):
            data = np.load(file['path'])
            indices = np.random.choice(data.shape[0], size=boundary, replace=False)
            selected_data += [data[i] for i in indices]
            sources += [file['source']] * boundary
            file_names += [os.path.basename(file['path'])] * boundary
        return np.array(selected_data), sources, file_names

            
    def get_standard_features(data):
        data = np.asarray(data).reshape(-1)
    
        features = {}
        std = np.std(data)
        var = np.var(data)
        freq_bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13)}
        freqs, psd = welch(data, fs=SAMPLE_RATE, axis=0)
        band_power = {band: np.sum(psd[(freqs >= low) & (freqs < high)], axis=0) for band, (low, high) in freq_bands.items()}
        features["CH_standard_deviation"] = float(std)
        features["CH_variance"] = float(var)
        features["CH_max"] = float(np.max(data))
        features["CH_min"] = float(np.min(data))
        features["CH_range"] = float(np.ptp(data))
        features["CH_alpha_delta_ratio"] = np.round(band_power['alpha'] / band_power['delta'], 2)
        features["CH_theta_alpha_ratio"] = np.round(band_power['theta'] / band_power['alpha'], 2)
        features["CH_delta_theta_ratio"] = np.round(band_power['delta'] / band_power['theta'], 2)

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
    
        features["CH_kurtosis"] = float(np.round(k, 2))
        features["CH_skewness"] = float(np.round(s, 2))
    
        return features
        
    @staticmethod
    def get_wavelet_features(data):
        features = {}
        coeffs = pywt.wavedec(data, 'db4', level=5)
        for k in range(1, 6):
            features[f'CH_level_{k}_power'] = np.sum(np.square(coeffs[k])) / len(coeffs[k])
        return features

    @staticmethod
    def get_band_pass_features(data):
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
    def get_stft_features(data):
        data = np.ravel(data)
        features = {}
        f, _, Sxx = stft(data, fs=SAMPLE_RATE, window='hann', nperseg=100, noverlap=50)
        avg_PSD = np.mean(np.log1p(np.abs(Sxx)), axis=1)
        #avg_PSD = np.abs(Sxx)
        freq_bands = {
            'delta': (0, 4),
            'theta': (4, 8), 'alpha': (8, 13),
            'beta': (13, 30), 'gamma': (30, 50)
        }
        for band, (low, high) in freq_bands.items():
            features[f'CH_{band}_power'] = np.mean(avg_PSD[low:high])
        return features
        
    def extract_multiple_feature_types(self, label_name, subfolders, feature_types, output_prefix, file_count):
        label_files = self.list_files(subfolders)
        logging.info(f"{label_name} files: {len(label_files)}")
        boundaries = self.get_file_boundaries(label_files)
        boundaries = self.generate_bounded_sum_array(boundaries, total_sum=file_count)
        logging.info(f"Total segments for {label_name}: {sum(boundaries)}")
        #final_data = self.get_data(label_files, boundaries)
        final_data, final_sources, final_filenames = self.get_data(label_files, boundaries)

        train_data, train_labels, test_data, test_labels = self.train_test_split(final_data, [label_name] * len(final_data))

        for feature_type in feature_types:
            for mode, data, labels in [("train", train_data, train_labels), ("test", test_data, test_labels)]:
                output = []
                if feature_type != 'raw':
                    template_path = f'/Group16T/common/cyj/code/channel_report_template_stft_ICL.txt'
                    #template_path = f'/Group16T/common/cyj/code/channel_report_template_{feature_type}.txt'
                    with open(template_path, 'r') as f:
                        template = f.read()

                for i in tqdm(range(len(data)), desc=f'{label_name} {mode} - {feature_type}', unit='file', dynamic_ncols=True):
                
                    features = FEATURE_FUNCTIONS[feature_type](data[i])
                
                    file_index = i if mode == "train" else len(train_data) + i
                    file_path = label_files[file_index]['path']
                
                    source = "unknown"
                    for cat in ['eyem', 'musc', 'chew', 'elec', 'shiv', 'non_artifact', 'cpsz', 'fnsz', 'gnsz', 'tcsz']:
                        if cat in file_path:
                            source = cat
                            break
                
                    spec_rel_path = None  

                    if feature_type.startswith('stft'):
                        spec_dir = os.path.join(
                            self.save_path,
                            'spectrogram',
                            self.channel,
                            mode
                        )
                        os.makedirs(spec_dir, exist_ok=True)
                
                        spec_name = f"{label_name}_{os.path.splitext(os.path.basename(file_path))[0]}_{i}.png"
                        abs_spec_path = os.path.join(spec_dir, spec_name)
                
                        save_stft_spectrogram(
                            data[i],
                            save_path=abs_spec_path,
                            sample_rate=SAMPLE_RATE
                        )
                
                        spec_rel_path = os.path.join(
                            self.channel,
                            mode,
                            spec_name
                        )

                    if feature_type == 'raw':
                        item = {
                            'prompt': features['raw_feature'],
                            'completion': label_name,
                            'source': source,
                            'file': os.path.basename(file_path)
                        }
                    else:
                        report = (
                            self.compose_report_ICL(features, template)
                            if 'ICL' in feature_type
                            else self.compose_report(features, template)
                        )
                
                        item = {
                            'prompt': report,
                            'completion': label_name,
                            'source': source,
                            'file': os.path.basename(file_path)
                        }
                
                        if spec_rel_path is not None:
                            item['spectrogram'] = spec_rel_path
                
                    output.append(item)
                #out_file = os.path.join(self.save_path, f'{output_prefix}_{mode}_{label_name}_{feature_type}.json')
                out_file = os.path.join(self.save_path, f'{output_prefix}_{self.channel}_{mode}_{label_name}_{feature_type}.json')
                with open(out_file, 'w') as f:
                    json.dump(output, f, indent=4)
    def compose_report(self, features, template):
        for key, value in features.items():
            if isinstance(value, (int, float, np.floating)):
                rep = f"{float(value):.4e}"
            else:
                rep = str(value)
            template = template.replace(key, rep)
        return template

    def compose_report_ICL(self, features, template):
        for key, value in features.items():
            if isinstance(value, (int, float, np.floating)):
                rep = f"{float(value):.4e}"
            else:
                rep = str(value)
            template = template.replace(key, rep)
        return template

    def train_test_split(self, data, labels, test_size=0.2):
        data, labels = np.array(data), np.array(labels)
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        split_index = int(len(data) * (1 - test_size))
        train_indices, test_indices = indices[:split_index], indices[split_index:]
        return data[train_indices], labels[train_indices], data[test_indices], labels[test_indices]


    def merge_json_outputs(self, prefix='output', feature_types=None):
        feature_types = feature_types or list(FEATURE_FUNCTIONS.keys())
        for ft in feature_types:
            merged_train = []
            merged_test = []
            for label in self.target_labels:
                train_path = os.path.join(self.save_path, f'{prefix}_{self.channel}_train_{label}_{ft}.json')
                test_path = os.path.join(self.save_path, f'{prefix}_{self.channel}_test_{label}_{ft}.json')

                if os.path.exists(train_path):
                    with open(train_path, 'r') as f:
                        merged_train.extend(json.load(f))
                    os.remove(train_path)
                if os.path.exists(test_path):
                    with open(test_path, 'r') as f:
                        merged_test.extend(json.load(f))
                    os.remove(test_path)

            train_filename = _remove_trailing_icl(f'{prefix}_{self.channel}_train_{ft}.json')
            test_filename  = _remove_trailing_icl(f'{prefix}_{self.channel}_test_{ft}.json')
    
            with open(os.path.join(self.save_path, train_filename), 'w') as f:
                json.dump(merged_train, f, indent=4)
            with open(os.path.join(self.save_path, test_filename), 'w') as f:
                json.dump(merged_test, f, indent=4)
    
            logging.info(f"Merged JSON saved for feature type: {ft}")


    def run(self, feature_types=None):
        feature_types = feature_types or list(FEATURE_FUNCTIONS.keys())

        artifact_root = os.path.join(self.input_path, 'artifact')
        seizure_root = os.path.join(self.input_path, 'seizure')

        artifact_subfolders = [
            os.path.join('artifact', d)
            for d in os.listdir(artifact_root)
            if os.path.isdir(os.path.join(artifact_root, d))
        ] if os.path.exists(artifact_root) else []

        seizure_subfolders = [
            os.path.join('seizure', d)
            for d in os.listdir(seizure_root)
            if os.path.isdir(os.path.join(seizure_root, d))
        ] if os.path.exists(seizure_root) else []

        folder_groups = {
            'artifact': artifact_subfolders,
            'non_artifact': ['non_artifact'],
            'sz': seizure_subfolders
        }

        logging.info(f"Detected folders: {folder_groups}")

        for label, file_count in zip(self.target_labels, self.target_files_list):
            if label in folder_groups:
                self.extract_multiple_feature_types(
                    label_name=label,
                    subfolders=folder_groups[label],
                    feature_types=feature_types,
                    output_prefix='output',
                    file_count=file_count
            )

        self.merge_json_outputs(prefix='output', feature_types=feature_types)
        logging.info("Preprocessing completed!")

def main():
    base_input_path = '/Group16T/common/cyj/code/channel_tuar'
    save_path = '/Group16T/common/cyj/code/tuar_stft_spec_prompt'
    priors = load_channel_priors("channel_summary.csv")

    exclude_channels = {}

    channel_dirs = [
        d for d in os.listdir(base_input_path)
        if os.path.isdir(os.path.join(base_input_path, d)) and d not in exclude_channels
    ]

    selected_features = ['stft_ICL']
    target_labels = ['artifact', 'non_artifact']
    target_files_list = [20000, 20000]

    for channel in channel_dirs:
        print(f"Processing channel: {channel}")
        preprocessor = EEGPreprocessor(
            input_path=os.path.join(base_input_path, channel),
            save_path=save_path,
            feature_type='raw',
            target_files_list=target_files_list,
            target_labels=target_labels,
            channel = channel,
            channel_priors=priors 
        )
        preprocessor.run(feature_types=selected_features)

if __name__ == '__main__':
    main()
