import os
import mne
from datetime import datetime, timezone
import numpy as np
import pandas as pd

source_directory = "/path/to/dataset"

target_directory = "/path/to/dataset"

os.makedirs(target_directory, exist_ok=True)

current_local_time = datetime.now()

current_utc_time = current_local_time.astimezone(timezone.utc)

for root, dirs, files in os.walk(source_directory):
    for file in files:
        if file.endswith('.edf'):
            file_path = os.path.join(root, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            raw.set_meas_date(None)

            relative_path = os.path.relpath(root, source_directory)
            target_path = os.path.join(target_directory, relative_path)
            os.makedirs(target_path, exist_ok=True)
            target_file_path = os.path.join(target_path, file.replace('.edf', '.fif'))

            raw.save(target_file_path, overwrite=True)
            
print("finish fif")

channel_pairs = [
    ("EEG FP1-REF", "EEG F7-REF"),
    ("EEG F7-REF", "EEG T3-REF"),
    ("EEG T3-REF", "EEG T5-REF"),
    ("EEG T5-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F8-REF"),
    ("EEG F8-REF", "EEG T4-REF"),
    ("EEG T4-REF", "EEG T6-REF"),
    ("EEG T6-REF", "EEG O2-REF"),
    ("EEG A1-REF", "EEG T3-REF"),  ## cgmh artifact people沒有的通道
    ("EEG T3-REF", "EEG C3-REF"),
    ("EEG C3-REF", "EEG CZ-REF"),
    ("EEG CZ-REF", "EEG C4-REF"),
    ("EEG C4-REF", "EEG T4-REF"),
    ("EEG T4-REF", "EEG A2-REF"),  ## cgmh artifact people沒有的通道
    ("EEG FP1-REF", "EEG F3-REF"),
    ("EEG F3-REF", "EEG C3-REF"),
    ("EEG C3-REF", "EEG P3-REF"),
    ("EEG P3-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F4-REF"),
    ("EEG F4-REF", "EEG C4-REF"),
    ("EEG C4-REF", "EEG P4-REF"),
    ("EEG P4-REF", "EEG O2-REF")
]

def process_and_save_fif(input_file, output_directory):
    raw = mne.io.read_raw_fif(input_file, preload=True)

    new_data = []
    new_ch_names = []

    for (ch1, ch2) in channel_pairs:
        if ch1 in raw.ch_names and ch2 in raw.ch_names:
            idx1 = raw.ch_names.index(ch1)
            idx2 = raw.ch_names.index(ch2)
            diff_data = raw.get_data(picks=idx1) - raw.get_data(picks=idx2)
            new_data.append(diff_data.squeeze())

            new_ch_name = f"{ch1.split()[1].split('-')[0]}-{ch2.split()[1].split('-')[0]}"
            new_ch_names.append(new_ch_name)

    new_data = np.array(new_data)

    info = mne.create_info(ch_names=new_ch_names, sfreq=raw.info['sfreq'], ch_types='eeg')
    new_raw = mne.io.RawArray(new_data, info)

    filename = os.path.basename(input_file)
    output_file = os.path.join(output_directory, filename.replace('.fif', '_TCP.fif'))

    new_raw.save(output_file, overwrite=True)
    print(f"save to {output_file}")

input_directory = "/Group16T/common/cyj/TUAR_multi_channel/fif"
output_directory = "/Group16T/common/cyj/TUAR_multi_channel/TCP"
os.makedirs(output_directory, exist_ok=True)

for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.endswith('.fif'):
            input_file = os.path.join(root, file)
            process_and_save_fif(input_file, output_directory)


l_freq = 0.5
h_freq = 35 

new_sfreq = 100  

clipping_threshold = 800  # µV

def preprocess_fif(input_file, output_directory):
    raw = mne.io.read_raw_fif(input_file, preload=True)

    raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=dict(order=4, ftype='butter'))

    raw.resample(sfreq=new_sfreq)

    data = raw.get_data()
    data = np.clip(data, -clipping_threshold, clipping_threshold)
    raw._data = data

    filename = os.path.basename(input_file)
    output_file = os.path.join(output_directory, filename)

    raw.save(output_file, overwrite=True)
    print(f"save to {output_file}")

input_directory = "/Group16T/common/cyj/TUAR_multi_channel/TCP"
output_directory = "/Group16T/common/cyj/TUAR_multi_channel/preprocessing"
os.makedirs(output_directory, exist_ok=True)

for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.endswith('.fif'):
            input_file = os.path.join(root, file)
            preprocess_fif(input_file, output_directory)


input_directory = '/Group16T/common/cyj/TUAR_multi_channel/preprocessing'
csv_directory = '/Group16T/raw_data/tuh_eeg/artifact/edf/01_tcp_ar'
artifact_directory = '/Group16T/common/cyj/TUAR_multi_channel/artifact'
non_artifact_directory = '/Group16T/common/cyj/TUAR_multi_channel/non_artifact'

os.makedirs(artifact_directory, exist_ok=True)
os.makedirs(non_artifact_directory, exist_ok=True)

segment_duration = 2
overlap_duration = 1
sfreq = 100
segment_samples = int(segment_duration * sfreq)
overlap_samples = int(overlap_duration * sfreq) 
step_samples = segment_samples - overlap_samples 

for file in os.listdir(input_directory):
    if file.endswith('.fif'):
        fif_file_path = os.path.join(input_directory, file)

        base_name = os.path.basename(fif_file_path).replace('_TCP.fif', '.csv')
        csv_file_path = os.path.join(csv_directory, base_name)

        raw = mne.io.read_raw_fif(fif_file_path, preload=True)

        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path, skiprows=6)

            col2_values = df.iloc[:, 1].values
            col3_values = df.iloc[:, 2].values

            time_intervals = list(set(zip(col2_values, col3_values)))
            
            print(f"{file}")
            print("artifact time segment: ", time_intervals)
        else:
            print(f"CSV file {csv_file_path} not exist")
            time_intervals = []

        n_samples = raw.n_times

        artifact_count = 0
        non_artifact_count = 0

        start_sample = 0
        while start_sample + segment_samples <= n_samples:
            end_sample = start_sample + segment_samples

            segment = raw[:, start_sample:end_sample][0]

            start_time = start_sample / sfreq
            end_time = end_sample / sfreq

            is_artifact = any(start <= start_time < end and start < end_time <= end for start, end in time_intervals)
            is_non_artifact = all(end_time <= start or start_time >= end for start, end in time_intervals)

            if is_artifact:
                artifact_output_file = os.path.join(
                    artifact_directory, f"{base_name.split('.')[0]}_artifact_{artifact_count:03d}.npy"
                )
                np.save(artifact_output_file, segment)
                artifact_count += 1
                print(f"Artifact segment save to {artifact_output_file}")
            elif is_non_artifact:
                non_artifact_output_file = os.path.join(
                    non_artifact_directory, f"{base_name.split('.')[0]}_non_artifact_{non_artifact_count:03d}.npy"
                )
                np.save(non_artifact_output_file, segment)
                non_artifact_count += 1
                print(f"Non-artifact save to {non_artifact_output_file}")

            start_sample += step_samples

        print(f"{file} finish，save {artifact_count} artifact segment and {non_artifact_count} non-artifact segment")
