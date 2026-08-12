import os
from mne.io import read_raw_edf
from mne import set_log_level
import mne
import numpy as np

set_log_level("WARNING")

input_dir = "/Group16T/raw_data/cgmh/Labeled_EEG_2025/Artifacts/EDF檔/"
output_dir = "/Group16T/common/cyj/preprocess_data_1/cgmh_people_artifact/fif"

for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(".edf"):
            edf_path = os.path.join(root, file)
            try:
                raw = read_raw_edf(edf_path, preload=True)

                relative_path = os.path.relpath(root, input_dir)
                relative_path = relative_path.replace(" ", "_")
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                base_filename = os.path.splitext(file)[0].replace(" ", "_")
                fif_filename = base_filename + ".fif"
                fif_path = os.path.join(output_subdir, fif_filename)

                raw.save(fif_path, overwrite=True)
                print(f"success：{edf_path} -> {fif_path}")
            except Exception as e:
                print(f"fail：{edf_path}")
                print(f"Error：{e}")


channel_pairs = [
    ("EEG Fp1-AVE", "EEG F7-AVE"),
    ("EEG F7-AVE", "EEG T3-AVE"),
    ("EEG T3-AVE", "EEG T5-AVE"),
    ("EEG T5-AVE", "EEG O1-AVE"),
    ("EEG Fp2-AVE", "EEG F8-AVE"),
    ("EEG F8-AVE", "EEG T4-AVE"),
    ("EEG T4-AVE", "EEG T6-AVE"),
    ("EEG T6-AVE", "EEG O2-AVE"),
    ("EEG T3-AVE", "EEG C3-AVE"),
    ("EEG C3-AVE", "EEG Cz-AVE"),
    ("EEG Cz-AVE", "EEG C4-AVE"),
    ("EEG C4-AVE", "EEG T4-AVE"),
    ("EEG Fp1-AVE", "EEG F3-AVE"),
    ("EEG F3-AVE", "EEG C3-AVE"),
    ("EEG C3-AVE", "EEG P3-AVE"),
    ("EEG P3-AVE", "EEG O1-AVE"),
    ("EEG Fp2-AVE", "EEG F4-AVE"),
    ("EEG F4-AVE", "EEG C4-AVE"),
    ("EEG C4-AVE", "EEG P4-AVE"),
    ("EEG P4-AVE", "EEG O2-AVE")
]

def process_and_save_fif(input_file, input_root, output_root):
    try:
        raw = mne.io.read_raw_fif(input_file, preload=True)

        new_data = []
        new_ch_names = []

        for (ch1, ch2) in channel_pairs:
            if ch1 in raw.ch_names and ch2 in raw.ch_names:
                idx1 = raw.ch_names.index(ch1)
                idx2 = raw.ch_names.index(ch2)
                diff = raw.get_data(picks=idx1) - raw.get_data(picks=idx2)
                new_data.append(diff.squeeze())

                name1 = ch1.split()[1].split("-")[0]
                name2 = ch2.split()[1].split("-")[0]
                new_ch_names.append(f"{name1}-{name2}")

        if len(new_data) == 0:
            print(f"No channel，skip：{input_file}")
            return

        new_data = np.array(new_data)
        info = mne.create_info(ch_names=new_ch_names, sfreq=raw.info['sfreq'], ch_types='eeg')
        new_raw = mne.io.RawArray(new_data, info)
        new_raw.set_meas_date(raw.info['meas_date']) 
        new_raw.set_annotations(raw.annotations)

        relative_path = os.path.relpath(input_file, input_root).replace(" ", "_")
        output_file = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        new_raw.save(output_file, overwrite=True)
        print(f"save：{output_file}")
    except Exception as e:
        print(f"fail：{input_file}")
        print(f"Error：{e}")

input_root = "/Group16T/common/cyj/preprocess_data_1/cgmh_people_artifact/fif"
output_root = "/Group16T/common/cyj/preprocess_data_1/cgmh_people_artifact/TCP"
os.makedirs(output_root, exist_ok=True)

for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith(".fif"):
            input_path = os.path.join(root, file)
            process_and_save_fif(input_path, input_root, output_root)

new_sfreq = 100
l_freq = 0.5 
h_freq = 35 

clipping_threshold = 800  # µV

def preprocess_fif(input_file, input_directory, output_directory):
    raw = mne.io.read_raw_fif(input_file, preload=True)

    raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir',
               iir_params=dict(order=4, ftype='butter'))

    raw.resample(sfreq=new_sfreq)

    data = raw.get_data()
    data = np.clip(data, -clipping_threshold, clipping_threshold)
    raw._data = data

    relative_path = os.path.relpath(input_file, input_directory)
    output_file = os.path.join(output_directory, relative_path)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    raw.save(output_file, overwrite=True)
    print(f"save：{output_file}")

input_directory = "/Group16T/common/cyj/preprocess_data_1/cgmh_people_artifact/TCP/Artifacts/EDF檔"
output_directory = "/Group16T/common/cyj/preprocess_data_1/cgmh_people_artifact/preprocessing"
os.makedirs(output_directory, exist_ok=True)

for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.endswith(".fif"):
            input_file = os.path.join(root, file)
            preprocess_fif(input_file, input_directory, output_directory)


input_directory = '/Group16T/common/cyj/preprocess_data_1/cgmh_people_artifact/preprocessing/'
output_root = '/Group16T/common/cyj/preprocess_data_1/cgmh_people_artifact/segment/'

label_mapping = {
    'Arti BLINK': 'eye',
    'Arti EYEM': 'eye',
    'Arti MUSC': 'muscle',
    'Arti M Burst': 'muscle',
    'Arti ELEC': 'electrode',
    'Arti CHEW': 'chewing',
}

segment_duration = 2
overlap_duration = 1
sfreq = 100
segment_samples = int(segment_duration * sfreq)
step_samples = int((segment_duration - overlap_duration) * sfreq)

total_warning_segments = 0

for root, _, files in os.walk(input_directory):
    for file in files:
        if file.endswith('.fif'):
            fif_file_path = os.path.join(root, file)
            raw = mne.io.read_raw_fif(fif_file_path, preload=True)
            sfreq = raw.info['sfreq']
            segment_samples = int(segment_duration * sfreq)
            step_samples = int((segment_duration - overlap_duration) * sfreq)

            annotations = raw.annotations
            time_intervals = [
                (a['onset'], a['onset'] + a['duration'], label_mapping.get(a['description'], None))
                for a in annotations if a['description'] in label_mapping
            ]

            print(f"{fif_file_path}")
            for start, end, desc in time_intervals:
                print(f"{desc}, start: {start:.2f}s, end: {end:.2f}s")

            subfolder = os.path.relpath(root, input_directory)
            base_name = os.path.splitext(file)[0]

            n_samples = raw.n_times
            start_sample = 0
            counts = {'eye': 0, 'muscle': 0,'electrode': 0, 'chewing': 0, 'background': 0}
            file_warning_segments = 0

            while start_sample + segment_samples <= n_samples:
                end_sample = start_sample + segment_samples
                start_time = start_sample / sfreq
                end_time = end_sample / sfreq

                categories = set()
                for ann_start, ann_end, category in time_intervals:
                    if category and ann_start <= start_time < ann_end and ann_start < end_time <= ann_end:
                        categories.add(category)

                if len(categories) > 1:
                    file_warning_segments += 1
                    print(f"skip {start_time:.2f}s–{end_time:.2f}s multi-artifact: {categories}")
                    start_sample += step_samples
                    continue

                segment = raw[:, start_sample:end_sample][0]

                if len(categories) == 1:
                    category = list(categories)[0]
                else:
                    is_clean = all(
                        not (
                            (start <= start_time < end) or 
                            (start < end_time <= end) or 
                            (start_time <= start and end <= end_time)
                        )
                        for start, end, _ in time_intervals
                    )
                    if is_clean:
                        category = 'background'
                    else:
                        start_sample += step_samples
                        continue

                save_dir = os.path.join(output_root, subfolder, category)
                os.makedirs(save_dir, exist_ok=True)
                count = counts[category]
                output_file = os.path.join(save_dir, f"{base_name}_{count:03d}.npy")
                np.save(output_file, segment)
                counts[category] += 1
                print(f"[{category}] save: {output_file}")

                start_sample += step_samples

            print(f"{file} finish：skip {file_warning_segments} segments，Count of each category: {counts}\n")
            total_warning_segments += file_warning_segments

print(f"All file finish，total skip {total_warning_segments} segments（multi-artifact）")