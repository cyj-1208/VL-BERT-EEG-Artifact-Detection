# Channel-Wise Vision-Language BERT Framework for EEG Artifact Detection
Construct the Channel-Wise VL-BERT-based EEG artifact detection framework that integrates textual and visual EEG representations through a multimodal feature fusion strategy to enhance artifact recognition. 

## Pipeline
This Pipeline implements a Channel-Wise Vision-Language BERT framework for EEG artifact detection using multimodal signal representations. Multichannel EEG recordings from the TUAR and CGMH datasets are preprocessed and segmented into fixed-length EEG segments. For each channel, EEG signal features (DWT, STFT, band-pass filtering, and descriptive statistics) are used as textual representations, while spectrogram features extracted by a ResNet-18-based CNN serve as visual representations. These multimodal features are fused within a Vision-Language BERT model to perform channel-level artifact detection. For the CGMH dataset, channel-level predictions are further combined using an OR-gate to obtain segment-level predictions.

![Pipeline](workflow.png)
## Prerequisites
* Hardware：NVIDIA RTX 4090 24G GPU
* Environment：PyTorch 2.5.1、Hugging Face Transformers 4.51.3、Python 3.11.10、Ubuntu 24.04.1 LTS
## Datasets

## Preprocessing

## Prompt engineering
Continuous EEG recordings are segmented into 2-second windows with a 1-second overlap. Signal features are then extracted from each segment to generate a textual prompt, while an STFT spectrogram is simultaneously generated as the visual input. See `prompt/prompt_example.txt` for a complete prompt example.
```
# Generate textual prompts and STFT spectrograms for the TUAR dataset
python ./prompt/prompt_tuar.py

# Generate textual prompts and STFT spectrograms for the CGMH dataset
python ./prompt/prompt_cgmh.py
```
## EEG artifact detection model
Textual and visual features extracted from each EEG channel are transformed into text tokens and visual tokens, respectively, and concatenated into a multimodal input sequence. The sequence is then encoded by a Vision-Language BERT framework using BERT, DistilBERT, RoBERTa as the language encoder for channel-level EEG artifact classification.
```
# training & internal testing & external testing
python workflow.py --feature all --model all --repeat 5
```

