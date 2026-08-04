# Channel-Wise Vision-Language BERT Framework for EEG Artifact Detection
Construct the Channel-Wise VL-BERT-based EEG artifact detection framework that integrates textual and visual EEG representations through a multimodal feature fusion strategy to enhance artifact recognition. 

## Pipeline
This Pipeline implements a Channel-Wise Vision-Language BERT framework for EEG artifact detection using multimodal signal representations. Multichannel EEG recordings from the TUAR and CGMH datasets are preprocessed and segmented into fixed-length EEG segments. For each channel, EEG signal features (DWT, STFT, band-pass filtering, and descriptive statistics) are used as textual representations, while spectrogram features extracted by a ResNet-18-based CNN serve as visual representations. These multimodal features are fused within a Vision-Language BERT model to perform channel-level artifact detection. For the CGMH dataset, channel-level predictions are further combined using an OR-gate to obtain segment-level predictions.

![Pipeline](workflow.png)
