
---

# CodeMix Translation Project

This project explores machine translation between English and Hinglish (Hindi written in the Roman script). The primary objective is to build and compare models for translating English text to Hinglish, leveraging pre-trained models and custom architectures.

## Models Included
### 1. **T5 Small** 
- **Dataset**: [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish)
- **Training Setup**:
  - The model was trained on 189,102 sentences using a T5-small architecture.
  - It ran for 5 hours using 2 Tesla T4 GPUs.
- **Evaluation**:
  - **Bleu Score on Test Data**: 26.98
- **Description**:
  - This notebook demonstrates the fine-tuning of a pre-trained `T5-small` model on the English-to-Hinglish dataset. The dataset consists of sentence pairs where the target text is in Hinglish.
  
  

### 2. **T5 Small(30% Dataset)** 
- **Dataset**: [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish)
- **Training Setup**:
  - The model was trained on 30% of original dataset using a T5-small architecture.
  - It ran for 45 minutes using 2 Tesla T4 GPUs.
- **Evaluation**:
  - **Bleu Score on Test Data**: 14.16
- **Description**:
  - This notebook demonstrates the fine-tuning of a pre-trained `T5-small` model on the English-to-Hinglish dataset. The dataset consists of sentence pairs where the target text is in Hinglish.
  
  

### 3. **Custom Transformer Encoder-Decoder Model**
- **Dataset**: [English-to-Hinglish (Devanagari Script)](https://drive.google.com/drive/folders/1knd9VI6hU499KR8fx3AyiKEinoTB05Nz)
- **Training Setup**:
  - The model uses a Transformer-based Encoder-Decoder architecture specifically designed to translate English to Hinglish written in the Devanagari script.
  - It ran for 10 epochs using 2 Tesla T4 GPUs.
- **Evaluation**:
  - **Bleu Score on Test Data**: 7.98
- **Description**:
  - This notebook showcases the implementation of a custom Transformer architecture for the translation task. The dataset used for this model contains Hinglish translations written in Devanagari script.

  

## Dataset Information
- **T5 Small Model**:
  - Dataset: [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish)
  - This dataset contains English to Hinglish translations where Hinglish is written in the Roman script.
  
- **Custom Transformer Model**:
  - Dataset: English-to-Hinglish (Devanagari Script)
  - This dataset contains English to Hinglish translations where Hinglish is written in the Devanagari script.


## Results Summary
| Model                  | Dataset                                              | Epochs | BLEU Score | Training Time | GPUs       |
|------------------------|------------------------------------------------------|--------|------------|---------------|------------|
| **T5 Small**            | [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish) | 5      | 26.98      | 5 hours       | 2x Tesla T4|
| **T5 Small Partial(30 %)**            | [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish) | 3      | 14.16      | 45 minutes       | 2x Tesla T4|
| **Custom Transformer**  | [English-to-Hinglish (Devanagari Script)](https://drive.google.com/drive/folders/1knd9VI6hU499KR8fx3AyiKEinoTB05Nz) | 10     | 7.98       | 2 hours           | P100 |

## Usage
To use the models or explore the training process, open the respective Jupyter notebooks:
- [T5 Small Notebook]
- [Custom Transformer Notebook]

You can run the notebooks on a GPU environment to reproduce the results or fine-tune the models further.
And also for further understanding kindly go through report.

One drive link for saved models ->[ Models Links](https://iiitaphyd-my.sharepoint.com/personal/shivashankar_gande_students_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fshivashankar%5Fgande%5Fstudents%5Fiiit%5Fac%5Fin%2FDocuments%2Ffiles%2FANLP%2Dinterim)  
---

