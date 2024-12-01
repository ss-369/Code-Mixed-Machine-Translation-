# CodeMix Translation Project

This project explores machine translation between English and Hinglish (Hindi written in the Roman script). The primary objective is to build, fine-tune, and compare models for translating English text to Hinglish, leveraging both pre-trained models and custom architectures.

---

## Directory Structure

```md
  .
  ├── README.md
  ├── Report.pdf
  ├── final-ppt.ppt
  ├── codes/
  │   ├── bart-base.py
  │   ├── custom-transformer.py
  │   ├── mbart-large-50-many-to-many-mmt.py
  │   ├── opus-mt-en-ROMANCE.py
  │   └── t5-small.py
  ├── notebooks/
  │   ├── bart-base.ipynb
  │   ├── mbart-large-50-many-to-many-mmt.ipynb
  │   ├── opus-mt-en-ROMANCE.ipynb
  │   └── t5-small.ipynb

```

### Descriptions
- **README.md**: Main documentation for the project.
- **codes/**: Contains Python scripts for model implementations.
- **notebooks/**: Jupyter notebooks for training and evaluation of respective models.

---

## Models Included

### 1. **T5 Small**
- **Dataset**: [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish)
- **Training Setup**:
  - The model was trained on 189,102 sentences using a T5-small architecture.
  - Training took 5 hours on 2 Tesla T4 GPUs.
- **Evaluation**:
  - **BLEU Score on Test Data**: 26.98
- **Description**:
  - This model demonstrates fine-tuning of a pre-trained `T5-small` architecture on the English-to-Hinglish dataset. The dataset contains English sentences paired with their Hinglish translations in the Roman script.

---

### 2. **Custom Transformer Encoder-Decoder Model**
- **Dataset**: [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish)
- **Training Setup**:
  - Utilized a custom Transformer-based Encoder-Decoder architecture for English-to-Hinglish translation (Devanagari script).
  - Ran for 10 epochs on 2 Tesla T4 GPUs.
- **Evaluation**:
  - **BLEU Score on Test Data**: 20.58
- **Description**:
  - This model was specifically designed to work with Hinglish translations written in the Devanagari script, exploring the challenges of alternate script systems.

---

### 3. **BART-Base**
- **Dataset**: [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish)
- **Training Setup**:
  - Fine-tuned on the full English-to-Hinglish dataset using lora .
  - Training took 5 hours on 2 Tesla T4 GPUs.
- **Evaluation**:
  - **BLEU Score on Test Data**: 33.06
- **Description**:
  - This model fine-tunes the BART-base architecture for machine translation tasks, leveraging its seq2seq capabilities.

---

### 4. **mBART-Large-50**
- **Dataset**: [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish)
- **Training Setup**:
  - Fine-tuned on the full dataset for 3 hours using 2 Tesla T4 GPUs.
- **Evaluation**:
  - **BLEU Score on Test Data**: 43.23
- **Description**:
  - The `mBART-large-50` model leverages multilingual pre-training and fine-tuning, showing improved performance due to its multilingual capabilities.

---

### 5. **Helsinki-NLP/opus-mt-en-ROMANCE**
- **Dataset**: [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish)
- **Training Setup**:
  - Fine-tuned on the full dataset for 3.5 hours using a 2 Tesla T4 GPUs using lora.
- **Evaluation**:
  - **BLEU Score on Test Data**: 32.22
- **Description**:
  - This model fine-tunes the Helsinki-NLP model specialized for translating English to Romance languages. It serves as a baseline comparison for models pre-trained on related language pairs.

---

## Dataset Information

### Roman Script Dataset
- **Source**: [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish)
- **Description**:
  - Contains English-to-Hinglish sentence pairs where Hinglish is written in the Roman script.

---

## Results Summary

| Model                             | Dataset                                              | Epochs | BLEU Score | Training Time | GPUs       |
|-----------------------------------|------------------------------------------------------|--------|------------|---------------|------------|
| **T5 Small**                      | [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish) | 5      | 26.98      | 5 hours       | 2x Tesla T4 |
| **T5 Small (30% Dataset)**        | [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish) | 3      | 14.16      | 45 minutes    | 2x Tesla T4 |
| **Custom Transformer**            | [English-to-Hinglish (Devanagari Script)](https://drive.google.com/drive/folders/1knd9VI6hU499KR8fx3AyiKEinoTB05Nz) | 10     | 7.98       | 2 hours       | P100        |
| **Custom Transformer**            | [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish) | 10     | 20.58      | 3 hours       | P100        |
| **BART-Base**                     | [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish) | 4      | 33.06      | 4 hours       | 2x Tesla T4 |
| **mBART-Large-50**                | [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish) | 3      | 43.23      | 3.36 hours       | 2x Tesla T4 |
| **Helsinki-NLP/opus-mt-en-ROMANCE** | [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish) | 5      | 32.22      | 4 hours     | P100        |

---

## Usage

To explore or use the models:
1. **T5 Models**:
   - Fine-tuning and evaluation are detailed in the `T5 Small` notebooks.
2. **Custom Transformer**:
   - Contains implementation details for the custom Transformer architecture.
3. **BART/mBART/Helsinki-NLP**:
   - Includes pre-trained multilingual setups and fine-tuning instructions.

**Run the notebooks on a GPU environment** for better performance and to reproduce the results.

---

## Resources
- **Saved Models**: [Models Links](https://iiitaphyd-my.sharepoint.com/personal/shivashankar_gande_students_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fshivashankar%5Fgande%5Fstudents%5Fiiit%5Fac%5Fin%2FDocuments%2Ffiles%2FANLP%2Dinterim)  
- **Datasets**:
  - [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish)


For further insights, refer to the detailed project report included in the Directory.