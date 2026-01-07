# A Holistic Approach to Khmer Optical Character Recognition Using an Efficient Hybrid CNN-Transformer Model

This repository contains the implementation, dataset generation scripts, and evaluation results for a high-performance Khmer Optical Character Recognition (OCR) system. The model utilizes a hybrid architecture combining Convolutional Neural Networks (CNN) for feature extraction and Transformers for sequence modeling, specifically designed to handle the complexity and length of Khmer script.

## Project Overview

Khmer script presents unique challenges for OCR due to its large character set, complex stacking, and variable text line lengths. This project proposes an efficient pipeline that:
1.  **Chunks** long text lines into manageable segments.
2.  **Extracts features** using a modified VGG (or ResNet) backbone.
3.  **Encodes** spatial features using a Transformer Encoder.
4.  **Merges** context across chunks.
5.  **Decodes** the final sequence using a Transformer Decoder.

## Datasets

The model was trained entirely on synthetic data and evaluated on real-world datasets.

### Training Data (Synthetic)
We generated **200,000 synthetic images** to ensure robustness against font variations and background noise.

| Dataset Type | Count | Generator / Source | Augmentations |
| :--- | :--- | :--- | :--- |
| **Document Text** | 100,000 | Pillow + Khmer Corpus | Erosion, noise, thinning/thickening, perspective distortion. |
| **Scene Text** | 100,000 | SynthTIGER + Stanford BG | Rotation, blur, noise, realistic backgrounds. |

### Evaluation Data (Real-World)
| Dataset | Type | Size | Description |
| :--- | :--- | :--- | :--- |
| **KHOB** | Real | 325 | Standard benchmark, clean backgrounds but compression artifacts. |
| **Legal Documents** | Real | 227 | High variation in degradation, illumination, and distortion. |
| **Printed Words** | Synthetic | 1,000 | Short, isolated words in 10 different fonts. |

![Dataset Overview](./assets/dataset-overview.png)
---

## Methodology & Architecture

### 1. Preprocessing: Chunking & Merging
To handle variable-length text lines without aggressive resizing:
*   **Padding:** Images are padded to be divisible by 100 width.
*   **Chunking:** Split into overlapping chunks (Size: 48x100 px, Overlap: 16 px).
*   **Merging:** Features are processed per chunk and merged before the decoding stage to preserve context.

### 2. Model Architecture
The model consists of five key modules:
1.  **CNN Feature Extractor:** (Modified VGG or ResNet) Outputs feature maps (512 channels, 2x32 size).
2.  **Patch Encoder:** Projects spatial features into 384-dim embedding space.
3.  **Transformer Encoder:** Captures contextual relationships among visual tokens.
4.  **Merging Module:** Aggregates features from all chunks of a single text line.
5.  **Transformer Decoder:** Generates the final Khmer character sequence.

![Model Architecture](./assets/Model-Architecture.png)

---

## Training Configuration

*   **Epochs:** 100
*   **Optimizer:** Adam
*   **Loss Function:** Cross-Entropy Loss
*   **Learning Rate Schedule:** Staged Cyclic
    *   *Epoch 0-15:* Fixed 1e-4 (Rapid convergence)
    *   *Epoch 16-30:* Cyclic 1e-4 to 1e-5 (Stability)
    *   *Epoch 31-100:* Cyclic 1e-5 to 1e-6 (Fine-tuning)
*   **Sampling:** 50,000 images randomly sampled/augmented per epoch.

---

## Quantitative Analysis

We benchmarked our **VGG-Transformer** and **ResNet-Transformer** models against Tesseract-OCR (v4/v5).

**Character Error Rate (CER %)** - *Lower is better*

TABLE 1: Character Error Rate (CER in %) results on the KHOB, Legal Documents, and Printed Word

| Model | KHOB | Legal Documents | Printed Word |
| :--- | :--- | :--- | :--- |
| Tesseract-OCR | 9.36% | 24.30% | 8.02% |
| VGG-Transformer | **5.07%** | **10.27%** | 3.61% |
| ResNet-Transformer | 5.85% | 11.57% | **2.80%** |

---

## Qualitative Analysis

TABLE 2: Failure cases of CNN-Transformer vs Tesseract OCR on KHOB, Legal Documents, and Printed Word

| Instance | Ground-Truth | VGG-Transformer | ResNet-Transformer | Tesseract-OCR |
| :---: | :--- | :--- | :--- | :--- |
| <img src="./assets/f_case_1.png" width="200"> | អគ្គលេខាធិការ<br>ដ្ឋានគណៈកម្មាធិ<br>ការដឹកនាំ | អគ្គរលេខ<span style="color:red">ខះ</span>ិការដ្ឋ<br>នគណៈកម្មា<span style="color:red">ឌិ</span>ការ<br><span style="color:red">ល</span>ដឹកនាំ | <span style="color:red">ភ្ភ្គរ</span>លេខ<span style="color:red">ះ</span>ធិការដ្ឋ<br>ន<span style="color:red">#</span>ណៈកម្មជិការ<br>ដឹកនាំ | .«<span style="color:red">ក្កលរេទានិការដ្ឋទកណៈ<br>កម្មាទិការដ្ឹកទាាំ</span> |
| <img src="./assets/f_case_2.png" width="200"> | និងកែសម្រួល<br>សមាសភាពរាជ<br>រដ្ឋាភិបាលនៃព្រះ<br>រាជាណាចក្រកម្ពុ<br>ជា | និងកែសម្រួល<br>សមាសភាពរាជ<br>រដ្ឋាភិបាល<span style="color:red">នៃ</span>ព្រះ<br>រាជាណាចក្រកម្ពុ<br>ជា | <span style="color:red">1664000 ប្រយុ<br>រសម្ព័ក៌មានស<br>ង្គមាត់០០០០ វិបែង<br>ស៊្យ័យ(4បុណ្ណ</span> | និងកែសម្រួលសមាស<br>ភាពរាជ<span style="color:red">ន្នា</span>ភិបាលនៃ<span style="color:red">ទ្រះ</span><br>រាជាណាចក្រកម្ពុជា |
| <img src="./assets/f_case_3.png" width="200"> | ឧបនាយករដ្ឋម<br>ន្ត្រី | <span style="color:red">ខួ</span>បនាយករដ្ឋម<span style="color:red">ន្ត្រី</span> | ឧបនាយករដ្ឋម<br><span style="color:red">ន្ត្រកី</span> | <span style="color:red">ទូ</span>បនាយករដ្ឋម<br><span style="color:red">ន្ត្រកី</span> |
| <img src="./assets/f_case_4.png" width="200"> | 180818125 | 1808<span style="color:red">1</span>8125 | 18<span style="color:red">0</span>818125 | 180818125 |

TABLE 3: Example of CNN-Transformer vs Tesseract OCR compared with the ground truth. Errors in the predictions are highlighted in red.

| Instance | Ground-Truth | VGG-Transformer | ResNet-Transformer | Tesseract-OCR |
| :---: | :--- | :--- | :--- | :--- |
| <img src="./assets/s_case_1.png" width="200"> | រាជរដ្ឋាភិបាលកម្ពុជា | រាជរដ្ឋាភិបាលកម្ពុជា | រាជរដ្ឋាភិបាលកម្ពុជា | រាជរដ្ឋា<span style="color:red">គិធា</span>លកម្ពុជា |
| <img src="./assets/s_case_2.png" width="200"> | ព្រះរាជាណាចក្រកម្ពុ<br>ជា | ព្រះរាជាណាចក្រកម្ពុជា | ព្រះរាជាណាចក្រកម្ពុជា | ព្រះរាជាណាច<span style="color:red">ត្រ</span>កម្ពុ<br>ជា |
| <img src="./assets/s_case_3.png" width="200"> | រាជរដ្ឋាភិបាលនៃព្រះ<br>រាជាណាចក្រកម្ពុជា | រាជរដ្ឋាភិបាលនៃព្រះ<br>រាជាណាចក្រកម្ពុជា | រាជរដ្ឋាភិបាលនៃព្រះ<br>រាជាណាចក្រកម្ពុជា | រាជរដ្ឋាភិបាលនៃព្រះ<span style="color:red">៖</span><br>រាជាណាចក្រកម្ពុជា |
| <img src="./assets/s_case_4.png" width="200"> | 011048599 | 011048599 | 011048599 | 0110<span style="color:red">H</span>85<span style="color:red">6</span>9<span style="color:red">:</span> |
| <img src="./assets/s_case_5.png" width="200"> | ឈុនហៀង | ឈុនហៀង | ឈុនហៀង | <span style="color:red">_</span>ឈុនហៀង |

**Key Findings:**
*   **VGG-Transformer** performs best on long, complex text lines (Documents).
*   **ResNet-Transformer** performs best on short, isolated words.
*   Both deep learning approaches significantly outperform Tesseract-OCR on degraded legal documents.

### Installation
```bash
git clone https://github.com/yourusername/khmer-ocr-cnn-transformer.git
cd khmer-ocr-cnn-transformer
pip install -r requirements.txt
```
---
## Inference Usage
This pipeline performs end-to-end OCR by first detecting text lines using Surya and then recognizing characters using our custom CNN-Transformer model.

![Inference Pipeline](./assets/inference-pipeline.jpg)

### 1. Prerequisites
You need to install Surya directly from its GitHub repository to ensure you have the latest detection features, or you just clone our repostiory which already contain surya:
```bash
# Install Surya from GitHub
git clone https://github.com/datalab-to/surya.git
```

### 2. How to Run
To run the OCR on an image, simply execute the script:
```bash
python inference.py
```
### 3. How to Change the Input Image
To recognize a different document, open inference.py in your text editor. Scroll to the very bottom of the file (inside the if __name__ == "__main__": block) and change the image_path variable:
```python
# ===================================================
# RUN EXAMPLE
# ===================================================
if __name__ == "__main__":
    # --- CONFIGURATION ---
    IMAGE_PATH = "test_image.png" # input image
    MODEL_PATH = "./checkpoints/khmerocr_epoch100.pth" # choose model
    CHAR2IDX_PATH = "char2idx.json" # Tokenizer
    
    # Output Folder
    RESULT_FOLDER = "results"

    # Example Results
    """"
    Line 0: នេះជាអត្ថបទភាសាខ្មែរខ្លីមួយ៖
    Line 1: ប្រទេសកម្ពុជាមានវប្បធម៌ និងប្រវត្តិសាស្ត្រយូរអង្វែង។ ប្រជាជនខ្មែររស់នៅដោយការគោរពប្រពៃណី
    Line 2: និងអភិរក្សអត្តសញ្ញាណជាតិ។ ការអប់រំ និងបច្ចេកវិទ្យាកំពុងមានគួនាទីសំខាន់ក្នុងការអភិវឌ្ឍសង្គម
    Line 3: និងសេដ្ឋកិច្ច។ ‹យុវជនត្រូវបានលើកទឹកចិត្តឲ្យរៀនសូត្រ _និងច្នៃប្រឌិត ‹ដើម្បីរួមចំណែកកសាង
    Line 4: អនាគតដ៏រីកចម្រើន។
    Line 5: បើអ្នកចង់បានអត្ថបទវែង ឬមានប្រធានបទជាក់លាក់ ដូចជា វិទ្យាសាស្ត្រ បច្ចេកវិទ្យា ឬអប់រំ ខ្ញុំអាច
    Line 6: បង្កើតឲ្យបាន។
    """
```