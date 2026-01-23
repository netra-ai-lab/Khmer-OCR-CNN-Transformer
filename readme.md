# A Squeeze-and-Excitation Transformer Network for Khmer Optical Character Recognition

This repository contains the implementation, datasets, and evaluation results for the **Squeeze-and-Excitation Transformer Network**, a high-performance Khmer Text Recognition model that utilizes a hybrid architecture combining **Squeeze-and-Excitation** blocks for feature extraction and **BiLSTM** smoothing for context smoothing, specifically designed to handle the complexity and length of Khmer script.

## Project Overview

Khmer script presents unique challenges for OCR due to its large character set, complex sub-consonant stacking, and variable text line lengths. This project proposes an enhanced pipeline that:
1.  **Chunks** long text lines into manageable overlapping segments.
2.  **Extracts Features** using a **Squeeze-and-Excitation Network** (SE-VGG) that preserves horizontal spatial information.
3.  **Encodes** local spatial features using a Transformer Encoder.
4.  **Merges** the encoded chunks into a unified sequence.
5.  **Smooths Context** using a **BiLSTM** layer to resolve boundary discontinuities between chunks.
6.  **Decodes** the final sequence using a Transformer Decoder.

## Datasets

The model was trained entirely on synthetic data and evaluated on real-world datasets. You can access the full training datasets on **Hugging Face**: 

[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/collections/Darayut/khmer-text-synthetic)

### Training Data (Synthetic)
We generated **200,000 synthetic images** to ensure robustness against font variations and background noise.

| Dataset Type | Count | Generator / Source | Augmentations |
| :--- | :--- | :--- | :--- |
| **Document Text** | 100,000 | Pillow + Khmer Corpus | Erosion, noise, thinning/thickening, perspective distortion. |
| **Scene Text** | 100,000 | SynthTIGER + Stanford BG | Rotation, blur, noise, realistic backgrounds. |

### Evaluation Data (Real-World + Synthetic)
| Dataset | Type | Size | Description |
| :--- | :--- | :--- | :--- |
| **KHOB** | Real | 325 | Standard benchmark, clean backgrounds but compression artifacts. |
| **Legal Documents** | Real | 227 | High variation in degradation, illumination, and distortion. |
| **Printed Words** | Synthetic | 1,000 | Short, isolated words in 10 different fonts. |

![Dataset Overview](./assets/dataset-overview.png)
---

## Methodology & Architecture

### 1. Preprocessing: Chunking & Merging
To handle variable-length text lines without aggressive resizing, we employ a "Chunk-and-Merge" strategy:
*   **Resize:** Input images are resized to a fixed height of 48 pixels while maintaining aspect ratio.
*   **Chunking:** The image is split into overlapping chunks (Size: 48x100 px, Overlap: 16 px).
*   **Independent Encoding:** Each chunk is processed independently by the Squeeze-and-Excitation Network and Transformer Encoder to allow for parallel batch processing.

### 2. Model Architecture: SeqSE-CRNN-Transformer
Our proposed architecture integrates sequence-aware attention and recurrent smoothing to overcome the limitations of standard chunk-based OCR. The model consists of six key modules:

![Model Architecture](./assets/proposed-architecture.png)

1.  **Squeeze-and-Excitation Network (SE-VGG):**
    *   A modified VGG backbone with **1D Squeeze-and-Excitation** blocks after convolutional layer **3**, **4**, and **5**.
    *   Unlike standard SE, these blocks use **vertical pooling** to refine feature channels while strictly preserving the horizontal width (sequence information).

        ![SE Module](<assets/Sequence Attention CNN.png>)


2.  **Patch Module:**
    *   Projects spatial features into a condensed **384-dimensional** embedding space.
    *   Adds local positional encodings to preserve spatial order within chunks.

3.  **Transformer Encoder:**
    *   Captures contextual relationships among visual tokens within each independent chunk.

4.  **Merging Module:**
    *   Concatenates the encoded features from all chunks into a single unified sequence.
    *   Adds **Global Positional Embeddings** to define the absolute position of tokens across the entire text line.

5.  **BiLSTM Context Smoother:**
    *   A Bidirectional LSTM layer that processes the merged sequence.
    *   **Purpose:** Bridges the "context gap" between independent chunks by smoothing boundary discontinuities, ensuring a seamless flow of information across the text line.

        ![Context Smoothing Module](assets/BiLSTM-Module.png)

6.  **Transformer Decoder:**
    *   Generates the final Khmer character sequence using the globally smoothed context.

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

We benchmarked our proposed **SeqSE-CRNN-Tranformer** against VGG-Transformer and ResNet-Transformer models, and Tesseract-OCR.

**Character Error Rate (CER %)** - *Lower is better*

TABLE 1: Character Error Rate (CER in %) results on the KHOB, Legal Documents, and Printed Word

| Model | KHOB | Legal Documents | Printed Word |
| :--- | :--- | :--- | :--- |
| Tesseract-OCR | 9.36 | 24.30 | 8.02 |
| VGG-Transformer | 5.07 | 10.27 | 3.61 |
| ResNet-Transformer | 5.85 | 11.57 | $\textcolor{yellow}{2.80}$ |
| **Proposed Model** | $\textcolor{yellow}{4.79}$ | $\textcolor{yellow}{9.13}$ | 3.44 |

---

## Qualitative Analysis

TABLE 2: Failure cases on KHOB, Legal Document, and Printed Word dataset
![failure cases](assets/failure_cases.png)

TABLE 3: Example of proposed, and baseline model compared with the ground truth. Errors in the predictions are highlighted in red
![success cases](assets/sucess_case.png)

**Key Findings:**
*   **The Proposed Model** achieves the highest accuracy on long, continuous text lines (KHOB), demonstrating that the **BiLSTM Context Smoother** effectively resolves the chunk boundary discontinuities that limit standard Transformer baselines.
*   On degraded and complex legal documents, **the proposed model** demonstrates superior robustness, significantly outperforming all baselines. This attributes to the **Squeeze-and-Excitation blocks**, which filter background noise while preserving character-specific features.
*   **ResNet-Transformer** retains a slight advantage on short, isolated words where global context is less critical, though our proposed model still outperforms the VGG-Transformer baseline in this category.


### Installation
```bash
git clone https://github.com/Netra-AI-Lab/Khmer-OCR-CNN-Transformer.git
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
To run the OCR on an image, and get result in the form of a text file, simply execute the script:
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
    MODEL_PATH = "./checkpoints/khmerocr_vgg_lstm_epoch100.pth" # choose model
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

### End-to-End Image to Edtiable PDF
In order to automatically convert an image directly to PDF file while preserving the layout structure, execute the script below:
```bash
python inference_pdf.py
```
Below is the original document image and the result of the editable PDF with layout preservation after OCR:
<p float="left">
  <img src="khmer_document_4.jpg" width="40%" />
  <img src="./assets/pdf_convert.png" width="45%" /> 
</p>

---
## References

1. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**  
   *Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.*  
   ICLR 2021.  
   [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

2. **TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models**  
   *Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei.*  
   AAAI 2023.  
   [arXiv:2109.10282](https://arxiv.org/abs/2109.10282)

3. **Toward a Low-Resource Non-Latin-Complete Baseline: An Exploration of Khmer Optical Character Recognition**  
   *R. Buoy, M. Iwamura, S. Srun and K. Kise.*  
   IEEE Access, vol. 11, pp. 128044-128060, 2023.  
   [DOI: 10.1109/ACCESS.2023.3332361](https://doi.org/10.1109/ACCESS.2023.3332361)

4. **Balraj98.** (2018). *Stanford background dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/balraj98/stanford-background-dataset

5. **EKYC Solutions.** (n.d.). *Khmer OCR benchmark dataset (KHOB)* [Data set]. GitHub. https://github.com/EKYCSolutions/khmer-ocr-benchmark-dataset

6. **Em, H., Valy, D., Gosselin, B., & Kong, P.** (2024). *Khmer text recognition dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/emhengly/khmer-text-recognition-dataset

7. **Squeeze-and-Excitation Networks**  
   *Jie Hu, Li Shen, and Gang Sun.*  
   CVPR 2018.  
   [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)

8. **Bidirectional Recurrent Neural Networks**  
   *Mike Schuster and Kuldip K. Paliwal.*  
   IEEE Transactions on Signal Processing, 1997.  
   [DOI: 10.1109/78.650093](https://doi.org/10.1109/78.650093)