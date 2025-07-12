# 📘 ReadMe: GenreNet – Parallel ViT Model

This repository contains the model training and testing and web-based implementation of a **Book Genre Classification** model utilizing Parallel ViT model for image classification and BERT model for text classification. Below are the instructions for running the respective files for training/testing the model (backend) and deploying it as a web application (frontend).

## 📑 Table of Contents

1. [🧠 Backend: Training & Testing via Jupyter](#1-🧠-backend-training--testing-via-jupyter)  
    1.1 [Prerequisites](#11-prerequisites)  
    1.2 [Environment Setup](#12-environment-setup)  
    1.3 [Running GenreNet-ParallelViT.ipynb](#13-running-genrenet-parallelvitipynb)  
    1.4 [Running GenreNet-BERT.ipynb](#14-running-genrenet-bertipynb)  
    1.5 [Running GenreNet-ProductMax.ipynb](#15-running-genrenet-productmaxipynb)  
    1.6 [Output Files](#16-output-files)  
2. [🌐 Frontend: Web Deployment via Streamlit](#2-🌐-frontend-web-deployment-via-streamlit)  
    2.1 [Environment Setup](#21-environment-setup)  
    2.2 [Running the Web App](#22-running-the-web-app)  

---

## 1. 🧠 Backend: Training & Testing via Jupyter

### 1.1 Prerequisites

**Tools & Access**
- Jupyter access on a GPU server  
- Python ≥ 3.8 (already configured server-side)  
- CUDA GPU access (recommended)  
- A virtual environment (optional but cleaner)  

**Required Files**
- `GenreNet-ParallelViT.ipynb` – image model training & testing notebook  
- `GenreNet-BERT.ipynb` – text model training & testing notebook  
- `GenreNet-ProductMax.ipynb` – multimodal model testing notebook  
- `requirements.txt` – list of packages/libraries used  

- **BookCover30 Dataset** folder structured as:
    ```
    bookcover30/
    ├── 224x224/
    │   ├── 0001484524.jpg
    │   ├── 006062213X.jpg
    │   └── ...
    ├── book30-listing-test.csv
    ├── book30-listing-train.csv
    ├── bookcover30-labels-test.text
    └── bookcover30-labels-train.text
    ```

- All images are in the `224x224` folder.  
- `.csv` files include all metadata.  
- `.text` files contain only labels.  

---

### 1.2 Environment Setup

Install the required packages using one of the following commands:

```bash
pip install pandas numpy opencv-python-headless torch torchvision transformers Pillow tqdm matplotlib seaborn scikit-learn timm
```

Or use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

### 1.3 Running GenreNet-ParallelViT.ipynb

1. Open the notebook in your IDE.  
2. Update the save path in **cell [8]**:  
   ```python
   output_dir = '/folder/model_name'
   ```
3. Run all cells sequentially:
    - Load data
    - Convert color spaces
    - Data augmentation
    - Initialize Parallel ViT
    - Train and evaluate
    - Save best model

---

### 1.4 Running GenreNet-BERT.ipynb

1. Open the notebook.  
2. Update the save path in **cell [4]**:  
   ```python
   output_dir = '/folder/model_name'
   ```
3. Run all cells:
    - Load data
    - Tokenize text
    - Initialize BERT
    - Train and evaluate
    - Save best model

---

### 1.5 Running GenreNet-ProductMax.ipynb

1. Open the notebook.  
2. Update the save path in **cell [20]**:  
   ```python
   plt.savefig("saves/graphs/final_conf-matrix.png", bbox_inches='tight')
   ```
3. Run all cells:
    - Load data
    - Initialize ViT and BERT
    - Evaluate individual models
    - Fuse results
    - Evaluate multimodal model

---

### 1.6 Output Files

Each notebook produces the following outputs:

| File Type     | Directory         | Description                            |
|---------------|-------------------|----------------------------------------|
| `.log`        | `saves/logs/`     | Training and validation logs           |
| `.pth`        | `saves/models/`   | Best image model                       |
| `.safetensors`| `saves/models/`   | Best text model                        |
| `.png`        | `saves/graphs/`   | Training loss & confusion matrix plots |

---

## 2. 🌐 Frontend: Web Deployment via Streamlit

This section explains how to deploy the trained GenreNet model using Streamlit.

### 2.1 Environment Setup

Ensure the following files exist in the same directory:

| File Name         | Purpose                                 |
|-------------------|------------------------------------------|
| `GenreNet.py`     | Streamlit app main file                  |
| `/model/parallel-vit.pth` | Trained image model              |
| `/model/bert/`    | Trained text model directory             |
| `requirements.txt`| Package dependencies                     |

Install dependencies:

```bash
pip install -r requirements.txt
```


### 2.2 Running the Web App

1. **Open a terminal** and navigate to the project directory:

   ```bash
   cd path/to/your/project/directory
   ```

2. **Launch the app** with:

   ```bash
   streamlit run GenreNet.py
   ```

3. The app will automatically open in your browser, usually at: [http://localhost:8501](http://localhost:8501)

---

### Features of the App

- Upload an RGB book cover image  
- View top-1 and top-3 class predictions  
- Color-coded confidence levels:
    - 🟢 Green: High (>60%)
    - 🟡 Yellow: Moderate (20–59%)
    - 🔴 Red: Low (<20%)


### ⚠️ Model File Consistency

Check filenames in the code:

```python
# Load image model
state_dict = torch.load('models/parallel-vit.pth', weights_only=True, map_location=torch.device('cpu'))

# Load text model
bert_model = BertForSequenceClassification.from_pretrained('models/bert', num_labels=30)
```

- These appear in `GenreNet.py` lines [261] and [266].

- Update paths or filenames as necessary to avoid `FileNotFoundError`.

- You can rename the model file or update the load_state_dict() path accordingly.

If you have a GPU with CUDA, replace line [261] with the following:
```python
# Load image model
state_dict = torch.load('models/parallel-vit.pth', weights_only=True, map_location=device)
```
