# IMPORTS
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import cv2
import pandas as pd
import numpy as np
import os

from torchvision import transforms
import torchvision.models as models  
from torchvision.transforms import Resize, RandomAffine, ToTensor, ToPILImage,  RandomErasing

import random
from PIL import Image, ImageEnhance 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import timm
import io

from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

# BACK-END
# CSV FILE PATHS MAPPING
test_csv_path = 'bookcover30/book30-listing-test.csv'
image_folder = 'bookcover30/224x224'
test_data = pd.read_csv(test_csv_path, header=None, encoding='ISO-8859-1')
test_texts = test_data.iloc[:, 3].tolist()
test_titles = test_data.iloc[:, 3].tolist()
test_labels = test_data.iloc[:, 6].tolist()
test_filenames = test_data.iloc[:, 1].tolist() 
test_filenames = [os.path.join(image_folder, fname) for fname in test_filenames]
label_names = [str(label) for label in sorted(np.unique(test_labels))]
label_to_index = {name: index for index, name in enumerate(label_names)}
test_labels_encoded = np.array([label_to_index[label] for label in test_labels])

# COLOR SPACE CONVERSION
COLOR_CONVERSIONS = {
    "RGB": None,
    "XYZ": cv2.COLOR_RGB2XYZ,
    "YCbCr": cv2.COLOR_RGB2YCrCb,
    "LAB": cv2.COLOR_RGB2Lab,
    "HSV": cv2.COLOR_RGB2HSV,
    "YUV": cv2.COLOR_RGB2YUV,
    "LUV": cv2.COLOR_RGB2Luv }

def convert_color_space(image, color_space):
    """Convert RGB image to a specified color space."""
    if color_space == "RGB":
        return image
    return cv2.cvtColor(image, COLOR_CONVERSIONS[color_space])

# DATA AUGMENTATION
def data_augmentation(image, rotation=False, blur=False, affine=False):
    image = Image.fromarray(image)  # Convert NumPy to PIL for PIL-based transforms
    pil_transforms = []  # Define list for PIL-based transforms

    # 🔹 Random Rotation
    if rotation:
        angle = np.random.uniform(-15, 15)
        image = image.rotate(angle)

    # 🔹 Random Affine Transformation (Shear)
    if affine:
        pil_transforms.append(RandomAffine(degrees=0, shear=10))

    # Apply PIL-based transformations AFTER adding them
    if pil_transforms:
        image = transforms.Compose(pil_transforms)(image)

    # Convert back to NumPy after augmentations
    image = np.array(image)

    # 🔹 Apply Gaussian Blur
    if blur:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    return image  

# PREPROCESSING

def preprocess_image(uploaded_file, color_spaces, img_size=(224, 224), augmentations=None, is_test=False): 
    # Read the image file from the uploaded file
    image_bytes = uploaded_file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    bgr_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # Read image as BGR

    if bgr_image is None:
        print("Error reading the image.")
        return None

    # Convert BGR image to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    tensors = []
    for space in color_spaces:
        converted = convert_color_space(rgb_image, space)  # Convert to target color space
        
        # Convert to PIL before applying PIL-based augmentations
        image = Image.fromarray(converted)
        
        # Apply augmentations only during training (excluding "erasing")
        if not is_test and augmentations:
            aug_params = {k: v for k, v in augmentations.items() if k != "erasing"}  # Exclude "erasing"
            converted = data_augmentation(np.array(image), **aug_params)  # Ensure NumPy format

        # Resize AFTER augmentations
        resized = cv2.resize(converted, img_size, interpolation=cv2.INTER_LINEAR)

        # Convert to PyTorch tensor
        tensor_image = ToTensor()(Image.fromarray(resized))

        # Apply `RandomErasing` (Only in training mode)
        if not is_test and augmentations.get("erasing", False):
            random_erasing = RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
            tensor_image = random_erasing(tensor_image)

        tensors.append(tensor_image)
    
    return tensors


# DATASET FUNCTIONS
class TestDataset(Dataset):
    def __init__(self, image_path, title, tokenizer, max_len, labels, color_spaces, img_size):
        self.image_path = image_path
        self.title = title
        self.labels = labels
        self.color_spaces = color_spaces
        self.img_size = img_size
        self.tokenizer = tokenizer
        self.max_len = max_len

        print(f"Image Path: {self.image_path}")
        print(f"Title: {self.title}")

    def __len__(self):
        return 1  # Since we now only have a single item, length is 1
        
    def __getitem__(self, idx):
        # No need to index because we only have a single item
        image_path = self.image_path
        label = self.labels[idx]
        book_title = self.title
        
        # Tokenize text (book title)
        encoding = self.tokenizer(
            book_title,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        image_tensors = preprocess_image(image_path, self.color_spaces, self.img_size, is_test=True)
        text_input_id = encoding["input_ids"].squeeze(0)
        text_attention_mask = encoding["attention_mask"].squeeze(0)

        # 🔹 If only one model is used, return a single tensor instead of a list
        if len(image_tensors) == 1:
            return image_tensors[0], text_input_id, text_attention_mask, label  # Return a single tensor
        
        return image_tensors, text_input_id, text_attention_mask, label  # Return multiple tensors for Parallel-ViT setup

    
# DATASET INITIALIZATION
# Define color spaces (Same for Train & Test)
color_spaces = ["RGB", "XYZ"]
img_size = (224, 224)

# Initialize tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128
BATCH_SIZE = 32

# PARALLEL VIT MODEL
class HybridModel(nn.Module):
    def __init__(self, num_classes=30, model_config=["cnn"], cnn_variant="resnext101_32x4d", vit_variant="vit_base_patch16_224", fusion_reduction=2, dropout_rate=0.1):
        super(HybridModel, self).__init__()
        self.model_config = model_config
        self.models = nn.ModuleList()
        self.feature_dims = []
        self.fusion_reduction = fusion_reduction
        self.dropouts = nn.ModuleList()
        self.cnn_variant = cnn_variant
        self.vit_variant = vit_variant


        # Load CNN Model 
        cnn_count = model_config.count("cnn")
        for _ in range(cnn_count):
            cnn_model = timm.create_model(cnn_variant, pretrained=True, num_classes=0)  # No classifier
            self.models.append(cnn_model)
            self.feature_dims.append(cnn_model.num_features)
            self.dropouts.append(nn.Dropout(dropout_rate))

        # Load ViT Model
        vit_count = model_config.count("vit")
        for _ in range(vit_count):
            vit_model = timm.create_model(vit_variant, pretrained=True, num_classes=0)  # No classifier layer
            self.models.append(vit_model)
            self.feature_dims.append(vit_model.num_features)
            self.dropouts.append(nn.Dropout(dropout_rate))

        # Feature fusion if multiple models are used
        if len(self.models) > 1:
            self.fusion_dim = sum(self.feature_dims)
            self.fusion_fc = nn.Linear(self.fusion_dim, self.fusion_dim // self.fusion_reduction)
        else:
            self.fusion_fc = None  

        # Final classifier
        final_feature_dim = (self.fusion_dim // self.fusion_reduction) if self.fusion_fc else self.feature_dims[0]
        self.classifier = nn.Linear(final_feature_dim, num_classes)
        
    def forward(self, *inputs):
        if len(self.models) != len(inputs):
            raise ValueError(f"Model input mismatch: Expected {len(self.models)} inputs but got {len(inputs)}.")
    
        features = []
        for model, inp, dropout in zip(self.models, inputs, self.dropouts):
            model_name = model._get_name().lower()
            x = model(inp)  
            x = x.mean(dim=1) if 'vit' in model._get_name().lower() else torch.flatten(x, 1)
    
            x = dropout(x)  # Apply dropout
            features.append(x)

        if len(features) > 1:
            x = torch.cat(features, dim=1)
            x = self.fusion_fc(x)
        else:
            x = features[0]
    
        x = self.classifier(x)
        return x

# Define device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model architecture
vit_model = HybridModel(
    num_classes=30,
    model_config=["vit", "vit"], 
    cnn_variant="resnext101_32x4d",
    vit_variant="vit_base_patch16_224",
    fusion_reduction=1,
    dropout_rate=0.5,
)

vit_model.to(device)

# Load the saved state_dict
state_dict = torch.load('models/parallel-vit.pth', weights_only=True, map_location=torch.device('cpu'))
vit_model.load_state_dict(state_dict)
vit_model = vit_model.to(device)

# Load BERT model from the local path
bert_model = BertForSequenceClassification.from_pretrained('models/bert', num_labels=30)
bert_model = bert_model.to(device)

def vit_predict(image, model):
    with torch.no_grad():
        if isinstance(image, torch.Tensor):
            image = image.to(device)
            logits = model(image)

        # 🔹 Handle Multiple Models (CNN-ViT, ViT-ViT, CNN-ViT-ViT)
        elif isinstance(image, list):
            image = [img.to(device) for img in image]
            logits = model(*image)
            
        probabilities = F.softmax(logits, dim=1).cpu().numpy()  # Apply softmax to logits
    return probabilities

def bert_predict(input_ids, attention_mask, model):
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # Forward pass to get logits
        probabilities = F.softmax(logits, dim=1).cpu().numpy()  # Apply softmax to logits
    return probabilities

def multimodal_predict(vit_probs, bert_probs, vit_weight, bert_weight, policy):
    # Normalize weights (for other policies)
    total_weight = vit_weight + bert_weight
    vit_w = vit_weight / total_weight
    bert_w = bert_weight / total_weight

    if policy == 'product-max':
        combined_probs = (vit_probs ** vit_weight) * (bert_probs ** bert_weight)
        predictions = np.argmax(combined_probs, axis=1)
    else:
        raise ValueError("Unsupported fusion policy")

    return combined_probs, predictions

def evaluate_multimodal_model(dataloader, vit_model, bert_model, vit_weight, bert_weight, policy):
    vit_model.eval()
    bert_model.eval()

    all_top1_labels = []  # List to store top-1 labels
    all_top3_labels = []  # List to store top-3 labels
    all_top1_confidences = []  # List to store top-1 confidence values
    all_top3_confidences = []  # List to store top-3 confidence values

    with torch.no_grad():
        for images, input_ids, attention_masks, labels in tqdm(dataloader, desc="Evaluating Multimodal Model"):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            if isinstance(images, torch.Tensor):
                images = images.to(device)

            # Handle Multiple Models (CNN-ViT, ViT-ViT, CNN-ViT-ViT)
            elif isinstance(images, list):
                images = [img.to(device) for img in images]
    
            # ViT and BERT predictions
            bert_probs = bert_predict(input_ids, attention_masks, bert_model)  # Shape: [batch_size, num_classes]
            vit_probs = vit_predict(images, vit_model)  # Shape: [batch_size, num_classes]
    
            # Combine probabilities using Product-Max heuristic with weights
            combined_probs, predictions_top1 = multimodal_predict(
                vit_probs, bert_probs, vit_weight=vit_weight, bert_weight=bert_weight, policy=policy
            )
    
            # Convert combined probabilities to a tensor for top-k evaluation
            combined_probs_tensor = torch.tensor(combined_probs, dtype=torch.float32).to(device)
    
            # Get Top-3 predictions and their confidence values
            preds_top3 = combined_probs_tensor.topk(3, dim=1).indices  # Top-3 predictions (indices)
            top3_confidence = combined_probs_tensor.topk(3, dim=1).values * 100  # Confidence for top-3 in percentage
    
            # Get the top-1 confidence value
            top1_confidence = top3_confidence[:, 0]  # First value is the top-1 confidence

            # Collect top-1 and top-3 labels and their confidence values
            all_top1_labels.extend(predictions_top1)
            all_top3_labels.extend(preds_top3.cpu().numpy())  # Flatten for easy use
            all_top1_confidences.extend(top1_confidence.cpu().numpy())
            all_top3_confidences.extend(top3_confidence.cpu().numpy())

    return all_top1_labels, all_top3_labels, all_top1_confidences, all_top3_confidences

# Weights for ViT(RGB) + BERT
vit_weight = 0.4  # Adjust weight for ViT
bert_weight = 0.6  # Adjust weight for BERT
policy = 'product-max'

# ===========================
# Streamlit App
# ===========================

def main():
    # Set browser tab title and favicon
    st.set_page_config(
        page_title="GenreNet"  # Change title as needed
    )

    # Sidebar content
    st.sidebar.markdown("""
        ## About GenreNet
        **Classifying books to their genre using a multimodal implementation of a Parallel ViT image classification model and BERT text classification model**  
        - **Task:** Book Genre Classification
        - **Image Model:** Parallel ViT (Vision Transformer)  
        - **Text Model:** BERT
        - **Inputs:** Book cover image and book title
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <h1 style='color: white; font-size: 40px; line-height: 0.1;'>
            GenreNet 📖
        </h1>
        <h3 style='color: white; font-size: 20px; line-height: 0.5;'>
            Identify a book's genre!
        </h3>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Book Cover Image", type=["jpg", "jpeg", "png"])
    title = st.text_input("Enter Book Title")

    if st.button("Classify"):
        if uploaded_file and title:
            # Create the test dataset (no augmentations for testing)
            test_dataset = TestDataset(uploaded_file, title, tokenizer, max_len, test_labels_encoded, color_spaces=color_spaces, img_size=img_size)

            # Dataloaders
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Evaluate the multimodal model
            top1_labels, top3_labels, top1_confidences, top3_confidences = evaluate_multimodal_model(
                test_loader, vit_model, bert_model, vit_weight=vit_weight, bert_weight=bert_weight, policy=policy
            )

            # Styling and displaying the top-1 label
            top1_label = label_names[top1_labels[0]]
            top1_confidence = top1_confidences[0]
            
            confidence_color = "#24be65" if top1_confidence >= 60 else "#ffeb3b" if top1_confidence >= 20 else "#d32f2f"

            # Add image display (optional)
            st.image(uploaded_file, use_container_width=False, width=300)

            # Reduced line spacing, white text, and bold confidence
            st.markdown(f"""
                <h3 style='font-size: 30px; line-height: 0.5;'>
                    Predicted Genre:
                </h3>

                <h1 style='color: white; font-size: 40px; line-height: 0.1;'>
                    <span style='font-weight: bold; color: {confidence_color};'>{top1_label}</span>
                </h1>

                <h3 style='font-size: 25px; line-height: 0.5;'>
                    Confidence: <span style='color: {confidence_color}; font-weight: bold'>{top1_confidence:.2f}% </span>
                </h3>
            """, unsafe_allow_html=True)

            # Display the top-2 and top-3 labels and their confidences
            st.subheader("Other Predictions:")
            other_predictions = []
            for i in range(1, 3):  # Start from index 1 to include top-2 and top-3
                label = label_names[top3_labels[0][i]]
                confidence = top3_confidences[0][i]
                rank = f"Top-{i+1}"
                other_predictions.append((rank, label, f"{confidence:.2f}%"))

            df_other_preds = pd.DataFrame(other_predictions, columns=["Rank", "Genre", "Confidence"])
            st.table(df_other_preds.set_index("Rank"))  # Remove default index

        else:
            st.warning("Please upload an image and enter a book title.")

if __name__ == "__main__":
    main()




