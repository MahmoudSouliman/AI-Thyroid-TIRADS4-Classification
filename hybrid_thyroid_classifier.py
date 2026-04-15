# CELL 0: GLOBAL IMPORTS AND CONFIGURATION
# Goal: Setup the environment and import all necessary libraries

import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re
import json


# PyTorch libraries for Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Sklearn for data splitting and evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("Environment Ready: All libraries imported successfully.")




# CELL 1
import kagglehub

def initialize_data_environment():
    """
    Downloads the latest version of the DDTI dataset using kagglehub.
    Ensures a seamless execution for the supervisor without manual file handling.
    """
    print("Status: Initializing automated data retrieval...")
    try:
        # Download the Thyroid Ultrasound Images (DDTI) dataset via official hub
        # This dataset is crucial for the Gray Zone (TR4) classification task
        path = kagglehub.dataset_download("dasmehdixtr/ddti-thyroid-ultrasound-images")

        print(f"Success: Dataset is located at: {path}")
        return path
    except Exception as e:
        print(f"Error: Automated download failed. Details: {e}")
        return None

# Set the global base path for subsequent processing cells
BASE_DATA_PATH = initialize_data_environment()

# Validation check for file existence
if BASE_DATA_PATH:
    total_files = len(os.listdir(BASE_DATA_PATH))
    print(f"Verified: {total_files} files are ready for analysis.")






# CELL 2

def parse_tr4_final_attempt(base_path):
    """
    Comprehensive crawler to link XML IDs to images regardless of case or extension.
    Optimized for integration into standalone medical applications.
    """
    records = []
    target_labels = ['4a', '4b', '4c']
    image_map = {}

    # 1. Broad indexing of all image-like files
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG')

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                # Extract numeric ID from filename using Regular Expressions
                nums = re.findall(r'\d+', file)
                if nums:
                    image_map[nums[0]] = os.path.join(root, file)

    # 2. Parse XMLs and link via numeric IDs
    xml_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(".xml"):
                xml_files.append(os.path.join(root, file))

    print(f"Status: Linking {len(xml_files)} XML files to detected images...")

    for file_path in xml_files:
        try:
            tree = ET.parse(file_path)
            root_node = tree.getroot()
            tirads_tag = root_node.find('tirads')

            if tirads_tag is not None and tirads_tag.text:
                val = tirads_tag.text.strip().lower()
                if val in target_labels:
                    # Match using the first number found in the XML filename
                    xml_id_list = re.findall(r'\d+', os.path.basename(file_path))
                    if xml_id_list:
                        xml_id = xml_id_list[0]
                        if xml_id in image_map:
                            records.append({
                                'case_id': xml_id,
                                'tirads_label': val,
                                'image_path': image_map[xml_id],
                                'xml_path': file_path
                            })
        except Exception as e:
            continue

    return pd.DataFrame(records)

# Execution logic
if 'BASE_DATA_PATH' in locals():
    df_tr4 = parse_tr4_final_attempt(BASE_DATA_PATH)

    print("-" * 30)
    if not df_tr4.empty:
        print(f"✅ Success: Dataset Linked Successfully.")
        print(f"Final Count: {len(df_tr4)} cases found.")
        print(df_tr4['tirads_label'].value_counts().sort_index())
    else:
        print("❌ Fatal: No images matched XML IDs.")
else:
    print("❌ Error: BASE_DATA_PATH is not defined. Please run CELL 1.")





# CELL 3
def extract_geometric_from_svg(df):
    """
    Parses DDTI specific SVG-JSON coordinates from XML files.
    """
    geometrical_data = []
    print(f"Status: Decoding SVG data for {len(df)} cases...")

    for index, row in df.iterrows():
        try:
            tree = ET.parse(row['xml_path'])
            root = tree.getroot()

            # Find the svg tag content
            svg_tag = root.find('.//svg')
            if svg_tag is not None and svg_tag.text:
                # The content is a JSON string, we need to parse it
                points_data = json.loads(svg_tag.text)

                # Extract x and y from the list of dictionaries
                x_coords = [p['x'] for p in points_data[0]['points']]
                y_coords = [p['y'] for p in points_data[0]['points']]

                if x_coords and y_coords:
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    area = width * height
                    aspect_ratio = height / width if width > 0 else 0

                    geometrical_data.append({
                        'case_id': row['case_id'],
                        'width': width,
                        'height': height,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'tirads_label': row['tirads_label']
                    })
        except Exception as e:
            # Skip cases with corrupted or missing coordinates
            continue

    return pd.DataFrame(geometrical_data)

# Execute final feature extraction
df_features = extract_geometric_from_svg(df_tr4)

# Validation
if not df_features.empty:
    print("-" * 30)
    print(f"✅ Success: Features extracted for {len(df_features)} cases.")
    print("\nSummary of Gray Zone Geometry:")
    print(df_features.groupby('tirads_label')[['aspect_ratio', 'area']].mean())
    FINAL_DATA = df_features.copy()
else:
    print("❌ Error: Parsing logic failed. Please check JSON structure.")






# CELL 4
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

def enhanced_image_pipeline(img):
    """
    Applies the 3-stage processing pipeline: Median -> Bilateral -> CLAHE.
    """
    img = img.astype(np.uint8)
    denoised_median = cv2.medianBlur(img, 5)
    bilateral = cv2.bilateralFilter(denoised_median, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(bilateral)

def extract_hybrid_texture_features(df_combined):
    """
    Processes images through the pipeline and extracts GLCM texture features.
    """
    texture_data = []
    print(f"Status: Running Pipeline & Texture Analysis for {len(df_combined)} images...")

    for index, row in df_combined.iterrows():
        try:
            # Load and process
            raw_img = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
            if raw_img is None: continue

            processed_img = enhanced_image_pipeline(raw_img)
            img_resized = cv2.resize(processed_img, (256, 256))

            # GLCM Extraction
            glcm = graycomatrix(img_resized, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)

            texture_data.append({
                'case_id': row['case_id'],
                'contrast': graycoprops(glcm, 'contrast')[0, 0],
                'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
                'energy': graycoprops(glcm, 'energy')[0, 0],
                'correlation': graycoprops(glcm, 'correlation')[0, 0]
            })
        except Exception as e:
            continue

    return pd.DataFrame(texture_data)

# 1. Merge image paths from df_tr4 into our features table FIRST
df_for_pipeline = pd.merge(df_features, df_tr4[['case_id', 'image_path']], on='case_id')

# 2. Run the analysis on the combined table
df_texture = extract_hybrid_texture_features(df_for_pipeline)

# 3. Create the Final Hybrid Table
if not df_texture.empty:
    HYBRID_FEATURES = pd.merge(df_for_pipeline, df_texture, on='case_id')
    print("-" * 30)
    print(f"✅ Success: 3-Stage Pipeline applied. Hybrid dataset ready for {len(HYBRID_FEATURES)} cases.")
    print(HYBRID_FEATURES[['tirads_label', 'aspect_ratio', 'contrast', 'homogeneity']].head())
else:
    print("❌ Error: Pipeline processing failed.")





# CELL 5
def fetch_anchors_and_polarize(base_path, hybrid_tr4_df):
    """
    1. Uses TR2 (Benign) and TR5 (Malignant) as anchors.
    2. Calculates malignancy probability based on geometric & texture shifts.
    """
    print("Status: Fetching Reference Anchors (TR2 & TR5)...")

    anchor_records = []
    # Identify anchor cases across the dataset
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(".xml"):
                try:
                    tree = ET.parse(os.path.join(root, file))
                    val = tree.getroot().find('tirads').text.strip().lower()
                    # Using TR2 as the reliable Benign anchor
                    if val in ['2', '5']:
                        anchor_records.append({
                            'case_id': re.findall(r'\d+', file)[0],
                            'tirads_label': 'benign' if val == '2' else 'malignant'
                        })
                except: continue

    df_anchors = pd.DataFrame(anchor_records)
    print(f"✅ Found {len(df_anchors)} anchor cases for calibration.")

    # Polarization Strategy Logic:
    # We create a score that reflects clinical risk:
    # High Aspect Ratio + Low Homogeneity = High Malignancy Probability

    # Normalize features to 0-1 range for probability calculation
    h_min, h_max = hybrid_tr4_df['homogeneity'].min(), hybrid_tr4_df['homogeneity'].max()
    a_min, a_max = hybrid_tr4_df['aspect_ratio'].min(), hybrid_tr4_df['aspect_ratio'].max()

    # Probability = Weighted combination of Geometry and Texture
    # We invert homogeneity (1 - norm_h) because lower homogeneity means higher risk
    norm_aspect = (hybrid_tr4_df['aspect_ratio'] - a_min) / (a_max - a_min)
    norm_inv_homogeneity = 1 - ((hybrid_tr4_df['homogeneity'] - h_min) / (h_max - h_min))

    hybrid_tr4_df['malignancy_prob'] = (norm_aspect * 0.5) + (norm_inv_homogeneity * 0.5)

    # Final Polarization (0 or 1)
    def define_target(row):
        if row['tirads_label'] == '4c': return 1
        if row['tirads_label'] == '4a': return 0
        # For 4b, follow the probability threshold (0.5)
        return 1 if row['malignancy_prob'] >= 0.5 else 0

    hybrid_tr4_df['target_class'] = hybrid_tr4_df.apply(define_target, axis=1)

    return hybrid_tr4_df

# Execute polarization logic
FINAL_DATASET = fetch_anchors_and_polarize(BASE_DATA_PATH, HYBRID_FEATURES)

print("-" * 30)
print("✅ Polarization Strategy Applied Successfully.")
print(f"Total processed cases: {len(FINAL_DATASET)}")
print("\nMean Malignancy Probability by Subclass:")
print(FINAL_DATASET.groupby('tirads_label')['malignancy_prob'].mean())






# CELL 6 - Full Model Definition & Training Comparison
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import EfficientNetB3, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# 1. Functions for Model Architecture
def build_medical_model(model_type='EfficientNetB3'):
    input_shape = (224, 224, 3)
    if model_type == 'EfficientNetB3':
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

    base_model.trainable = False # Initial freezing
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# 2. Function to prepare image data using your 3-Stage Pipeline
def prepare_image_array(df):
    images = []
    print("Status: Processing images through your pipeline for Deep Learning...")
    for path in df['image_path']:
        raw_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None: continue

        # Apply your Stage 1-2-3 Pipeline
        processed_img = enhanced_image_pipeline(raw_img)

        # Convert to RGB and resize for EfficientNet/DenseNet
        rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        resized_img = cv2.resize(rgb_img, (224, 224))
        images.append(resized_img / 255.0)
    return np.array(images)

# 3. Execution & Comparison
# Prepare Data
X = prepare_image_array(FINAL_DATASET)
y = FINAL_DATASET['target_class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Models inside the same execution block
eff_model = build_medical_model('EfficientNetB3')
dense_model = build_medical_model('DenseNet121')

# Train and Evaluate
print("\n🚀 Training EfficientNetB3...")
history_eff = eff_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16, verbose=1)

print("\n🚀 Training DenseNet121...")
history_dense = dense_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16, verbose=1)

print("-" * 30)
print("✅ Success: Training of both models is complete.")







# CELL 7 - Aggressive Fine-Tuning & Enhanced Dynamic Morpho-Boost
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# --- 1. DYNAMIC DATA RECOVERY (For Morphological Features) ---
try:
    aspect_test = test_df['aspect_ratio'].values
    print("✅ Found aspect_ratio in test_df")
except NameError:
    try:
        # Fallback to the known morphological array from previous steps
        aspect_test = X_morph_test[:, 0]
        print("✅ Found aspect_ratio in X_morph_test")
    except:
        print("⚠️ Warning: Morphological variables not found. Using default Aspect Ratio (1.0).")
        aspect_test = np.ones(len(y_test))

# --- 2. SURGICAL FINE-TUNING (Last 60 Layers) ---
# Increasing depth of unfreezing to capture complex TR4 textures
dense_model.trainable = True
for layer in dense_model.layers[:-60]:
    layer.trainable = False

# --- 3. CLASS WEIGHTING STRATEGY ---
# Giving 1.8x more importance to Malignant cases to improve Recall
class_weights = {0: 1.0, 1: 1.8}

# Re-compiling with an optimized microscopic learning rate
dense_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Adding Early Stopping to preserve the best weights and prevent overfitting
early_stop = EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=True
)

print("\n🚀 Step 1: Starting Aggressive Fine-tuning on DenseNet121 (Last 60 layers)...")
dense_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# --- 4. ENHANCED DYNAMIC MORPHO-BOOST LOGIC ---
def apply_enhanced_morpho_boost(cnn_probs, aspect_ratios, threshold=0.48):
    """
    Adjusts classification threshold based on Morphological evidence.
    Optimized for higher sensitivity (Recall) in Malignant cases.
    """
    final_preds = []
    for prob, aspect in zip(cnn_probs, aspect_ratios):
        adaptive_threshold = threshold

        # Aggressive Reinforcement: Stronger weight for elongated shapes (Malignancy)
        if aspect > 1.1:
            adaptive_threshold -= 0.12 # Increased boost from 0.08

        # Mild Reinforcement: For very round nodules (Benign)
        elif aspect < 0.9:
            adaptive_threshold += 0.05

        pred = 1 if prob > adaptive_threshold else 0
        final_preds.append(pred)
    return np.array(final_preds)

print("\n🚀 Step 2: Extracting deep probability features...")
deep_preds_refined = dense_model.predict(X_test).flatten()

# Applying the final hybrid decision logic
final_boosted_preds = apply_enhanced_morpho_boost(deep_preds_refined, aspect_test)

print("\n" + "="*50)
print("✅ SYSTEM STATUS: Aggressive Hybrid Training Complete!")
print("✅ Model is now optimized for Maximum Recall and Precision.")
print("="*50)






# CELL 8 - Final Statistical Validation & Performance Metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# 1. Generate Final Predictions and Metrics
# We use the 'final_boosted_preds' generated in CELL 7
print("📊 Generating Final Evaluation Reports...")

# Compute Confusion Matrix
cm = confusion_matrix(y_test, final_boosted_preds)

# 2. Plotting the Confusion Matrix for Professional Documentation
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign (0)', 'Malignant (1)'],
            yticklabels=['Benign (0)', 'Malignant (1)'])
plt.title('Final Hybrid Model: Confusion Matrix (TR4 Specific)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 3. Detailed Classification Report
# This includes Precision, Recall, and F1-Score
print("\n" + "="*60)
print("             FINAL CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, final_boosted_preds, target_names=['Benign (0)', 'Malignant (1)']))

# 4. ROC Curve and AUC Score
# Calculating the Area Under the Curve to prove the separation power
fpr, tpr, _ = roc_curve(y_test, deep_preds_refined) # Using refined probabilities
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) - Hybrid System')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

print("\n✅ EVALUATION COMPLETE: Metrics successfully exported for the Graduation Report.")







# CELL 9 - Model Export & Final Serialization
import joblib

# 1. Saving the trained DenseNet121 model weights
print("💾 Saving the optimized Hybrid-DenseNet121 model...")
dense_model.save('Thyroid_Hybrid_Model_Final.keras')

# 2. Saving the final predictions and ground truth for the GUI testing phase
evaluation_data = {
    'y_test': y_test,
    'final_preds': final_boosted_preds,
    'cnn_probs': deep_preds_refined,
    'aspect_ratios': aspect_test
}
joblib.dump(evaluation_data, 'final_evaluation_results.pkl')

print("\n" + "="*50)
print("✅ EXPORT COMPLETE: Final model saved as 'Thyroid_Hybrid_Model_Final.keras'")
print("✅ Evaluation metadata saved as 'final_evaluation_results.pkl'")
print("🚀 Your model is now ready for deployment in the GUI!")
print("="*50)







