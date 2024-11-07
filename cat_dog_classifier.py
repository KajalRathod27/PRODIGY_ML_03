import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

# Set image size for resizing
image_size = (64, 64)
data = []
labels = []

# Define the path to the dataset folder
dataset_dir = "C:/Users/Kajal Rathod/OneDrive/Desktop/PRODIGY_INTERNSHIP/PRODIGY_ML_03/Dataset"
pet_images_dir = os.path.join(dataset_dir, "PetImages")

# Ensure PETImages folder exists
if not os.path.exists(pet_images_dir):
    raise FileNotFoundError(f"PETImages folder not found in {dataset_dir}")

# Define augmentation pipeline (smaller set of augmentations)
augmenter = iaa.Sequential([iaa.Fliplr(0.5)])  # Horizontal flips

# Loop through the subdirectories (Cat and Dog)
for label in ["Cat", "Dog"]:
    folder_path = os.path.join(pet_images_dir, label)

    # Ensure subfolder exists
    if not os.path.exists(folder_path):
        print(f"Warning: Subfolder {label} not found in PETImages.")
        continue

    image_count = 0
    for image_name in os.listdir(folder_path):
        # Skip non-image files
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(folder_path, image_name)

        # Read and resize image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping invalid image: {image_path}")
            continue

        image = cv2.resize(image, image_size)

        # Add original image
        data.append(image.flatten())
        labels.append(label)

        # Apply one augmentation (not multiple)
        aug_image = augmenter(image=image)
        data.append(aug_image.flatten())
        labels.append(label)

        # Limit to 500 images per class for speed
        image_count += 1
        if image_count >= 500:
            break

# Convert lists to numpy arrays
X = np.array(data)
y = np.array(labels)

# Ensure there is enough data
if len(X) == 0 or len(y) == 0:
    raise ValueError("No valid images were loaded. Check your dataset structure and files.")

# Encode labels ('Cat' -> 0, 'Dog' -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split Data into Train and Test Sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Train the SVM Model
svm_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=50),  # Reduce PCA components
    SVC(kernel="rbf", random_state=42)
)

# Hyperparameter tuning with reduced grid
param_grid = {
    "svc__C": [1, 10],  # Smaller range for C
    "svc__gamma": [0.01, 0.1]  # Smaller range for gamma
}
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)

print("Training the model with GridSearchCV...")
grid_search.fit(X_train, y_train)

# Best parameters and score
print("\nBest Parameters:", grid_search.best_params_)
print("Best Training Accuracy:", grid_search.best_score_)

# Evaluate the Model
print("Evaluating the model...")
y_pred = grid_search.best_estimator_.predict(X_test)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Store original images before preprocessing (For visualization)
X_original = []
y_original = []

for label in ["Cat", "Dog"]:
    folder_path = os.path.join(pet_images_dir, label)

    image_count = 0
    for image_name in os.listdir(folder_path):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image = cv2.resize(image, image_size)
        X_original.append(image)
        y_original.append(label)

        image_count += 1
        if image_count >= 500:  # Limit for speed
            break

# Split the original dataset into train and test sets
X_original = np.array(X_original)
y_original_encoded = label_encoder.transform(y_original)  # Reuse label_encoder

_, X_test_original, _, y_test_original = train_test_split(
    X_original, y_original_encoded, test_size=0.2, stratify=y_original_encoded, random_state=42
)

# Visualization of some predictions
print("\nVisualizing some predictions...")

num_samples = 10
indices = np.random.choice(len(X_test_original), size=num_samples, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

for i, idx in enumerate(indices):
    original_image = X_test_original[idx]
    true_label = label_encoder.classes_[y_test_original[idx]]
    predicted_label = label_encoder.classes_[y_pred[idx]]

    # Display the image
    axes[i].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    axes[i].set_title(f"True: {true_label}\nPred: {predicted_label}", fontsize=10)
    axes[i].axis("off")

plt.tight_layout()
plt.show()
