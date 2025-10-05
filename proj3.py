import os
import cv2
import random
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ======== Settings ========
base_path = r"C:\Users\Reference\Desktop\svm img\archive (2)\animals"

img_size = (128, 128)

# ======== Load Dataset ========
X = []
y = []
image_paths = []

for label, folder in enumerate(["cat", "dog"]):
    folder_path = os.path.join(base_path, folder)
    count = 0
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        
        # Extract HOG features
        features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys')
        X.append(features)
        y.append(label)
        image_paths.append(img_path)
        count += 1
    
    print(f"{folder.capitalize()} images: {count}")

# ======== Train-Test Split ========
X_train, X_test, y_train, y_test, train_paths, test_paths = train_test_split(
    X, y, image_paths, test_size=0.2, random_state=42
)

# ======== Train SVM ========
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# ======== Evaluate ========
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# ======== Show Predictions ========
plt.figure(figsize=(10, 6))
for i in range(6):
    idx = random.randint(0, len(X_test) - 1)
    img_path = test_paths[idx]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    
    prediction = clf.predict([X_test[idx]])[0]
    plt.subplot(2, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title("Pred: " + ("Cat" if prediction == 0 else "Dog"))
    plt.axis("off")

plt.tight_layout()
plt.show()
