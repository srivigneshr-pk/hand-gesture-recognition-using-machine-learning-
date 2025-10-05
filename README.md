README
Hand Gesture Recognition using Machine Learning
📌 Overview
This project implements a Hand Gesture Recognition system using Machine Learning (SVM classifier) and OpenCV. The model is trained on the LeapGestRecog dataset to classify different hand gestures from images.
🚀 Features
Preprocesses images (grayscale + resize to 64×64)
Flattens images into feature vectors
Trains an SVM classifier on gesture images
Achieves high accuracy on test data
Supports testing with your own image
🛠 Tech Stack
Python 3
OpenCV (cv2)
scikit-learn
NumPy
📂 Dataset
The project uses the LeapGestRecog dataset containing gesture images. Download and extract it before running the code.
▶ UsageTrain & Evaluate Model
Run the program:
python program.py

Test with Your Own Image
Place your image (e.g., my_hand.jpg) in the project folder and update:

test_img = cv2.imread("my_hand.jpg", cv2.IMREAD_GRAYSCALE)

📊 Results

Model accuracy: 1.0 (100%) on the dataset

Successfully predicts gestures like palm, fist, thumbs up, etc.

🔮 Future Work

Implement real-time gesture recognition using webcam

Try with deep learning (CNNs) for better accuracy
