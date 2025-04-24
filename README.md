# 📸 No-Reference Image Quality Assessment (NRIQA)

🚀 Automatically assess image quality without a reference! This project predicts the Mean Opinion Score (MOS) of an image using a deep learning model.

### 📌 Project Overview

No-Reference Image Quality Assessment (NRIQA) is a machine learning model that predicts image quality without needing a high-quality reference image. It uses a Convolutional Neural Network (CNN) trained on the TID2013 dataset to classify images into different quality levels.

### ✔ Objective: Predict image quality in real-time.
✔ Dataset Used: TID2013
✔ Model Type: TensorFlow/Keras CNN
✔ Deployment: Hugging Face Spaces + Gradio

### 🔍 Features

✅ Predicts the perceived quality of an image.
✅ Outputs a Mean Opinion Score (MOS) and quality category (Excellent, Good, Average, Poor, Very Poor).
✅ Works without requiring a reference image.
✅ Supports real-time predictions using Gradio.

### 📺 Dataset Used

We used the TID2013 dataset, which consists of 3,000 distorted images across 25 reference images with 5 different levels of distortion. Each image has a Mean Opinion Score (MOS) based on human perception.

### 🗂 Dataset Structure:

Images: Ixx_xx_x.bmp (Distorted Images)

MOS Scores: Provided in mos_with_names.txt

### ⚙️ Model Architecture

The deep learning model follows a CNN-based architecture:

1️⃣ Preprocessing:

Resize images to (224, 224, 3).

Normalize pixel values to [0,1].

Convert image names from lowercase (i) to uppercase (I) to match dataset files.

2️⃣ Model Training:

 MobileNet-inspired CNN layers extract image features.

Fully connected DNN layers map features to MOS scores.

Mean Squared Error (MSE) is used as the loss function.

3️⃣ Prediction:

Given an image, the model predicts its MOS score and quality category.

### 📾 Installation & Setup

🔹 1️⃣ Clone the Repository

git clone https://github.com/Janardhan11/no-reference-iqa.git
cd no-reference-iqa

🔹 2️⃣ Install Dependencies

pip install -r requirements.txt

🔹 3️⃣ Download the Model

The trained model (image_quality_model.h5) is not included in this repository due to size limitations.

➡️ Download the model and place it inside the model/ folder.

### 🚀 Running the Project

🔹 1️⃣ Run the Gradio App

python app.py

🔹 2️⃣ Upload an Image

The app will analyze the image and return:

MOS Score (Numeric score for quality)

Quality Category (Excellent, Good, Average, etc.)

### 🎨 Example Output

✅ Input: Sample image

✅ Output:

Predicted Quality: ✅ Good

### 🛠️ Deployment

The project is deployed on Hugging Face Spaces using Gradio.➡️ Try it here: 🔗 [Live Demo](https://huggingface.co/spaces/JanardhanM/no-reference-iqa)

🌟 Future Improvements

🚀 Further improvements can include:✔ Training on more diverse datasets✔ Using Vision Transformers (ViTs) for better performance✔ Deploying with TensorFlow.js for browser-based predictions

👨‍💻 Author

Developed by Janardhan Medathati🔗 GitHub: @Janardhan11🔗 Hugging Face: @JanardhanM

⭐ Contributing

Got ideas? Feel free to fork the repo, open a pull request, or create an issue!
