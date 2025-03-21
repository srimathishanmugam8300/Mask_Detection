# Face Mask Detection Using MobileNetV2

## 📌 Overview
This project implements a deep learning model for real-time face mask detection using **MobileNetV2**. The model classifies images into two categories:
- **With Mask** 😷
- **Without Mask** 😷❌

The project leverages **TensorFlow, Keras, and TensorFlow Hub** for transfer learning, using a pre-trained MobileNetV2 model.

---

## 📂 Project Structure
```
📁 Face-Mask-Detection
│-- 📁 dataset/                 # Dataset directory (train, test, validation images)
│-- 📁 models/                  # Saved trained models
│-- model_creation.py           # Model Creation
│-- model_detection.py          # Detection
│-- requirements.txt            # Required dependencies
│-- README.md                   # Project documentation
```

---

## 🚀 Features
✅ Uses **MobileNetV2** for efficient and fast classification  
✅ **Transfer Learning** for better accuracy with fewer training samples  
✅ Supports **real-time detection** via webcam or images  
✅ Evaluates performance with **accuracy, precision, recall, F1-score, and confusion matrix**  
✅ Can be deployed as a **web app** using Flask or FastAPI  

---

## 🛠️ Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/srimathishanmugam8300/Mask_Detection.git
cd Mask_Detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download Dataset
Prepare a dataset with two folders:
```
📁 dataset/Test
📁 dataset/Train
    📁 with_mask
    📁 without_mask
```

---

## 🏗️ Work Flow:
- Load the dataset
- Apply **data augmentation**
- Train the **MobileNetV2** model
- Save the trained model in the `models/` directory

---

## 🤖 Technologies Used
- **TensorFlow** & **Keras** – Deep Learning framework
- **TensorFlow Hub** – Pre-trained models (MobileNetV2)
- **scikit-learn** – Model evaluation metrics

---

## 📌 Future Enhancements
- ✅ Improve real-time performance with **TensorFlow Lite**
- ✅ Deploy as a **mobile app** using **TFLite + Android**

---
