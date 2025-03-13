# 🐟 Multiclass Fish Image Classification

This project focuses on classifying different fish species using **Convolutional Neural Networks (CNNs)** and **Transfer Learning** with pre-trained models to enhance performance. The goal is to build a robust image classification model using TensorFlow and compare various deep learning architectures and deploying a Streamlit application to predict fish categories from user-uploaded images.

## 📌 Project Overview
- **Dataset:** Fish images belonging to multiple classes.
- **Preprocessing:** Rescaling images, applying data augmentation (rotation, flipping, zoom-in).
- **Model Training:**
  - **CNN from Scratch**
  - **Pre-trained Models:** VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix.
- **Best Model Selection:** The highest-performing model is saved in `.h5` or `.pkl` format for future use.

## 🚀 Setup and Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/artikwh/multiclass-fish-classification.git
   ```
2. **Train the models (Optional if pre-trained models are available):**
   ```bash
   python train_models.py
   ```
3. **Run the Streamlit app:** 
   ```
   streamlit run app.py
   ```
## 📊 Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|---------|-----------|--------|---------|
| CNN from Scratch | XX% | XX% | XX% | XX% |
| VGG16 | XX% | XX% | XX% | XX% |
| ResNet50 | XX% | XX% | XX% | XX% |
| MobileNet | XX% | XX% | XX% | XX% |
| InceptionV3 | XX% | XX% | XX% | XX% |
| EfficientNetB0 | XX% | XX% | XX% | XX% |

## 🔍 Results & Insights
- Transfer learning models **outperformed** the CNN trained from scratch.
- **Data augmentation** improved generalization and model robustness.
- **Confusion matrices** provided insights into misclassifications and dataset challenges.

## 🤝 Contributing
Feel free to fork this repository and submit **pull requests**! Any contributions, issues, or suggestions are welcome. 😊

## 📬 Contact
**Arti Kushwaha**  
📧 Email: [artikwh@gmail.com](mailto:artikwh@gmail.com)  
🔗 LinkedIn: [Arti Kushwaha](https://www.linkedin.com/in/arti-kushwaha-32a68634/)  

---

⭐ If you found this project helpful, please **star** this repository!

#DeepLearning #MachineLearning #ImageClassification #TensorFlow #AI #DataScience #TransferLearning 🚀
