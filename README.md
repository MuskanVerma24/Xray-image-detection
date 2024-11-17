# **X-ray Image Recognition 🩻**

**An AI-powered solution for classifying bone fractures in X-ray images using deep learning 🤖.**

---

## **Overview** 💡
This project leverages the power of deep learning to **automatically classify X-ray images** into two categories: **fractured** or **not fractured**. Using state-of-the-art machine learning techniques, the model achieved an impressive **98.73% accuracy** on test data, making it a reliable tool for medical image analysis and diagnosis.

---

## **Features** ✨
- **🔍 High Accuracy**: Achieved **98.73% accuracy** on test data, ensuring reliable predictions.
- **⚖️ Binary Classification**: Efficiently classifies X-ray images as either **fractured** or **not fractured**.
- **🖥️ User-Friendly**: Simple interface to upload and classify X-ray images.

---

## **Installation Guide** 🛠️

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MuskanVerma24/Xray-Image-Recognition.git
   cd Xray-Image-Recognition
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## **How to Use** 🚀

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Upload an X-ray image** and the model will classify it as **fractured** or **not fractured**.

3. **View the result** directly in the application interface.

---

## Dataset Information

The model was trained on the **Bone Fracture Multi-Region X-ray Data** dataset. Below are the details:

- **Dataset Name**: Bone Fracture Dataset 
- **Source**: [Kaggle Bone Fracture Dataset](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data) 📥
- **Download Link**: [Dataset Download](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data) ⬇️
- **Size**: Approximately 505 MB 
- **License**: [PDDL License 1.0](https://opendatacommons.org/licenses/pddl/1-0/) 

### Instructions
1. Download the dataset from the link above. 
2. Extract the files to the `data` folder of the project. 🗂️
3. The data is divided into 'fractured' and 'non-fractured' categories. 

---


## **Model Performance** 📈
The model was trained on a **Bone Fracture Binary Classification dataset** and achieved an impressive **98.73% accuracy** on the test set, showcasing its strong ability to detect fractures in X-ray images. The model uses deep learning techniques built with **TensorFlow** and **Keras** to ensure high precision and reliability.


---


## **Acknowledgments** 🙏
- **TensorFlow/Keras** for providing the deep learning framework 🧠.
- **Bone Fracture Binary Classification Dataset** for enabling the training and testing of the model 📊.
