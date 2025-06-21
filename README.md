# 🚢 Naval Ship Detection Using YOLOv5

This project uses the YOLOv5 deep learning model to detect naval ships from aerial and satellite images. It was built as part of an internship project using the [ShipRSImageNet.v39i](https://captain-whu.github.io/RSImageNet/) dataset.

## 🔗 Live Demo
Check out the live app here:  
👉 [Naval Ship Detection on Streamlit](https://naval-ship-detection-dd8yvnzbepd4ezw2uf5e8g.streamlit.app/)

---

## 🧠 Model
- **Architecture:** YOLOv5s
- **Dataset:** ShipRSImageNet.v39i (VOC format)
- **Framework:** PyTorch
- **mAP@0.5:** _Your result_  
- **Precision & Recall:** _Your values_

---

## 🚀 Features
- Upload ship images (.jpg/.png)
- View real-time detection results
- View model performance metrics

---

## 🛠️ Installation & Running Locally

```bash
git clone https://github.com/Manvitha0704/naval-ship-detection.git
cd naval-ship-detection
pip install -r requirements.txt
streamlit run app.py
