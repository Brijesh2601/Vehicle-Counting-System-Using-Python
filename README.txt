# 🚗 Vehicle Counting System using Python & OpenCV

A computer vision-based application that detects and counts vehicles in real-time from video streams. This project leverages OpenCV techniques to process frames, identify moving objects, and maintain an accurate vehicle count across a defined region.

---

## 📌 Overview

The Vehicle Counting System is designed to automate traffic monitoring by detecting vehicles in video footage and counting them as they cross a predefined line or region. This can be useful for:

* Traffic analysis and management
* Smart city applications
* Parking and toll systems
* Surveillance and monitoring

---

## ✨ Features

* 🎯 Real-time vehicle detection and counting
* 📹 Supports video file input (can be extended to live camera feed)
* 📊 Frame-by-frame processing using OpenCV
* 🚦 Region/line-based counting logic
* ⚡ Lightweight and easy to run

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** OpenCV, NumPy
* **Concepts Used:**

  * Computer Vision
  * Object Detection (basic contour-based / motion detection)
  * Image Processing

---

## 📂 Project Structure

```
vehicle-counter/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/Brijesh2601/Vehicle-Counting-System-Using-Python.git
cd Vehicle-Counting-System-Using-Python
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the project

```
python main.py
```


---

## 🧠 How It Works

1. The video stream is captured frame by frame
2. Frames are converted to grayscale and processed
3. Motion detection / contour detection is applied
4. Vehicles are identified as moving objects
5. A virtual line/region is defined
6. Vehicles are counted when they cross the line

---

## 🚀 Future Improvements

* 🔍 Integrate deep learning models (YOLO, SSD) for better accuracy
* 📈 Add analytics dashboard (vehicle trends, graphs)
* 🌐 Deploy as a web application
* 🎥 Support live CCTV / IP camera feeds
* 🧠 Multi-class vehicle detection (car, truck, bike, etc.)

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repository and submit pull requests.

---

## 📧 Contact

**Brijesh Sheladiya**
GitHub: https://github.com/Brijesh2601

---

## ⭐ Acknowledgements

* OpenCV community
* Python ecosystem contributors

---

## 📜 License

This project is open-source and available under the MIT License.
