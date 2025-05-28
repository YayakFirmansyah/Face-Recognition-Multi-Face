# 🎯 Face Recognition System
**Real-time Face Recognition using MTCNN + FaceNet + SVM**

## 📋 Features
- **MTCNN**: Face detection & alignment
- **FaceNet**: 128-dim feature extraction  
- **SVM**: Optimized classification
- **Real-time**: Webcam recognition
- **High Accuracy**: 80-95% with optimizations

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
```
dataset/
├── person1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── person2/
│   ├── photo1.jpg
│   └── photo2.jpg
└── person3/
    └── photo1.jpg
```

### 3. Run System
```bash
python main.py
```

### 4. Quick Demo
```bash
python main.py
# Choose: 2 (Quick Demo)
# Enter dataset path
# System will auto: Load → Train → Evaluate → Real-time
```

## 📁 File Structure
```
face-recognition-system/
├── main.py                 # Main application
├── face_detector.py        # MTCNN detection
├── feature_extractor.py    # FaceNet features
├── face_classifier.py     # SVM classification
├── dataset_loader.py      # Dataset processing
├── realtime_recognizer.py # Real-time recognition
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🔧 Key Optimizations
- **Data Augmentation**: 4x more training data
- **Dataset Balancing**: Equal samples per person
- **L2 Normalization**: Consistent embeddings
- **Grid Search**: Optimal SVM parameters
- **Enhanced Preprocessing**: Better face quality

## 🎯 Usage Tips
1. **Dataset Quality**: Use 8-15 clear photos per person
2. **Lighting**: Ensure good lighting in photos
3. **Angles**: Include different face angles
4. **Resolution**: Minimum 160x160 face size
5. **Training**: Use optimization for best results

## 📊 Expected Performance
- **Accuracy**: 80-95% (with optimizations)
- **Speed**: ~10-15 FPS real-time
- **Memory**: ~500MB RAM usage

## 🔍 Troubleshooting
- **Low accuracy**: Enable optimization in training
- **Slow performance**: Reduce camera resolution
- **No faces detected**: Check lighting & camera quality
- **Import errors**: Verify all dependencies installed