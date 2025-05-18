# Emotion-Based Egyptian Music Player

![GitHub stars](https://img.shields.io/github/stars/FathyTamerMobileDevelopment/emotion-music-player?style=social)

> **Experience music that matches your mood!** This advanced application detects your facial emotions in real-time and automatically plays Egyptian music that complements your emotional state.

## 📸 Application Demo

<div align="center">
  <kbd>
    <img src="docs/images/app_demo.png" alt="Application Demo" width="600"/>
  </kbd>
  <p><i>The main interface of the Emotion-Based Egyptian Music Player, showing real-time emotion detection and music selection</i></p>
</div>

## 🚀 Key Features

- **Real-time Facial Emotion Detection**: Accurately detects 7 emotions (happy, sad, angry, surprised, neutral, fear, disgust)
- **Automatic Egyptian Music Playback**: Plays songs that match your detected emotion
- **Multi-threaded Model Training**: Parallel processing to enhance performance (user-defined thread count)
- **High Confidence Tracking**: Visual emotion confidence meter
- 🎚**Customizable Interface**: Volume control and easy-to-use controls

## Technical Architecture

<div align="center">
  <kbd>
    <img src="docs/images/architecture.png" alt="System Architecture" width="700"/>
  </kbd>
  <p><i>The system architecture showing the integration of emotion detection, music playback, and UI components</i></p>
</div>

## Threaded Training Performance

Our system implements state-of-the-art multi-threaded training to enhance performance:

<div align="center">
  <kbd>
    <img src="docs/images/training_performance.png" alt="Threaded Training Performance" width="650"/>
  </kbd>
  <p><i>Training time comparison with different thread counts, showing significantly improved performance with multi-threading</i></p>
</div>

## The Emotion Detection Model

The emotion detection model employs a CNN architecture enhanced with BatchNormalization and Dropout layers:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')
])
```

## Multi-threaded Training

<div align="center">
  <kbd>
    <img src="docs/images/thread_selection.png" alt="Thread Selection Interface" width="500"/>
  </kbd>
  <p><i>User interface for selecting the number of threads for model training</i></p>
</div>

Our application introduces an innovative approach to emotion model training:

- **User-Defined Thread Selection**: Choose how many CPU cores to utilize
- **Parallel Data Processing**: Distributes preprocessing across multiple threads
- **Distributed Model Training**: Divides the training process for faster results
- **Adaptive Resource Utilization**: Automatically detects system capabilities

## Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- CustomTkinter
- NumPy
- PyGame
- PIL (Pillow)

### Quick Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-music-player.git
cd emotion-music-player
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your Egyptian music files:
   - Place happy Egyptian songs in `data/music/happy/`
   - Place sad Egyptian songs in `data/music/sad/`
   - Place neutral Egyptian songs in `data/music/neutral/`
   - Place angry Egyptian songs in `data/music/angry/`
   - Place surprised Egyptian songs in `data/music/surprised/`

## Usage

### Running the Application

```bash
python main.py
```

### Training the Emotion Model

There are multiple ways to specify the number of threads for training:

1. **Command-line argument**:
```bash
python main.py --train --threads 8
```

2. **Interactive GUI prompt**:
```bash
python main.py --train --gui
```

3. **Default interactive prompt**:
```bash
python main.py --train
```

## 📁 Project Structure

```
emotion-music-player/
│
├── data/
│   └── music/           # Music organized by emotion (happy, sad, etc.)
│
├── models/              # Trained emotion detection models
│
├── src/
│   ├── emotion_detector.py  # Emotion detection module
│   ├── music_player.py      # Music playback module
│   └── ui_manager.py        # User interface module
│
├── utils/
│   └── helpers.py           # Utility functions
│
├── docs/
│   └── images/              # Documentation images
│
├── threaded_emotion_training.py  # Multi-threaded training implementation
├── main.py                  # Main application entry point
└── README.md                # Project documentation
```

## 💡 How It Works

1. The webcam captures your facial expressions in real-time
2. Our CNN-based model analyzes your expression and classifies the emotion
3. The application selects appropriate Egyptian music matching your detected emotion
4. The music automatically changes when a new emotion is detected with high confidence

## 🎻 Music Selection Logic

The application maps detected emotions to music categories:
- 😊 Happy → Upbeat Egyptian music
- 😢 Sad → Melancholic Egyptian tunes
- 😐 Neutral → Moderate tempo classical Egyptian music
- 😠 Angry → Powerful, energetic Egyptian pieces
- 😲 Surprised → Intriguing Egyptian compositions
- 😨 Fear → Mapped to sad Egyptian music
- 🤢 Disgust → Mapped to angry Egyptian music

## Performance Analysis

- **Detection Speed**: ~30 FPS on mid-range hardware
- **Emotion Accuracy**: ~70% in real-world conditions
- **Training Performance**: Scales almost linearly with thread count

## Contributors

- [FathyTamerMobileDevelopment]([https://github.com/yourusername](https://github.com/FathyTamerMobileDevelopment)) - Main Developer

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [FER2013 Dataset](https://www.kaggle.com/msambare/fer2013) for emotion training data
- [TensorFlow](https://www.tensorflow.org/) for the machine learning framework
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for modern UI elements
- Special thanks to the Egyptian music community for inspiration

---

<div align="center">
  <p>Made with ❤️ in Egypt</p>
  <p>For academic purposes - Cairo University Computer Engineering</p>
</div>
