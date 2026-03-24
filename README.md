# InstaFlore

A Flutter mobile application that identifies flowers in real-time using on-device AI — no internet connection required.

---

## Features

- **Live Camera Detection** — Point your camera at a flower and get instant identification with a confidence score
- **Photo Gallery Analysis** — Pick an image from your gallery and analyze it with one tap
- **Fully Offline** — The model runs entirely on-device via TensorFlow Lite
- **5 Flower Classes** — Daisy, Dandelion, Rose, Sunflower, Tulip
- **Onboarding Flow** — 3-step introduction carousel on first launch

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| Mobile Framework | Flutter 3.5+ / Dart |
| On-device Inference | TensorFlow Lite (`tflite_flutter`) |
| Camera Stream | `camera` package |
| Gallery Picker | `image_picker` |
| ML Model | MobileNetV2 (Transfer Learning) |
| Training | Python / TensorFlow / Keras |

---

## Model

The classifier is based on **MobileNetV2** pre-trained on ImageNet, fine-tuned on the TensorFlow Flowers dataset (~3,670 images).

### Training pipeline

1. **Phase 1** — Base model frozen, classification head trained (20 epochs, lr=1e-3)
2. **Phase 2** — Fine-tuning from layer 100 onward (30 epochs, lr=5e-5)

Techniques used: Data augmentation, Dropout, L2 regularization, EarlyStopping, ReduceLROnPlateau

**Test accuracy: 94%**

### Image preprocessing pipeline

```text
CameraImage (YUV420 / BGRA8888)
    → RGB Conversion
    → Resize 224×224
    → Normalization ×2.0 − 1.0  →  Float32List [1×224×224×3]
    → TFLite Inference
```

The normalization step (`value × 2.0 − 1.0`) maps pixel values to `[−1, 1]`, as required by MobileNetV2.

---

## Project Structure

```text
lib/
├── core/
│   └── widgets/               # Shared UI components
└── features/
    ├── startup/               # App entry, onboarding check
    ├── onboarding/            # First-launch carousel
    ├── home/                  # Main screen
    ├── flower_detection/      # Real-time camera detection
    │   ├── domain/entities/   # FlowerPrediction model
    │   └── data/services/     # TFLite service + preprocessor
    └── photo_detection/       # Gallery-based detection

assets/
├── model_clustered.tflite     # TFLite model (float32)
└── labels.txt                 # 5 class names
```

---

## Getting Started

**Prerequisites:**

- Flutter SDK `>=3.5.0`
- Android API 21+ or iOS 12+

**Run the app:**

```bash
flutter pub get
flutter run
```

**Build release APK:**

```bash
flutter build apk --release
```

---

## Supported Platforms

| Platform | Status |
| --- | --- |
| Android | Supported |
| iOS | Supported |
