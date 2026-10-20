# Bundled ML models

All files are unmodified TFLite models published by Google's MediaPipe team under the Apache License 2.0 and run
locally through `ai-edge-litert` (nothing is uploaded anywhere):

| File | Purpose | Source |
|---|---|---|
| `face_detector.tflite` | BlazeFace short-range face detector | face_landmarker.task bundle |
| `face_landmarks_detector.tflite` | Face Mesh, 478 landmarks (lips, eyes, jaw) | face_landmarker.task bundle |
| `selfie_multiclass_256x256.tflite` | Person segmentation (hair, skin, clothes vs background) | image_segmenter/selfie_multiclass_256x256 |

Sources: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
and https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
