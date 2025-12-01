# Wildlife Distance Calculator

A Python application for estimating the distance of wildlife from camera trap images using monocular depth estimation (DPT) and machine learning.

## Workflow Flowchart

```mermaid
graph TD
    subgraph "Phase 1: Annotation & Training"
        A[Load Images] --> B[Annotate Wildlife]
        B -->|Draw Polygons| C[Input Real Distance]
        C --> D[Train Model]
        D -->|Extract Depth Features| E[DPT Depth Estimation]
        E --> F[Train Regression Model]
        F --> G[Save Model (.keras) & Scaler (.joblib)]
    end

    subgraph "Phase 2: Distance Calculation"
        H[Load New Images] --> I{Auto-Load Model?}
        I -->|Yes| J[Load .keras & .joblib]
        I -->|No| K[Manual Load]
        K --> J
        J --> L[Auto-Calculate All]
        L -->|YOLOv5| M[Detect Wildlife]
        M -->|DPT| N[Generate Depth Map]
        N --> O[Predict Distance]
        O --> P[Display Results & Export CSV]
    end
```

## Features

*   **Modern UI**: "Liquid Glass" theme with intuitive controls.
*   **Annotation Tool**:
    *   Polygon annotation for precise object selection.
    *   Integrated DPT depth map visualization.
    *   Model training with performance metrics.
*   **Distance Calculator**:
    *   **Auto-Detection**: Uses YOLOv5 to automatically detect animals.
    *   **Auto-Calculation**: Automatically estimates distance for all detections.
    *   **Auto-Loading**: Automatically loads model and scaler files from the image directory.
    *   Export results to CSV.

## Installation

1.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `requirements.txt` includes: `PyQt5`, `opencv-python`, `numpy`, `tensorflow`, `scikit-learn`, `transformers`, `torch`, `matplotlib`, `ultralytics`, `pandas`, `seaborn`)*

## Usage

1.  **Run the Launcher**:
    ```bash
    python3 launcher.py
    ```

2.  **Annotate & Train**:
    *   Open "Annotate & Train Tool".
    *   Load a directory of images.
    *   Draw polygons around animals (Left-click points, Double-click to finish).
    *   Enter the known real distance.
    *   Click "Train Model" to generate your `.keras` model and `.joblib` scaler.

3.  **Calculate Distances**:
    *   Open "Distance Calculator".
    *   Load a directory of new images (ensure your model files are in the same folder for auto-loading).
    *   Click "Auto-Calculate All" to detect animals and get their distances.
