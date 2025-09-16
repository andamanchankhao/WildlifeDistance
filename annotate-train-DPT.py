########################################################
### Dev : Andaman Chankhao andmanchankhao@gmail.com ####
########################################################

import sys
import os
import cv2
import json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGraphicsView, QGraphicsScene, QInputDialog, QFileDialog, QMessageBox, QDialog, QSizePolicy,
    QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt, QPointF, QRectF, QThread, pyqtSignal, QTemporaryFile, QSize

# Import TensorFlow and Keras for model training
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model # type: ignore
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Input # type: ignore
    from tensorflow.keras.callbacks import Callback # type: ignore
    from tensorflow.keras.metrics import MeanAbsoluteError # Import MAE metric # type: ignore
    # MobileNetV2 is removed as we're replacing it for DPT-based approach
except ImportError:
    tf = None
    print("TensorFlow not installed. Please install it with 'pip install tensorflow' to enable training.")

# Import scikit-learn for additional metrics (RMSE, R-squared)
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split # For splitting data
    from sklearn.preprocessing import StandardScaler # For scaling features
except ImportError:
    mean_squared_error = None
    mean_absolute_error = None
    r2_score = None
    train_test_split = None
    StandardScaler = None
    print("Scikit-learn not installed. Install with 'pip install scikit-learn' for full metrics report.")

# Import joblib for saving/loading scaler
try:
    import joblib
except ImportError:
    joblib = None
    print("Joblib not installed. Install with 'pip install joblib' to save/load the scaler.")


# Import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    # Use 'Agg' backend for non-interactive plotting, which is suitable for PyQt integration
    plt.switch_backend('Agg')
except ImportError:
    plt = None
    print("Matplotlib not installed. Install with 'pip install matplotlib' to generate plots.")

# Import DPT related libraries
try:
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    import torch
    print("Hugging Face Transformers and PyTorch installed.")
except ImportError:
    DPTForDepthEstimation = None
    DPTImageProcessor = None
    torch = None
    print("Hugging Face Transformers or PyTorch not installed. Install with 'pip install transformers torch' to enable DPT.")
    print("Note: PyTorch is required for DPT models from Hugging Face.")


# Custom Keras Callback to update PyQt GUI
class PyQtCallback(Callback):
    def __init__(self, update_status_signal):
        super().__init__()
        self.update_status_signal = update_status_signal

    def on_epoch_end(self, epoch, logs=None):
        # Report both loss (MSE) and MAE
        self.update_status_signal.emit(
            f"Epoch {epoch+1} finished. Loss (MSE): {logs['loss']:.4f}, MAE: {logs.get('mean_absolute_error', 'N/A'):.4f}, "
            f"Val Loss (MSE): {logs.get('val_loss', 'N/A'):.4f}, Val MAE: {logs.get('val_mean_absolute_error', 'N/A'):.4f}"
        )

# New class for the plot window (for Actual vs. Predicted)
class PlotWindow(QDialog):
    def __init__(self, plot_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Actual vs. Predicted Distance Plot")
        self.setGeometry(200, 200, 800, 600) # Set a default size for the plot window

        layout = QVBoxLayout(self)

        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        # Make the label expand to fill available space
        self.plot_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.plot_label)

        self.plot_path = plot_path # Store plot_path for resizeEvent and showEvent

    def load_plot(self, plot_path):
        """Loads and scales the plot image to fit the label."""
        print(f"DEBUG(PlotWindow): Attempting to load plot from: {plot_path}")
        if plot_path and os.path.exists(plot_path):
            original_pixmap = QPixmap(plot_path)
            if not original_pixmap.isNull():
                label_size = self.plot_label.size()
                print(f"DEBUG(PlotWindow): QLabel size for scaling: {label_size.width()}x{label_size.height()}")
                if label_size.width() > 0 and label_size.height() > 0:
                    scaled_pixmap = original_pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.plot_label.setPixmap(scaled_pixmap)
                    print("DEBUG(PlotWindow): Plot pixmap set successfully.")
                else:
                    self.plot_label.setText("Error: Plot label size is zero. Cannot scale image.")
                    print("ERROR(PlotWindow): Plot label size is zero. Cannot scale image.")
            else:
                self.plot_label.setText(f"Error: Could not load plot image from {plot_path}")
                print(f"ERROR(PlotWindow): QPixmap failed to load plot from {plot_path}. Is file corrupted or path wrong?")
        else:
            self.plot_label.setText("Error: Plot image path is invalid or file does not exist.")
            print(f"ERROR(PlotWindow): Plot image path is invalid or file does not exist: {plot_path}")

    def showEvent(self, event):
        """Called when the widget is shown. Load plot here to ensure label size is correct."""
        print("DEBUG(PlotWindow): showEvent triggered.")
        self.load_plot(self.plot_path)
        super().showEvent(event)

    def resizeEvent(self, event):
        """Re-scale the pixmap when the window is resized."""
        print("DEBUG(PlotWindow): resizeEvent triggered.")
        # Only reload if a pixmap is already set, otherwise it might try to load before showEvent
        if self.plot_label.pixmap():
            self.load_plot(self.plot_path)
        super().resizeEvent(event)

# New class for displaying DPT depth map
class DepthMapWindow(QDialog):
    def __init__(self, depth_map_array, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DPT Depth Map")
        self.setGeometry(250, 250, 700, 500) # Default size for depth map window

        layout = QVBoxLayout(self)
        self.depth_map_label = QLabel()
        self.depth_map_label.setAlignment(Qt.AlignCenter)
        self.depth_map_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.QSizePolicy.Expanding)
        layout.addWidget(self.depth_map_label)

        self.depth_map_array = depth_map_array # Store the numpy array

    def load_depth_map(self):
        """Loads and scales the depth map array to fit the label."""
        if self.depth_map_array is None:
            self.depth_map_label.setText("No depth map data available.")
            return

        # Normalize the depth map to 0-255 for visualization as a grayscale image
        # Ensure it's float32 for min/max operations, then convert to uint8
        normalized_depth = self.depth_map_array.astype(np.float32)
        if normalized_depth.max() - normalized_depth.min() > 0:
            normalized_depth = (normalized_depth - normalized_depth.min()) / \
                               (normalized_depth.max() - normalized_depth.min()) * 255
        else: # Handle case where all values are the same (flat depth)
            normalized_depth = np.zeros_like(normalized_depth) * 255 # All white or black

        normalized_depth = normalized_depth.astype(np.uint8)

        h, w = normalized_depth.shape
        # Create QImage from grayscale NumPy array
        q_image = QImage(normalized_depth.data, w, h, w, QImage.Format_Grayscale8)
        original_pixmap = QPixmap.fromImage(q_image)

        if not original_pixmap.isNull():
            label_size = self.depth_map_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_pixmap = original_pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.depth_map_label.setPixmap(scaled_pixmap)
            else:
                self.depth_map_label.setText("Error: Depth map label size is zero. Cannot scale image.")
        else:
            self.depth_map_label.setText("Error: Could not convert depth map to QPixmap.")

    def showEvent(self, event):
        """Called when the widget is shown. Load depth map here to ensure label size is correct."""
        self.load_depth_map()
        super().showEvent(event)

    def resizeEvent(self, event):
        """Re-scale the pixmap when the window is resized."""
        self.load_depth_map()
        super().resizeEvent(event)

# Custom QWidget for displaying thumbnail with annotation status
class ThumbnailItemWidget(QWidget):
    def __init__(self, pixmap, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.is_annotated = False # Default status
        self.is_selected = False # New flag for selection status
        self.setFixedSize(QSize(100, 100)) # Fixed size for thumbnails

        layout = QVBoxLayout(self)
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        # Scale pixmap for the thumbnail label
        self.thumbnail_label.setPixmap(pixmap.scaled(90, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(self.thumbnail_label)
        layout.setContentsMargins(5, 5, 5, 5) # Small margins for internal content

        # Set transparent background for the widget itself initially
        self.setStyleSheet("background-color: transparent;")

    def set_annotated_status(self, annotated):
        """Sets the annotation status."""
        if self.is_annotated != annotated:
            self.is_annotated = annotated
            # No visual update here, as green border is removed.
            # self.update() # No need to trigger repaint just for internal flag

    def set_selected(self, selected):
        """Sets the selection status and updates the background color."""
        if self.is_selected != selected:
            self.is_selected = selected
            if self.is_selected:
                self.setStyleSheet("background-color: lightgray;") # Grey background when selected
            else:
                self.setStyleSheet("background-color: transparent;") # Transparent when not selected
            self.update() # Trigger repaint to apply style sheet changes

    def paintEvent(self, event):
        """Custom paint event. No green border drawing, but handles selection background."""
        super().paintEvent(event)
        # The background color is now handled by the stylesheet in set_selected method.
        # No custom painting for border is needed here.


# --- QThread for Training (to prevent UI freeze) ---
class TrainingThread(QThread):
    # Signal to update the status bar
    update_status = pyqtSignal(str)
    # Signal to indicate training completion
    training_finished = pyqtSignal()
    # New signal to send final metrics report to the UI
    final_metrics_report = pyqtSignal(str)
    # New signal to send plot image path to the UI
    plot_generated = pyqtSignal(str)

    def __init__(self, image_files, dpt_processor, dpt_model, device, image_directory, camera_id):
        super().__init__()
        self.image_files = image_files
        self.processor = dpt_processor
        self.dpt_model = dpt_model
        self.device = device
        self.image_directory = image_directory
        self.camera_id = camera_id


    def get_depth_for_bbox(self, depth_map, bbox_coords):
        """
        Extracts depth values for a given bounding box from a depth map
        and returns a representative distance (e.g., median).
        """
        x_min, y_min, x_max, y_max = bbox_coords

        # Ensure coordinates are within bounds of the depth map
        h, w = depth_map.shape
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        if x_max <= x_min or y_max <= y_min:
            return None # Invalid or empty bounding box after clamping

        # Crop the depth map to the bounding box region
        cropped_depth = depth_map[y_min:y_max, x_min:x_max]

        # Filter out any non-finite values (NaN, Inf) if they exist, though DPT usually outputs finite numbers.
        valid_depths = cropped_depth[np.isfinite(cropped_depth)]

        if valid_depths.size == 0:
            return None # No valid depth values found in the bounding box

        # Return the median depth as the representative distance.
        # Median is often more robust to outliers than mean for depth estimation.
        return np.median(valid_depths)

    def run(self):
        # Check if all necessary libraries are installed
        if tf is None:
            self.update_status.emit("TensorFlow is not installed. Cannot perform training.")
            self.training_finished.emit()
            return
        if mean_squared_error is None or train_test_split is None or StandardScaler is None:
            self.update_status.emit("Scikit-learn (for metrics, splitting, or scaling) is not installed. Full accuracy metrics and data splitting/scaling will not be available.")
            # Continue execution, but with limited metrics/features
        if joblib is None:
            self.update_status.emit("Joblib is not installed. Scaler will not be saved.")
            # Continue execution, but scaler won't be saved
        if plt is None:
            self.update_status.emit("Matplotlib is not installed. Prediction plot will not be generated.")
            # Continue execution, but without plotting
        if self.processor is None or self.dpt_model is None:
            self.update_status.emit("DPT model is not loaded. Cannot proceed with DPT-based training.")
            self.training_finished.emit()
            return

        self.update_status.emit("Training started... This might take a while.")
        try:
            # 1. Load all annotations and corresponding image data
            all_data = []
            for img_path in self.image_files:
                annotation_filename = img_path + ".json"
                if os.path.exists(annotation_filename):
                    try:
                        with open(annotation_filename, 'r') as f:
                            annotations = json.load(f)
                            for ann in annotations:
                                all_data.append({
                                    "image_path": img_path,
                                    "annotation": ann
                                })
                    except json.JSONDecodeError:
                        self.update_status.emit(f"Error reading JSON from {annotation_filename}. Skipping.")
                        continue

            if not all_data:
                self.update_status.emit("No annotations found to train on. Please annotate some images first.")
                self.training_finished.emit()
                return

            self.update_status.emit(f"Collected {len(all_data)} annotations for training.")

            # 2. Prepare data by extracting DPT-derived features
            X_features = [] # Features (DPT-derived depth)
            y_labels = []   # Labels (annotated distances)

            # Cache to store depth maps to avoid re-processing the same image multiple times
            processed_images = {} 

            for item in all_data:
                img_path = item["image_path"]
                annotation = item["annotation"]

                if annotation["type"] == "bounding_box":
                    coords = annotation["coordinates"]
                    distance = annotation["distance_meters"]

                    # Get Depth Map using DPT model
                    depth_map = None
                    if img_path in processed_images:
                        depth_map = processed_images[img_path]
                    else:
                        img_cv_full = cv2.imread(img_path)
                        if img_cv_full is None:
                            self.update_status.emit(f"Could not load image for DPT inference: {img_path}")
                            continue

                        # CROP THE IMAGE to top 92.593%
                        h, _, _ = img_cv_full.shape
                        crop_h = int(h * 0.92593)
                        img_cv = img_cv_full[0:crop_h, :]
                        
                        # Convert BGR (OpenCV default) to RGB (DPT model expects RGB)
                        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                        # Process image with DPT processor and move to the appropriate device (CPU/GPU)
                        inputs = self.processor(images=img_rgb, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()} # Move tensors to device

                        with torch.no_grad(): # Disable gradient calculation for inference to save memory and speed up
                            outputs = self.dpt_model(**inputs)
                            # The 'predicted_depth' tensor contains the depth map.
                            # .squeeze() removes singleton dimensions, .cpu() moves to CPU, .numpy() converts to NumPy array.
                            predicted_depth_unscaled = outputs.predicted_depth.squeeze().cpu().numpy()

                            # Resize the depth map to the original cropped image dimensions.
                            # This is crucial for accurate bounding box extraction later.
                            # INTER_AREA is good for shrinking, INTER_LINEAR for enlarging.
                            depth_map = cv2.resize(predicted_depth_unscaled, (img_cv.shape[1], img_cv.shape[0]), interpolation=cv2.INTER_AREA)

                            processed_images[img_path] = depth_map # Cache the generated depth map

                    if depth_map is None:
                        self.update_status.emit(f"Skipping {img_path} due to DPT depth map generation issue.")
                        continue

                    # Extract representative depth from the bounding box in the depth map
                    bbox_depth = self.get_depth_for_bbox(depth_map, coords)

                    if bbox_depth is not None:
                        X_features.append([bbox_depth]) # Wrap in a list as it's a single feature for the regression model
                        y_labels.append(distance)
                    else:
                        self.update_status.emit(f"Skipping annotation for {img_path} at {coords} due to invalid depth extraction from bbox.")
                        continue

            if not X_features:
                self.update_status.emit("No valid features extracted from DPT. Check annotations or image processing.")
                self.training_finished.emit()
                return

            X_features = np.array(X_features, dtype=np.float32)
            y_labels = np.array(y_labels, dtype=np.float32)

            self.update_status.emit(f"Prepared {len(X_features)} samples with DPT-derived features for regression training.")

            # Split data into training and validation sets if scikit-learn is available
            X_train_reg, X_val_reg, y_train_reg, y_val_reg = np.array([]), np.array([]), np.array([]), np.array([])
            if train_test_split and len(X_features) > 1:
                X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
                self.update_status.emit(f"Split data: {len(X_train_reg)} training, {len(X_val_reg)} validation samples.")
            else:
                X_train_reg, y_train_reg = X_features, y_labels
                self.update_status.emit("Scikit-learn not available or not enough data, training on full dataset without validation split.")


            # Scale features (important for neural networks to converge better)
            scaler = None
            if StandardScaler:
                scaler = StandardScaler()
                X_train_reg = scaler.fit_transform(X_train_reg)
                if X_val_reg.size > 0: # Only transform validation set if it exists
                    X_val_reg = scaler.transform(X_val_reg)
                self.update_status.emit("Features scaled using StandardScaler.")

                # Save the fitted scaler
                if joblib:
                    # UPDATED: Save scaler inside the image directory with camera name
                    scaler_filename = f"{self.camera_id}_scaler.joblib"
                    scaler_save_path = os.path.join(self.image_directory, scaler_filename)
                    joblib.dump(scaler, scaler_save_path)
                    self.update_status.emit(f"Scaler saved successfully to: {scaler_save_path}")
                else:
                    self.update_status.emit("Joblib not available, scaler will not be saved.")
            else:
                self.update_status.emit("Scikit-learn StandardScaler not available, features will not be scaled.")

            # Define a simple Keras regression model to map DPT depth to actual distance
            # This model is much simpler as DPT already provides the complex feature extraction.
            # It takes the single DPT-derived depth feature and predicts the actual distance.
            
            inputs_reg = Input(shape=(X_train_reg.shape[1],)) # Input shape is (number of features,) - here, 1
            x = Dense(64, activation='relu')(inputs_reg) # First hidden layer
            x = Dense(32, activation='relu')(x) # Second hidden layer
            outputs_reg = Dense(1)(x) # Output layer for single distance regression

            regression_model = Model(inputs_reg, outputs_reg)

            # Compile the regression model with Adam optimizer and Mean Squared Error loss.
            # Mean Absolute Error is included as an additional metric for interpretability.
            regression_model.compile(optimizer='adam', loss='mse', metrics=[MeanAbsoluteError()])

            self.update_status.emit("Starting regression model training (DPT features -> Actual Distance)...")

            # Create a custom callback to update the PyQt GUI during training epochs
            pyqt_callback = PyQtCallback(self.update_status)

            # Train the regression model
            history = regression_model.fit(X_train_reg, y_train_reg,
                                           epochs=50, # Number of training epochs (can be adjusted)
                                           batch_size=8, # Batch size for training (can be adjusted)
                                           validation_data=(X_val_reg, y_val_reg) if X_val_reg.size > 0 else None, # Use validation data if available
                                           callbacks=[pyqt_callback], # Pass the custom callback
                                           verbose=0) # Set verbose to 0 to prevent Keras from printing to console directly

            self.update_status.emit("Regression model training complete.")

            # Evaluate and report final metrics on the full dataset
            # Ensure predictions are made on the scaled features
            X_all_scaled = scaler.transform(X_features) if scaler else X_features
            y_pred = regression_model.predict(X_all_scaled).flatten() # Get predictions and flatten to 1D array

            report_message = "--- Training Metrics Report (DPT-based Regression) ---"
            if mean_absolute_error is not None:
                final_mae = mean_absolute_error(y_labels, y_pred)
                report_message += f"\nOverall Mean Absolute Error (MAE): {final_mae:.4f} meters"
            if mean_squared_error is not None:
                final_mse = mean_squared_error(y_labels, y_pred)
                final_rmse = np.sqrt(final_mse)
                report_message += f"\nOverall Root Mean Squared Error (RMSE): {final_rmse:.4f} meters"
            if r2_score is not None:
                final_r2 = r2_score(y_labels, y_pred)
                report_message += f"\nOverall R-squared (R²): {final_r2:.4f}"

            self.final_metrics_report.emit(report_message) # Emit the full report to the UI
            print(report_message) # Also print to console for easier review/debugging

            # Generate and save the prediction vs. actual plot
            if plt is not None:
                if len(y_labels) == 0 or len(y_pred) == 0:
                    self.update_status.emit("Not enough data to generate plot. Need at least one annotation.")
                    print("DEBUG: No data for plotting (y_labels or y_pred is empty).")
                else:
                    try:
                        # Create a temporary file for the plot image
                        temp_file = QTemporaryFile()
                        # Set the file template directly. QTemporaryFile will handle the unique part.
                        temp_file.setFileTemplate("plot_XXXXXX.png")
                        if temp_file.open():
                            plot_path = temp_file.fileName()
                            temp_file.close() # Close the file so matplotlib can write to it
                            print(f"DEBUG: Plot saved to: {plot_path}")

                            plt.figure(figsize=(8, 6)) # Set figure size
                            plt.scatter(y_labels, y_pred, alpha=0.6) # Scatter plot of actual vs. predicted
                            # Determine plot limits based on data range to ensure all points are visible
                            min_val = min(y_labels.min(), y_pred.min()) if y_labels.size > 0 and y_pred.size > 0 else 0
                            max_val = max(y_labels.max(), y_pred.max()) if y_labels.size > 0 and y_pred.size > 0 else 1
                            # Plot a diagonal line representing perfect prediction
                            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                            
                            plt.xlabel("Actual Distance (meters)")
                            plt.ylabel("Predicted Distance (meters)")
                            plt.title(f"Actual vs. Predicted Distance (Camera: {self.camera_id})")
                            plt.grid(True) # Add a grid for better readability
                            plt.legend() # Show legend for the perfect prediction line
                            plt.tight_layout() # Adjust plot to prevent labels from overlapping
                            plt.savefig(plot_path) # Save the plot to the temporary file
                            plt.close() # Close the plot to free memory
                            print(f"DEBUG: Plot saved to: {plot_path}")

                            self.plot_generated.emit(plot_path) # Emit the path to the new window
                            self.update_status.emit("Prediction plot generated.")
                        else:
                            self.update_status.emit("Could not create temporary file for plot.")
                            print("ERROR: QTemporaryFile could not open.")
                    except Exception as plot_e:
                        self.update_status.emit(f"Error generating plot: {str(plot_e)}")
                        print(f"Plotting Exception: {plot_e}")

            # UPDATED: Save the trained regression model inside the image directory with camera name
            model_filename = f"{self.camera_id}_model.keras"
            model_save_path = os.path.join(self.image_directory, model_filename)
            regression_model.save(model_save_path)
            self.update_status.emit(f"Regression model saved successfully to: {model_save_path}")

        except Exception as e:
            self.update_status.emit(f"Training failed: {str(e)}")
            print(f"Training Exception: {e}") # Print full exception for debugging
        finally:
            self.training_finished.emit()

# Custom QGraphicsView to handle mouse events directly
class CustomGraphicsView(QGraphicsView):
    # Signals to be emitted when mouse events occur
    mouse_pressed = pyqtSignal(object)
    mouse_moved = pyqtSignal(object)
    mouse_released = pyqtSignal(object)

    def mousePressEvent(self, event):
        self.mouse_pressed.emit(event)
        # Call super's method to ensure default behavior (like focus) is preserved
        super().mousePressEvent(event) 

    def mouseMoveEvent(self, event):
        self.mouse_moved.emit(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.mouse_released.emit(event)
        super().mouseReleaseEvent(event)


class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wildlife Distance Annotation Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.current_image_path = None
        self.current_image_cv = None # This will store the CROPPED image
        self.annotations = [] # Annotations for the currently displayed image
        self.all_annotations_data = [] # All annotations loaded from all image files in the directory
        self.image_files = []
        self.current_image_index = -1
        self.is_dpt_view_active = False # New flag to track DPT view state
        
        # NEW: Store image directory and camera ID
        self.image_directory = None
        self.camera_id = None

        self.training_thread = None
        self.plot_window = None # Keep a reference to the plot window
        self.depth_map_window = None # Keep a reference to the depth map window

        # DPT Model components
        self.dpt_processor = None
        self.dpt_model = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"

        # Thumbnail list and map
        self.thumbnail_list_widget = QListWidget()
        self.image_item_map = {} # Maps image_path to ThumbnailItemWidget instance

        self.init_ui()
        self.load_dpt_inference_model() # Load DPT model on startup

    def load_dpt_inference_model(self):
        """Loads the pre-trained DPT model for inference."""
        if DPTForDepthEstimation is None or DPTImageProcessor is None or torch is None:
            self.status_label.setText("DPT dependencies not installed. Cannot load DPT model for inference.")
            return

        try:
            model_name = "Intel/dpt-hybrid-midas"
            self.dpt_processor = DPTImageProcessor.from_pretrained(model_name)
            self.dpt_model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
            self.dpt_model.eval() # Set model to evaluation mode
            self.status_label.setText(f"DPT model '{model_name}' loaded successfully on {self.device}.")
        except Exception as e:
            self.status_label.setText(f"Failed to load DPT model: {str(e)}. Check internet/dependencies.")
            print(f"DPT Inference Model Load Error: {e}")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout: QHBoxLayout to split into left (thumbnails), center (main image/controls), and right (table/metrics)
        main_h_layout = QHBoxLayout(central_widget)

        # Left Panel: Thumbnail List and Load Directory button
        left_v_layout = QVBoxLayout()
        
        # Moved "Load Directory" button to the top of the left panel
        self.load_dir_btn = QPushButton("Load Directory")
        self.load_dir_btn.clicked.connect(self.load_directory)
        left_v_layout.addWidget(self.load_dir_btn)

        self.thumbnail_list_widget.setFixedWidth(120) # Fixed width for the thumbnail panel
        self.thumbnail_list_widget.setIconSize(QSize(90, 90)) # Size for icons within list items
        self.thumbnail_list_widget.setResizeMode(QListWidget.Adjust) # Adjust items on resize
        self.thumbnail_list_widget.setViewMode(QListWidget.IconMode) # Display as icons (thumbnails)
        self.thumbnail_list_widget.setMovement(QListWidget.Static) # Items don't move
        self.thumbnail_list_widget.setSpacing(5) # Spacing between items
        # Connect item click to load image
        self.thumbnail_list_widget.itemClicked.connect(self._on_thumbnail_clicked)
        left_v_layout.addWidget(self.thumbnail_list_widget)
        main_h_layout.addLayout(left_v_layout)

        # Center Panel: Image Display with Navigation Arrows
        center_v_layout = QVBoxLayout()
        center_v_layout.setContentsMargins(0, 0, 0, 0) # Remove margins for tighter layout

        # Image display area with navigation arrows
        image_nav_layout = QHBoxLayout()
        image_nav_layout.setContentsMargins(0, 0, 0, 0) # Remove margins for tighter layout
        image_nav_layout.setSpacing(0) # Remove spacing between elements

        self.prev_btn = QPushButton("<") # Changed to arrow
        self.prev_btn.setFixedSize(30, 30) # Make it small and square
        self.prev_btn.clicked.connect(self.load_previous_image)
        image_nav_layout.addWidget(self.prev_btn)

        # Use CustomGraphicsView instead of QGraphicsView
        self.graphics_view = CustomGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setMouseTracking(True) # Enable mouse tracking for live coordinate display
        image_nav_layout.addWidget(self.graphics_view, 1) # Stretch factor 1 to make it dynamic

        # Connect custom signals from CustomGraphicsView to AnnotationTool's methods
        self.graphics_view.mouse_pressed.connect(self.mouse_press_event)
        self.graphics_view.mouse_moved.connect(self.mouse_move_event)
        self.graphics_view.mouse_released.connect(self.mouse_release_event)


        self.next_btn = QPushButton(">") # Changed to arrow
        self.next_btn.setFixedSize(30, 30) # Make it small and square
        self.next_btn.clicked.connect(self.load_next_image)
        image_nav_layout.addWidget(self.next_btn)
        
        center_v_layout.addLayout(image_nav_layout) # Add the image and nav layout to the center panel

        # Control buttons layout (remaining buttons)
        control_layout = QHBoxLayout()
        # "Save Annotations" button moved to right_v_layout
        
        self.train_btn = QPushButton("Train DPT Data") # Renamed for clarity
        self.train_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.train_btn)

        self.show_dpt_btn = QPushButton("Show DPT Image") # Initial text
        self.show_dpt_btn.clicked.connect(self.toggle_dpt_view) # Connect to new toggle method
        control_layout.addWidget(self.show_dpt_btn)

        center_v_layout.addLayout(control_layout)
        main_h_layout.addLayout(center_v_layout, 1) # Add center panel to main layout with stretch

        # Right Panel: Annotation Data Table and Metrics
        right_v_layout = QVBoxLayout()
        
        # Annotation Data Table
        self.annotation_table = QTableWidget()
        self.annotation_table.setColumnCount(3)
        self.annotation_table.setHorizontalHeaderLabels(["Image", "Coordinates", "Distance (m)"])
        self.annotation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch) # Stretch columns
        self.annotation_table.setEditTriggers(QTableWidget.NoEditTriggers) # Make table read-only
        right_v_layout.addWidget(self.annotation_table, 1) # Stretch factor 1 to make it dynamic

        # Buttons below the table
        table_buttons_layout = QHBoxLayout()
        
        # New "Delete Annotation" button
        self.delete_annotation_btn = QPushButton("Delete Annotation")
        self.delete_annotation_btn.clicked.connect(self.delete_selected_annotation)
        table_buttons_layout.addWidget(self.delete_annotation_btn)

        # Moved "Save Annotations" button here
        self.save_btn = QPushButton("Save Annotations")
        self.save_btn.clicked.connect(self.save_annotations)
        table_buttons_layout.addWidget(self.save_btn)
        
        right_v_layout.addLayout(table_buttons_layout)


        # Status bar label (moved to main window's status bar)
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

        # Label to display final training metrics
        self.metrics_label = QLabel("Training Metrics: N/A")
        self.metrics_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.metrics_label.setWordWrap(True) # Allow text to wrap
        self.metrics_label.setFixedHeight(100) # Give it some fixed height to prevent layout shifts
        right_v_layout.addWidget(self.metrics_label)
        
        main_h_layout.addLayout(right_v_layout, 0) # Add right panel to main layout, no stretch for fixed width

        self.drawing_rect = False # Flag to indicate if a rectangle is currently being drawn
        self.start_point = QPointF() # Starting point of the rectangle
        self.current_rect_item = None # Reference to the QGraphicsRectItem being drawn

    def _on_thumbnail_clicked(self, item):
        """Slot to handle clicks on thumbnail items in the QListWidget."""
        # Get the custom widget associated with the QListWidgetItem
        thumbnail_widget = self.thumbnail_list_widget.itemWidget(item)
        if thumbnail_widget:
            # Ensure we switch back to real image view if DPT was active
            if self.is_dpt_view_active:
                self.toggle_dpt_view() # This will switch to real image and update button text
            self.load_image(thumbnail_widget.image_path)

    def create_thumbnail_pixmap(self, image_path, size=(90, 90)):
        """Creates a QPixmap thumbnail from an image file, cropping it first."""
        img_cv_full = cv2.imread(image_path)
        if img_cv_full is None:
            print(f"Warning: Could not create thumbnail for {image_path}")
            return QPixmap() # Return empty pixmap on error

        # CROP THE IMAGE to top 92.593%
        h, w, ch = img_cv_full.shape
        crop_h = int(h * 0.92593)
        img_cv = img_cv_full[0:crop_h, :]

        # Update height and width for QImage conversion
        h, w, ch = img_cv.shape
        bytes_per_line = ch * w
        q_image = QImage(img_cv.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        return pixmap.scaled(size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def update_thumbnail_status(self, image_path, is_annotated):
        """Updates the internal annotation status of a thumbnail, without visual border."""
        thumbnail_widget = self.image_item_map.get(image_path)
        if thumbnail_widget:
            thumbnail_widget.set_annotated_status(is_annotated)

    def load_directory(self):
        """Opens a dialog to select an image directory, asks for a camera name, and loads files."""
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if directory:
            # NEW: Prompt for Camera Name/ID
            camera_id, ok = QInputDialog.getText(self, "Enter Camera Name", "Please enter a name for this camera (e.g., 'Camera-01'):")
            if not ok or not camera_id.strip():
                self.status_label.setText("Directory loading cancelled. Camera name is required.")
                return # Abort if user cancels or enters empty name
            
            self.image_directory = directory
            self.camera_id = camera_id.strip()

            # Filter for common image file extensions
            self.image_files = [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
            ]
            self.image_files.sort() # Sort files alphabetically

            self.thumbnail_list_widget.clear() # Clear existing thumbnails
            self.image_item_map.clear() # Clear map
            self.all_annotations_data.clear() # Clear all annotations for the table

            if self.image_files:
                for img_path in self.image_files:
                    thumbnail_pixmap = self.create_thumbnail_pixmap(img_path)
                    
                    # Create a QListWidgetItem and set its custom widget
                    item = QListWidgetItem(self.thumbnail_list_widget)
                    thumbnail_widget = ThumbnailItemWidget(thumbnail_pixmap, img_path)
                    item.setSizeHint(thumbnail_widget.sizeHint()) # Set size hint for the list item
                    self.thumbnail_list_widget.setItemWidget(item, thumbnail_widget)
                    
                    self.image_item_map[img_path] = thumbnail_widget # Store reference to the custom widget

                    # Load annotations for this image and add to all_annotations_data
                    annotation_filename = img_path + ".json"
                    if os.path.exists(annotation_filename):
                        try:
                            with open(annotation_filename, 'r') as f:
                                existing_anns = json.load(f)
                                for ann in existing_anns:
                                    # Add image_path to each annotation for table display
                                    ann_with_path = ann.copy()
                                    ann_with_path['image_path'] = img_path
                                    self.all_annotations_data.append(ann_with_path)
                                if existing_anns:
                                    thumbnail_widget.set_annotated_status(True)
                                else:
                                    thumbnail_widget.set_annotated_status(False)
                        except json.JSONDecodeError:
                            thumbnail_widget.set_annotated_status(False) # If JSON is corrupt, treat as not annotated
                    else:
                        thumbnail_widget.set_annotated_status(False) # No JSON file means no annotations

                self.current_image_index = 0
                self.load_image(self.image_files[self.current_image_index]) # Load the first image
                self.status_label.setText(f"Loaded {len(self.image_files)} images for '{self.camera_id}' from {directory}")
                self.update_annotation_table() # Populate the table with all loaded annotations
            else:
                self.status_label.setText("No supported image files found in directory.")
        else:
            self.status_label.setText("No directory selected.")

    def load_image(self, image_path):
        """Loads, CROPS, and displays the specified image, along with its existing annotations."""
        # Deselect previously selected thumbnail
        if self.current_image_path and self.current_image_path in self.image_item_map:
            prev_thumbnail_widget = self.image_item_map[self.current_image_path]
            prev_thumbnail_widget.set_selected(False)

        # Save annotations of the current image before loading a new one
        if self.current_image_path:
            self.save_annotations_for_current_image() 

        self.current_image_path = image_path
        img_cv_full = cv2.imread(image_path) # Load full image using OpenCV
        if img_cv_full is None:
            self.status_label.setText(f"Error loading image: {image_path}")
            return
            
        # NEW: CROP THE IMAGE to the top 92.593%
        h, _, _ = img_cv_full.shape
        crop_h = int(h * 0.92593)
        self.current_image_cv = img_cv_full[0:crop_h, :] # Store the cropped image

        # Ensure we are in real image view when loading a new image
        self.is_dpt_view_active = False
        self.show_dpt_btn.setText("Show DPT Image")

        self.graphics_scene.clear() # Clear previous image and annotations from the scene
        h, w, ch = self.current_image_cv.shape
        bytes_per_line = ch * w
        # Convert OpenCV BGR image to QImage (RGB swapped for correct display)
        q_image = QImage(self.current_image_cv.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.graphics_scene.addPixmap(pixmap)
        # Fit the image into the view, maintaining aspect ratio
        self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        self.annotations = self.load_existing_annotations(image_path) # Load annotations for the new image
        self.draw_existing_annotations() # Draw them on the scene
        self.update_annotation_table() # Update the table with current image's annotations

        self.status_label.setText(f"Displaying: {os.path.basename(image_path)}")

        # Update current image index and select corresponding thumbnail
        if image_path in self.image_item_map:
            for i in range(self.thumbnail_list_widget.count()):
                item = self.thumbnail_list_widget.item(i)
                widget = self.thumbnail_list_widget.itemWidget(item)
                if widget and widget.image_path == image_path:
                    self.current_image_index = i
                    self.thumbnail_list_widget.setCurrentItem(item)
                    widget.set_selected(True) # Select the current thumbnail
                    break


    def load_previous_image(self):
        """Loads the previous image in the loaded directory."""
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.image_files[self.current_image_index])
        else:
            self.status_label.setText("No previous image.")

    def load_next_image(self):
        """Loads the next image in the loaded directory."""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image(self.image_files[self.current_image_index])
        else:
            self.status_label.setText("No next image.")

    def mouse_press_event(self, event):
        """Handles mouse press events for drawing bounding boxes."""
        if self.is_dpt_view_active: # Prevent annotation when DPT image is active
            self.status_label.setText("Cannot annotate on DPT depth map. Switch to real image view.")
            return

        if event.button() == Qt.LeftButton:
            # Map mouse position from view coordinates to scene coordinates
            scene_pos = self.graphics_view.mapToScene(event.pos())
            self.start_point = scene_pos
            self.drawing_rect = True
            # Add a temporary rectangle item to the scene for visual feedback during drawing
            self.current_rect_item = self.graphics_scene.addRect(QRectF(self.start_point, self.start_point), QPen(QColor(255, 0, 0), 2))

    def mouse_move_event(self, event):
        """Handles mouse move events for updating the drawing rectangle and displaying coordinates."""
        scene_pos = self.graphics_view.mapToScene(event.pos())
        self.status_label.setText(f"Mouse: ({int(scene_pos.x())}, {int(scene_pos.y())})")

        if self.drawing_rect and self.current_rect_item:
            # Update the rectangle's dimensions as the mouse moves
            rect = QRectF(self.start_point, scene_pos).normalized() # .normalized() ensures positive width/height
            self.current_rect_item.setRect(rect)

    def mouse_release_event(self, event):
        """Handles mouse release events, finalizing the bounding box and prompting for distance."""
        if self.is_dpt_view_active: # Prevent annotation when DPT image is active
            return # Already handled in mouse_press_event, but good for robustness

        if event.button() == Qt.LeftButton and self.drawing_rect:
            self.drawing_rect = False
            end_point = self.graphics_view.mapToScene(event.pos())

            if self.current_rect_item:
                self.graphics_scene.removeItem(self.current_rect_item) # Remove the temporary drawing rectangle
                self.current_rect_item = None

            # Calculate the final bounding box coordinates
            x_min = int(min(self.start_point.x(), end_point.x()))
            y_min = int(min(self.start_point.y(), end_point.y()))
            x_max = int(max(self.start_point.x(), end_point.x()))
            y_max = int(max(self.start_point.y(), end_point.y()))

            # Basic validation for very small bounding boxes
            if abs(x_max - x_min) < 5 or abs(y_max - y_min) < 5:
                self.status_label.setText("Bounding box too small. Please draw a larger box.")
                return

            # Prompt user for the distance
            distance, ok = QInputDialog.getDouble(self, "Enter Distance", "Distance (meters):", 0.0, 0.0, 1000.0, 2)
            if ok:
                annotation_data = {
                    "type": "bounding_box",
                    "coordinates": [x_min, y_min, x_max, y_max],
                    "distance_meters": distance
                }
                self.annotations.append(annotation_data) # Add to current image's annotations list
                
                # Also add to the all_annotations_data list for the table
                ann_with_path = annotation_data.copy()
                ann_with_path['image_path'] = self.current_image_path
                self.all_annotations_data.append(ann_with_path)

                self.draw_annotation(annotation_data) # Draw the finalized annotation on the scene
                self.update_annotation_table() # Update the table after adding new annotation
                self.status_label.setText(f"Annotated: {distance}m at [{x_min}, {y_min}, {x_max}, {y_max}]")
            else:
                self.status_label.setText("Annotation cancelled.")

    def draw_annotation(self, annotation_data):
        """Draws a single annotation (bounding box and text) on the graphics scene."""
        if annotation_data["type"] == "bounding_box":
            coords = annotation_data["coordinates"]
            
            # Robust error checking for coordinate format
            if not isinstance(coords, list):
                print(f"ERROR: Invalid coordinate format for drawing. 'coordinates' is not a list. Type: {type(coords)}, Value: {coords}")
                return
            if len(coords) != 4:
                print(f"ERROR: Invalid coordinate format for drawing. 'coordinates' list length is not 4. Length: {len(coords)}, Value: {coords}")
                return

            x_min, y_min, x_max, y_max = coords
            distance = annotation_data["distance_meters"]
            
            # Draw the bounding box with a RED pen (changed from green)
            self.graphics_scene.addRect(x_min, y_min, x_max - x_min, y_max - y_min, QPen(QColor(255, 0, 0), 2))
            # Add the distance text label with increased font size (changed from 12 to 30)
            text_item = self.graphics_scene.addText(f"{distance}m")
            text_item.setDefaultTextColor(QColor(255, 0, 0)) # Text color also red for consistency
            text_item.setFont(QFont("Arial", 30)) # Font size changed to 30
            text_item.setPos(x_min, y_min - 35) # Adjusted position to account for larger font size

    def draw_existing_annotations(self):
        """Draws all annotations currently loaded for the image."""
        for annotation in self.annotations:
            self.draw_annotation(annotation)

    def load_existing_annotations(self, image_path):
        """Loads annotations from a JSON file associated with the image."""
        annotation_filename = image_path + ".json"
        loaded_annotations = []
        if os.path.exists(annotation_filename):
            try:
                with open(annotation_filename, 'r') as f:
                    loaded_annotations = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading JSON from {annotation_filename}. Returning empty annotations.")
        
        # Update thumbnail status based on whether annotations were loaded
        self.update_thumbnail_status(image_path, bool(loaded_annotations)) 
        return loaded_annotations

    def save_annotations_for_current_image(self):
        """Saves the current image's annotations to a JSON file."""
        if self.current_image_path:
            annotation_filename = self.current_image_path + ".json"
            if self.annotations: # Only save if there are annotations
                with open(annotation_filename, 'w') as f:
                    json.dump(self.annotations, f, indent=4) # Save with pretty printing
                self.status_label.setText(f"Saved annotations for {os.path.basename(self.current_image_path)}")
                self.update_thumbnail_status(self.current_image_path, True)
            else: # If no annotations, remove the file if it exists
                if os.path.exists(annotation_filename):
                    os.remove(annotation_filename)
                    self.status_label.setText(f"Removed annotations for {os.path.basename(self.current_image_path)}")
                self.update_thumbnail_status(self.current_image_path, False)
        
        # Rebuild all_annotations_data and update table to reflect changes for the current image
        self.rebuild_all_annotations_data()
        self.update_annotation_table() 

    def rebuild_all_annotations_data(self):
        """Rebuilds the self.all_annotations_data list by re-reading all JSON files."""
        self.all_annotations_data.clear()
        for img_path in self.image_files:
            annotation_filename = img_path + ".json"
            if os.path.exists(annotation_filename):
                try:
                    with open(annotation_filename, 'r') as f:
                        existing_anns = json.load(f)
                        for ann in existing_anns:
                            ann_with_path = ann.copy()
                            ann_with_path['image_path'] = img_path
                            self.all_annotations_data.append(ann_with_path)
                except json.JSONDecodeError:
                    print(f"Warning: Could not read JSON from {annotation_filename} during rebuild.")


    def save_annotations(self):
        """Saves annotations for the currently displayed image."""
        self.save_annotations_for_current_image()
        self.status_label.setText("All annotations saved.")

    def delete_selected_annotation(self):
        """Deletes the selected annotation from the table and internal data."""
        if self.is_dpt_view_active:
            self.status_label.setText("Cannot delete annotations while DPT depth map is active. Switch to real image view.")
            return

        selected_rows = self.annotation_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select a row in the table to delete an annotation.")
            return

        # Get the row index of the first selected row (assuming single selection for simplicity)
        row_to_delete = selected_rows[0].row()

        # Get the image path and coordinates from the selected row to uniquely identify the annotation
        image_name_item = self.annotation_table.item(row_to_delete, 0)
        coords_str_item = self.annotation_table.item(row_to_delete, 1)

        if not image_name_item or not coords_str_item:
            self.status_label.setText("Error: Could not retrieve annotation data from selected row.")
            return

        image_name = image_name_item.text()
        coords_str = coords_str_item.text()

        # Convert coordinates string back to list of integers
        try:
            coords_list = json.loads(coords_str) # Safely parse the string representation of list
        except json.JSONDecodeError:
            self.status_label.setText("Error: Could not parse coordinates from table.")
            return

        # Find and remove the annotation from self.all_annotations_data
        original_len_all_data = len(self.all_annotations_data)
        self.all_annotations_data = [
            ann for ann in self.all_annotations_data
            if not (os.path.basename(ann['image_path']) == image_name and ann['coordinates'] == coords_list)
        ]
        if len(self.all_annotations_data) == original_len_all_data:
            self.status_label.setText("Warning: Selected annotation not found in all_annotations_data.")
        else:
            self.status_label.setText(f"Deleted annotation for {image_name}.")

        # If the deleted annotation was for the currently viewed image, remove it from self.annotations too
        if os.path.basename(self.current_image_path) == image_name:
            original_len_current_anns = len(self.annotations)
            self.annotations = [
                ann for ann in self.annotations
                if not (ann['coordinates'] == coords_list)
            ]
            if len(self.annotations) != original_len_current_anns:
                # Refresh the current image display to remove the deleted bounding box
                self.graphics_scene.clear()
                h, w, ch = self.current_image_cv.shape
                bytes_per_line = ch * w
                q_image = QImage(self.current_image_cv.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_image)
                self.graphics_scene.addPixmap(pixmap)
                self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
                self.draw_existing_annotations()

        self.update_annotation_table() # Refresh the table display


    def update_annotation_table(self):
        """Populates the QTableWidget with all annotations from the directory."""
        self.annotation_table.setRowCount(0) # Clear existing rows
        if not self.all_annotations_data:
            return

        for row_idx, annotation in enumerate(self.all_annotations_data):
            self.annotation_table.insertRow(row_idx)
            
            # Image Name (use basename for brevity)
            image_name_item = QTableWidgetItem(os.path.basename(annotation['image_path']))
            image_name_item.setFlags(image_name_item.flags() & ~Qt.ItemIsEditable) # Make non-editable
            self.annotation_table.setItem(row_idx, 0, image_name_item)

            # Coordinates
            coords_str = str(annotation['coordinates']) # Simpler string conversion
            coords_item = QTableWidgetItem(coords_str)
            coords_item.setFlags(coords_item.flags() & ~Qt.ItemIsEditable) # Make non-editable
            self.annotation_table.setItem(row_idx, 1, coords_item)

            # Distance
            distance_item = QTableWidgetItem(f"{annotation['distance_meters']:.2f}m")
            distance_item.setFlags(distance_item.flags() & ~Qt.ItemIsEditable) # Make non-editable
            self.annotation_table.setItem(row_idx, 2, distance_item)

            # Highlight the row if it corresponds to the currently displayed image
            if annotation['image_path'] == self.current_image_path:
                for col_idx in range(self.annotation_table.columnCount()):
                    item = self.annotation_table.item(row_idx, col_idx)
                    if item:
                        item.setBackground(QColor(Qt.lightGray))
            else:
                # Set transparent to inherit parent background (which is the app's default)
                for col_idx in range(self.annotation_table.columnCount()):
                    item = self.annotation_table.item(row_idx, col_idx)
                    if item:
                        item.setBackground(QColor(Qt.transparent))


    def start_training(self):
        """Initiates the training process in a separate thread."""
        print("DEBUG: start_training method called.") # Diagnostic print
        if not self.image_files or not self.camera_id or not self.image_directory:
            QMessageBox.warning(self, "Not Ready for Training", "Please load an image directory and provide a camera name first.")
            return
        if self.dpt_model is None:
            QMessageBox.warning(self, "DPT Model Not Loaded", "DPT model could not be loaded. Please check your internet connection and dependencies.")
            return

        self.save_annotations_for_current_image() # Ensure current image's annotations are saved

        self.set_buttons_enabled(False) # Disable buttons during training
        self.status_label.setText("Initiating training process...")
        self.metrics_label.setText("Training Metrics: Calculating...") # Clear previous metrics display

        # Create and start the training thread, passing the loaded DPT model components and new info
        self.training_thread = TrainingThread(self.image_files, self.dpt_processor, self.dpt_model, self.device, self.image_directory, self.camera_id)
        self.training_thread.update_status.connect(self.status_label.setText) # Connect thread's status signal to UI label
        self.training_thread.training_finished.connect(self.on_training_finished) # Connect completion signal
        self.training_thread.final_metrics_report.connect(self.update_metrics_display) # Connect metrics signal
        self.training_thread.plot_generated.connect(self.show_plot_window) # Connect plot signal to new window handler
        self.training_thread.start()

    def on_training_finished(self):
        """Slot called when the training thread finishes."""
        self.set_buttons_enabled(True) # Re-enable buttons
        self.status_label.setText("Training process completed. Ready for next steps.")
        QMessageBox.information(self, "Training Complete", "The training process has finished and the model is saved.")

    def update_metrics_display(self, report_message):
        """Slot to receive and display the final training metrics."""
        self.metrics_label.setText(report_message)

    def show_plot_window(self, plot_path):
        """Slot to create and show the plot in a new window."""
        print(f"DEBUG: show_plot_window received path: {plot_path}")
        if plot_path and os.path.exists(plot_path):
            # Check if a plot window already exists and is visible
            if self.plot_window is None or not self.plot_window.isVisible():
                self.plot_window = PlotWindow(plot_path, self) # Pass self as parent to ensure proper garbage collection
                self.plot_window.show()
                print("DEBUG: New PlotWindow created and shown.")
            else:
                # If window exists, update its content and bring to front
                self.plot_window.load_plot(plot_path)
                self.plot_window.activateWindow() # Bring to front
                self.plot_window.raise_() # Raise to top of stack
                print("DEBUG: Existing PlotWindow updated.")
        else:
            QMessageBox.warning(self, "Plot Error", "Could not display plot. Image file not found or invalid.")
            print(f"ERROR: Plot image path is invalid or file does not exist for new window: {plot_path}")

    def toggle_dpt_view(self):
        """Toggles between displaying the real image and its DPT depth map."""
        if self.current_image_cv is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first to toggle DPT depth map.")
            return
        if self.dpt_model is None or self.dpt_processor is None:
            QMessageBox.warning(self, "DPT Model Not Loaded", "DPT model is not loaded. Please check your internet connection and dependencies.")
            return

        self.graphics_scene.clear() # Clear current display

        if not self.is_dpt_view_active: # Currently showing real image, switch to DPT
            self.status_label.setText("Generating DPT depth map...")
            try:
                # Convert BGR (OpenCV default) to RGB (DPT model expects RGB)
                img_rgb = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2RGB)

                # Process image with DPT processor and move to the appropriate device (CPU/GPU)
                inputs = self.dpt_processor(images=img_rgb, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad(): # Disable gradient calculation for inference
                    outputs = self.dpt_model(**inputs)
                    predicted_depth_unscaled = outputs.predicted_depth.squeeze().cpu().numpy()

                    # Resize the depth map to the original cropped image dimensions for accurate visualization
                    depth_map = cv2.resize(predicted_depth_unscaled, 
                                           (self.current_image_cv.shape[1], self.current_image_cv.shape[0]), 
                                           interpolation=cv2.INTER_AREA)

                # Normalize the depth map to 0-255 for visualization as a grayscale image
                normalized_depth = depth_map.astype(np.float32)
                if normalized_depth.max() - normalized_depth.min() > 0:
                    normalized_depth = (normalized_depth - normalized_depth.min()) / \
                                       (normalized_depth.max() - normalized_depth.min()) * 255
                else:
                    normalized_depth = np.zeros_like(normalized_depth) * 255

                normalized_depth = normalized_depth.astype(np.uint8)

                h, w = normalized_depth.shape
                q_image = QImage(normalized_depth.data, w, h, w, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(q_image)
                self.graphics_scene.addPixmap(pixmap)
                self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

                self.is_dpt_view_active = True
                self.show_dpt_btn.setText("Show Real Image")
                self.status_label.setText("DPT depth map displayed.")

            except Exception as e:
                self.status_label.setText(f"Error generating DPT depth map: {str(e)}")
                print(f"DPT Depth Map Generation Error: {e}")
        else: # Currently showing DPT, switch to real image
            h, w, ch = self.current_image_cv.shape
            bytes_per_line = ch * w
            q_image = QImage(self.current_image_cv.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            self.graphics_scene.addPixmap(pixmap)
            self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self.draw_existing_annotations() # Redraw annotations on real image

            self.is_dpt_view_active = False
            self.show_dpt_btn.setText("Show DPT Image")
            self.status_label.setText("Real image displayed.")


    def set_buttons_enabled(self, enabled):
        """Enables or disables control buttons."""
        self.load_dir_btn.setEnabled(enabled)
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.train_btn.setEnabled(enabled)
        self.show_dpt_btn.setEnabled(enabled) # Enable/disable the new button
        self.delete_annotation_btn.setEnabled(enabled) # Enable/disable the new delete button

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnnotationTool()
    window.show()
    sys.exit(app.exec_())
