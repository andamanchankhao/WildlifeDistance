#################################################################
### Dev: Andaman Chankhao andmanchankhao@gmail.com            ####
### Rev: Gemini (Code Refinement and Feature Enhancement)     ####
#################################################################

import sys
import os
import json
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGraphicsView, QGraphicsScene, QInputDialog,
    QFileDialog, QMessageBox, QDialog, QSizePolicy, QListWidget,
    QListWidgetItem, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt, QPointF, QRectF, QThread, pyqtSignal, QTemporaryFile, QSize

# --- Dependency Imports with Graceful Fallbacks ---

# TensorFlow and Keras for model training
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model # type: ignore
    from tensorflow.keras.layers import Input, Dense, Dropout # type: ignore
    from tensorflow.keras.callbacks import Callback # type: ignore
    from tensorflow.keras.metrics import MeanAbsoluteError # type: ignore
except ImportError:
    tf = None
    print("Warning: TensorFlow not installed. Training functionality will be disabled. Install with 'pip install tensorflow'.")

# Scikit-learn for metrics, data splitting, and scaling
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    mean_squared_error = r2_score = train_test_split = StandardScaler = None
    print("Warning: Scikit-learn not installed. Advanced metrics, data splitting, and scaling will be unavailable. Install with 'pip install scikit-learn'.")

# Joblib for saving/loading the scaler
try:
    import joblib
except ImportError:
    joblib = None
    print("Warning: Joblib not installed. The feature scaler will not be saved. Install with 'pip install joblib'.")

# Matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')  # Use non-interactive backend for PyQt
except ImportError:
    plt = None
    print("Warning: Matplotlib not installed. Plot generation will be disabled. Install with 'pip install matplotlib'.")

# Hugging Face Transformers and PyTorch for DPT model
try:
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    import torch
except ImportError:
    DPTForDepthEstimation = DPTImageProcessor = torch = None
    print("Warning: Transformers or PyTorch not installed. DPT functionality is disabled. Install with 'pip install transformers torch'.")

# --- Constants ---
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp')
IMAGE_CROP_FACTOR = 0.92593  # Crops the bottom part of the image to remove potential timestamps.
DPT_MODEL_NAME = "Intel/dpt-hybrid-midas"
MODEL_FILENAME_TEMPLATE = "{camera_id}_distance_model.keras"
SCALER_FILENAME_TEMPLATE = "{camera_id}_scaler.joblib"
THUMBNAIL_SIZE = QSize(90, 90)
ANNOTATION_PEN = QPen(QColor("red"), 2)
ANNOTATION_FONT = QFont("Arial", 16, QFont.Bold)
TEMP_RECT_PEN = QPen(QColor("red"), 2, Qt.DashLine)

# --- Helper Functions ---
def convert_cv_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    """Converts an OpenCV image (BGR) to a QPixmap."""
    if cv_img is None:
        return QPixmap()
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(q_img)

def generate_dpt_depth_map(cv_img: np.ndarray, processor, model, device: str) -> Optional[np.ndarray]:
    """Generates a depth map from an OpenCV image using a DPT model."""
    if any(lib is None for lib in [cv_img, processor, model, torch]):
        return None
    try:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        inputs = processor(images=img_rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()

        # Resize depth map to match original image for accurate coordinate mapping
        depth_map_resized = cv2.resize(
            predicted_depth,
            (cv_img.shape[1], cv_img.shape[0]),
            interpolation=cv2.INTER_AREA
        )
        return depth_map_resized
    except Exception as e:
        print(f"Error during DPT depth map generation: {e}")
        return None

# --- PyQt GUI Classes ---

class PyQtCallback(Callback):
    """Custom Keras Callback to update the PyQt GUI during training."""
    def __init__(self, update_status_signal: pyqtSignal):
        super().__init__()
        self.update_status_signal = update_status_signal

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        status_msg = (
            f"Epoch {epoch + 1}: "
            f"Loss: {logs.get('loss', 0):.4f}, "
            f"MAE: {logs.get('mean_absolute_error', 0):.4f}, "
            f"Val Loss: {logs.get('val_loss', 0):.4f}, "
            f"Val MAE: {logs.get('val_mean_absolute_error', 0):.4f}"
        )
        self.update_status_signal.emit(status_msg)

class PlotWindow(QDialog):
    """A dialog window to display the matplotlib plot."""
    def __init__(self, plot_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Actual vs. Predicted Distance Plot")
        self.setGeometry(200, 200, 800, 600)
        self.plot_path = plot_path

        layout = QVBoxLayout(self)
        self.plot_label = QLabel("Loading Plot...")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        layout.addWidget(self.plot_label)
        self.original_pixmap = QPixmap(self.plot_path)

    def showEvent(self, event):
        """Load plot when the widget is shown to ensure correct initial size."""
        super().showEvent(event)
        self.update_plot_display()

    def resizeEvent(self, event):
        """Re-scale the pixmap when the window is resized."""
        super().resizeEvent(event)
        self.update_plot_display()

    def update_plot_display(self):
        """Loads and scales the plot image to fit the label."""
        if not self.original_pixmap.isNull():
            scaled_pixmap = self.original_pixmap.scaled(
                self.plot_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.plot_label.setPixmap(scaled_pixmap)
        else:
            self.plot_label.setText(f"Error: Could not load plot from {self.plot_path}")

class ThumbnailItemWidget(QWidget):
    """Custom widget for displaying an image thumbnail in the QListWidget."""
    def __init__(self, pixmap: QPixmap, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.is_annotated = False
        self.setFixedSize(QSize(100, 100))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        thumbnail_label = QLabel()
        thumbnail_label.setAlignment(Qt.AlignCenter)
        thumbnail_label.setPixmap(pixmap.scaled(
            THUMBNAIL_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        layout.addWidget(thumbnail_label)
        self.setAutoFillBackground(True)
        self.set_selected(False) # Set initial background

    def set_annotated_status(self, annotated: bool):
        self.is_annotated = annotated

    def set_selected(self, selected: bool):
        """Updates the selection status and background color."""
        palette = self.palette()
        color = QColor("lightblue") if selected else QColor("transparent")
        palette.setColor(self.backgroundRole(), color)
        self.setPalette(palette)


class TrainingThread(QThread):
    """Worker thread for training the model to prevent UI freezing."""
    update_status = pyqtSignal(str)
    training_finished = pyqtSignal()
    final_metrics_report = pyqtSignal(str)
    plot_generated = pyqtSignal(str)

    def __init__(self, image_files: List[str], annotations: Dict[str, List], dpt_processor, dpt_model, device: str, save_dir: str, camera_id: str):
        super().__init__()
        self.image_files = image_files
        self.annotations = annotations
        self.processor = dpt_processor
        self.dpt_model = dpt_model
        self.device = device
        self.save_dir = save_dir
        self.camera_id = camera_id

    def get_depth_for_bbox(self, depth_map: np.ndarray, bbox_coords: List[int]) -> Optional[float]:
        """Extracts a representative depth value (median) from a bounding box."""
        x_min, y_min, x_max, y_max = bbox_coords
        h, w = depth_map.shape
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        if x_max <= x_min or y_max <= y_min:
            return None

        cropped_depth = depth_map[y_min:y_max, x_min:x_max]
        valid_depths = cropped_depth[np.isfinite(cropped_depth)]

        return np.median(valid_depths) if valid_depths.size > 0 else None

    def run(self):
        """The main training logic executed in the worker thread."""
        if tf is None:
            self.update_status.emit("Error: TensorFlow is not installed. Training aborted.")
            self.training_finished.emit()
            return

        try:
            self.update_status.emit("Preparing data for training...")
            X_features, y_labels = [], []
            depth_map_cache = {}

            all_annotations = [ann for anns in self.annotations.values() for ann in anns]
            if not all_annotations:
                self.update_status.emit("No annotations found. Please annotate images first.")
                self.training_finished.emit()
                return

            for ann in all_annotations:
                img_path = ann["image_path"]
                if img_path in depth_map_cache:
                    depth_map = depth_map_cache[img_path]
                else:
                    img_full = cv2.imread(img_path)
                    if img_full is None: continue
                    h, _, _ = img_full.shape
                    img_cropped = img_full[0:int(h * IMAGE_CROP_FACTOR), :]
                    depth_map = generate_dpt_depth_map(img_cropped, self.processor, self.dpt_model, self.device)
                    depth_map_cache[img_path] = depth_map

                if depth_map is None: continue

                bbox_depth = self.get_depth_for_bbox(depth_map, ann["coordinates"])
                if bbox_depth is not None:
                    X_features.append([bbox_depth])
                    y_labels.append(ann["distance_meters"])

            if not X_features:
                self.update_status.emit("No valid features could be extracted. Training aborted.")
                self.training_finished.emit()
                return

            X, y = np.array(X_features, dtype=np.float32), np.array(y_labels, dtype=np.float32)
            self.update_status.emit(f"Extracted {len(X)} features. Preparing model...")

            # Data Splitting
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Feature Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Save Scaler
            scaler_path = os.path.join(self.save_dir, SCALER_FILENAME_TEMPLATE.format(camera_id=self.camera_id))
            joblib.dump(scaler, scaler_path)
            self.update_status.emit(f"Scaler saved to {scaler_path}")
            
            # Build and Compile Model
            inputs = Input(shape=(1,))
            x = Dense(64, activation='relu')(inputs)
            x = Dropout(0.2)(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.2)(x)
            outputs = Dense(1)(x)
            model = Model(inputs, outputs)
            model.compile(optimizer='adam', loss='mse', metrics=[MeanAbsoluteError()])
            
            # Train Model
            self.update_status.emit("Starting model training...")
            model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=50, batch_size=8,
                callbacks=[PyQtCallback(self.update_status)],
                verbose=0
            )

            self.update_status.emit("Training complete. Evaluating...")

            # Evaluation and Reporting
            X_all_scaled = scaler.transform(X)
            y_pred = model.predict(X_all_scaled).flatten()
            
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            report = (
                f"--- Training Metrics Report ---\n"
                f"Camera ID: {self.camera_id}\n"
                f"Mean Absolute Error (MAE): {mae:.4f} meters\n"
                f"Root Mean Squared Error (RMSE): {rmse:.4f} meters\n"
                f"R-squared (R²): {r2:.4f}"
            )
            self.final_metrics_report.emit(report)

            # Generate Plot
            if plt:
                with QTemporaryFile(suffix=".png", delete=False) as temp_file:
                    plot_path = temp_file.name
                
                plt.figure(figsize=(8, 6))
                plt.scatter(y, y_pred, alpha=0.7, label="Predictions")
                min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                plt.title(f"Actual vs. Predicted Distance (Camera: {self.camera_id})")
                plt.xlabel("Actual Distance (meters)")
                plt.ylabel("Predicted Distance (meters)")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                self.plot_generated.emit(plot_path)

            # Save Model
            model_path = os.path.join(self.save_dir, MODEL_FILENAME_TEMPLATE.format(camera_id=self.camera_id))
            model.save(model_path)
            self.update_status.emit(f"Model saved to {model_path}")

        except Exception as e:
            self.update_status.emit(f"An error occurred during training: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.training_finished.emit()


class CustomGraphicsView(QGraphicsView):
    """Subclassed QGraphicsView to emit signals for mouse events."""
    mouse_pressed = pyqtSignal(object)
    mouse_moved = pyqtSignal(object)
    mouse_released = pyqtSignal(object)

    def mousePressEvent(self, event):
        self.mouse_pressed.emit(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.mouse_moved.emit(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.mouse_released.emit(event)
        super().mouseReleaseEvent(event)

class AnnotationTool(QMainWindow):
    """Main application window for the annotation tool."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wildlife Distance Annotation Tool")
        self.setGeometry(100, 100, 1400, 900)

        # State variables
        self.image_directory: Optional[str] = None
        self.camera_id: Optional[str] = None
        self.image_files: List[str] = []
        self.current_image_path: Optional[str] = None
        self.current_image_cv: Optional[np.ndarray] = None
        self.annotations_by_image: Dict[str, List] = {}
        self.is_dpt_view_active = False
        
        # Drawing state
        self.drawing_rect = False
        self.start_point = QPointF()
        self.current_rect_item = None
        
        # DPT Model components
        self.dpt_processor = self.dpt_model = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"

        # Child windows and threads
        self.training_thread: Optional[TrainingThread] = None
        self.plot_window: Optional[PlotWindow] = None
        self.image_item_map: Dict[str, ThumbnailItemWidget] = {}

        self.init_ui()
        self.load_dpt_inference_model()
    
    def load_dpt_inference_model(self):
        """Loads the pre-trained DPT model from Hugging Face."""
        if DPTForDepthEstimation is None:
            self.status_label.setText("DPT dependencies not installed. Inference and training are disabled.")
            return
        try:
            self.status_label.setText(f"Loading DPT model '{DPT_MODEL_NAME}'...")
            self.dpt_processor = DPTImageProcessor.from_pretrained(DPT_MODEL_NAME)
            self.dpt_model = DPTForDepthEstimation.from_pretrained(DPT_MODEL_NAME).to(self.device)
            self.dpt_model.eval()
            self.status_label.setText(f"DPT model loaded successfully on {self.device}.")
        except Exception as e:
            self.status_label.setText(f"Failed to load DPT model: {e}")
            print(f"DPT Model Load Error: {e}")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel (Directory and Thumbnails) ---
        left_panel = QVBoxLayout()
        self.load_dir_btn = QPushButton("Load Directory")
        self.load_dir_btn.clicked.connect(self.load_directory)
        left_panel.addWidget(self.load_dir_btn)

        self.thumbnail_list_widget = QListWidget()
        self.thumbnail_list_widget.setFixedWidth(120)
        self.thumbnail_list_widget.setViewMode(QListWidget.IconMode)
        self.thumbnail_list_widget.setIconSize(THUMBNAIL_SIZE)
        self.thumbnail_list_widget.setSpacing(5)
        self.thumbnail_list_widget.itemClicked.connect(self._on_thumbnail_clicked)
        left_panel.addWidget(self.thumbnail_list_widget)
        main_layout.addLayout(left_panel)

        # --- Center Panel (Image Viewer) ---
        center_panel = QVBoxLayout()
        image_nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("<")
        self.prev_btn.setFixedSize(30, 30)
        self.prev_btn.clicked.connect(self.load_previous_image)
        image_nav_layout.addWidget(self.prev_btn)
        
        self.graphics_view = CustomGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.mouse_pressed.connect(self.mouse_press_event)
        self.graphics_view.mouse_moved.connect(self.mouse_move_event)
        self.graphics_view.mouse_released.connect(self.mouse_release_event)
        image_nav_layout.addWidget(self.graphics_view)

        self.next_btn = QPushButton(">")
        self.next_btn.setFixedSize(30, 30)
        self.next_btn.clicked.connect(self.load_next_image)
        image_nav_layout.addWidget(self.next_btn)
        center_panel.addLayout(image_nav_layout)

        controls_layout = QHBoxLayout()
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.start_training)
        controls_layout.addWidget(self.train_btn)

        self.show_dpt_btn = QPushButton("Show DPT Depth")
        self.show_dpt_btn.setCheckable(True)
        self.show_dpt_btn.toggled.connect(self.toggle_dpt_view)
        controls_layout.addWidget(self.show_dpt_btn)
        center_panel.addLayout(controls_layout)
        main_layout.addLayout(center_panel, 1)

        # --- Right Panel (Annotations Table and Metrics) ---
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(5, 0, 5, 0)
        self.annotation_table = QTableWidget()
        self.annotation_table.setColumnCount(3)
        self.annotation_table.setHorizontalHeaderLabels(["Image", "Coordinates", "Distance (m)"])
        self.annotation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.annotation_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.annotation_table.setSelectionBehavior(QTableWidget.SelectRows)
        right_panel.addWidget(self.annotation_table, 1)

        table_buttons_layout = QHBoxLayout()
        self.delete_annotation_btn = QPushButton("Delete Selected Annotation")
        self.delete_annotation_btn.clicked.connect(self.delete_selected_annotation)
        table_buttons_layout.addWidget(self.delete_annotation_btn)
        self.save_btn = QPushButton("Save Current Annotations")
        self.save_btn.clicked.connect(self.save_annotations_for_current_image)
        table_buttons_layout.addWidget(self.save_btn)
        right_panel.addLayout(table_buttons_layout)

        self.metrics_label = QLabel("Training Metrics: N/A")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setMinimumHeight(100)
        right_panel.addWidget(self.metrics_label, 0, Qt.AlignTop)
        main_layout.addLayout(right_panel)
        
        # --- Status Bar ---
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

    def _on_thumbnail_clicked(self, item: QListWidgetItem):
        """Slot to handle clicks on thumbnail items."""
        thumbnail_widget = self.thumbnail_list_widget.itemWidget(item)
        if thumbnail_widget and thumbnail_widget.image_path != self.current_image_path:
            if self.show_dpt_btn.isChecked():
                self.show_dpt_btn.setChecked(False) # This will trigger toggle_dpt_view to switch back
            self.load_image(thumbnail_widget.image_path)

    def load_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if not directory:
            return

        camera_id, ok = QInputDialog.getText(self, "Enter Camera Name", "Please provide a unique name for this camera:")
        if not ok or not camera_id.strip():
            QMessageBox.warning(self, "Input Required", "A camera name is required to proceed.")
            return
        
        self.save_annotations_for_current_image() # Save work before switching directory
        self.image_directory = directory
        self.camera_id = camera_id.strip()

        self.image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)])

        # Reset UI and state
        self.thumbnail_list_widget.clear()
        self.image_item_map.clear()
        self.annotations_by_image.clear()
        self.annotation_table.setRowCount(0)
        self.graphics_scene.clear()

        if not self.image_files:
            self.status_label.setText("No supported image files found in the selected directory.")
            return

        for img_path in self.image_files:
            # Load annotations
            self.annotations_by_image[img_path] = self.load_annotations_from_file(img_path)

            # Create and add thumbnail
            pixmap = QPixmap(img_path)
            item = QListWidgetItem()
            widget = ThumbnailItemWidget(pixmap, img_path)
            widget.set_annotated_status(bool(self.annotations_by_image[img_path]))
            item.setSizeHint(widget.sizeHint())
            self.thumbnail_list_widget.addItem(item)
            self.thumbnail_list_widget.setItemWidget(item, widget)
            self.image_item_map[img_path] = widget
            
        self.status_label.setText(f"Loaded {len(self.image_files)} images for camera '{self.camera_id}'.")
        self.load_image(self.image_files[0])
        self.update_annotation_table()

    def load_image(self, image_path: str):
        """Loads and displays an image and its annotations."""
        if self.current_image_path:
            self.save_annotations_for_current_image()
            if self.current_image_path in self.image_item_map:
                self.image_item_map[self.current_image_path].set_selected(False)
        
        self.current_image_path = image_path
        img_full = cv2.imread(image_path)
        if img_full is None:
            self.status_label.setText(f"Error: Could not load image {os.path.basename(image_path)}")
            return
        
        h, _, _ = img_full.shape
        self.current_image_cv = img_full[0:int(h * IMAGE_CROP_FACTOR), :]

        self.display_image(self.current_image_cv)
        self.draw_existing_annotations()
        self.update_annotation_table()
        
        # Update UI selection
        if image_path in self.image_item_map:
            self.image_item_map[image_path].set_selected(True)
        self.status_label.setText(f"Displaying: {os.path.basename(image_path)}")

    def display_image(self, cv_img: np.ndarray):
        """Clears the scene and displays the given OpenCV image."""
        self.graphics_scene.clear()
        pixmap = convert_cv_to_qpixmap(cv_img)
        self.graphics_scene.addPixmap(pixmap)
        self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def navigate_image(self, direction: int):
        """Navigate to the next or previous image."""
        if not self.image_files or self.current_image_path is None: return
        try:
            current_index = self.image_files.index(self.current_image_path)
            new_index = current_index + direction
            if 0 <= new_index < len(self.image_files):
                self.load_image(self.image_files[new_index])
        except ValueError:
            pass # Current image not in list

    def load_previous_image(self): self.navigate_image(-1)
    def load_next_image(self): self.navigate_image(1)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for navigation."""
        if event.key() == Qt.Key_Left:
            self.load_previous_image()
        elif event.key() == Qt.Key_Right:
            self.load_next_image()
        else:
            super().keyPressEvent(event)
            
    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton and not self.is_dpt_view_active:
            self.start_point = self.graphics_view.mapToScene(event.pos())
            self.drawing_rect = True
            self.current_rect_item = self.graphics_scene.addRect(
                QRectF(self.start_point, self.start_point), TEMP_RECT_PEN
            )

    def mouse_move_event(self, event):
        scene_pos = self.graphics_view.mapToScene(event.pos())
        self.statusBar().showMessage(f"Coordinates: ({int(scene_pos.x())}, {int(scene_pos.y())})")
        if self.drawing_rect and self.current_rect_item:
            rect = QRectF(self.start_point, scene_pos).normalized()
            self.current_rect_item.setRect(rect)

    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton and self.drawing_rect:
            self.drawing_rect = False
            if self.current_rect_item:
                rect = self.current_rect_item.rect()
                self.graphics_scene.removeItem(self.current_rect_item)
                self.current_rect_item = None
                
                if rect.width() < 5 or rect.height() < 5:
                    self.status_label.setText("Annotation cancelled: Bounding box too small.")
                    return

                distance, ok = QInputDialog.getDouble(self, "Enter Distance", "Distance (meters):", 0.0, 0.0, 1000.0, 2)
                if ok:
                    coords = [int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())]
                    annotation = {
                        "type": "bounding_box",
                        "coordinates": coords,
                        "distance_meters": distance,
                        "image_path": self.current_image_path # Store path for easy reference
                    }
                    self.annotations_by_image[self.current_image_path].append(annotation)
                    self.draw_annotation(annotation)
                    self.update_annotation_table()
                    self.image_item_map[self.current_image_path].set_annotated_status(True)

    def draw_annotation(self, annotation: Dict):
        """Draws a single annotation on the scene and stores its Qt items."""
        coords = annotation["coordinates"]
        rect = QRectF(coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1])
        rect_item = self.graphics_scene.addRect(rect, ANNOTATION_PEN)

        text_item = self.graphics_scene.addText(f"{annotation['distance_meters']:.2f}m", ANNOTATION_FONT)
        text_item.setDefaultTextColor(ANNOTATION_PEN.color())
        text_item.setPos(rect.topLeft() - QPointF(0, ANNOTATION_FONT.pointSize() * 1.2))
        
        # Store Qt graphics items with the annotation for easy removal
        annotation['qt_items'] = [rect_item, text_item]

    def draw_existing_annotations(self):
        if self.current_image_path:
            for ann in self.annotations_by_image.get(self.current_image_path, []):
                self.draw_annotation(ann)

    def update_annotation_table(self):
        """Populates the table with all annotations from all loaded images."""
        self.annotation_table.setRowCount(0)
        current_anns = []
        other_anns = []

        # Separate current image's annotations to display them first
        for img_path, annotations in self.annotations_by_image.items():
            for ann in annotations:
                if img_path == self.current_image_path:
                    current_anns.append(ann)
                else:
                    other_anns.append(ann)
        
        sorted_annotations = current_anns + sorted(other_anns, key=lambda x: os.path.basename(x['image_path']))

        for row_idx, ann in enumerate(sorted_annotations):
            self.annotation_table.insertRow(row_idx)
            items = [
                QTableWidgetItem(os.path.basename(ann['image_path'])),
                QTableWidgetItem(str(ann['coordinates'])),
                QTableWidgetItem(f"{ann['distance_meters']:.2f}")
            ]
            for col_idx, item in enumerate(items):
                # Store a reference to the annotation object in the first column's item
                if col_idx == 0:
                    item.setData(Qt.UserRole, ann)
                
                if ann['image_path'] == self.current_image_path:
                    item.setBackground(QColor("lightyellow"))

                self.annotation_table.setItem(row_idx, col_idx, item)

    def delete_selected_annotation(self):
        selected_items = self.annotation_table.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Please select a row to delete.")
            return

        # Use UserRole data to get the exact annotation object
        ann_to_delete = selected_items[0].data(Qt.UserRole)
        if not ann_to_delete: return
        
        img_path = ann_to_delete['image_path']
        
        # Remove from data structure
        self.annotations_by_image[img_path].remove(ann_to_delete)

        # Remove from scene if it's on the current image
        if 'qt_items' in ann_to_delete:
            for item in ann_to_delete['qt_items']:
                self.graphics_scene.removeItem(item)
        
        # Update UI
        if not self.annotations_by_image[img_path] and img_path in self.image_item_map:
            self.image_item_map[img_path].set_annotated_status(False)
            
        self.update_annotation_table()
        self.status_label.setText(f"Annotation deleted for {os.path.basename(img_path)}.")
        self.save_annotations_for_image(img_path) # Persist the deletion

    def load_annotations_from_file(self, image_path: str) -> List[Dict]:
        """Loads annotations from a JSON file for a specific image."""
        ann_path = image_path + ".json"
        if os.path.exists(ann_path):
            try:
                with open(ann_path, 'r') as f:
                    annotations = json.load(f)
                    # Add image_path to each loaded annotation for consistency
                    for ann in annotations:
                        ann['image_path'] = image_path
                    return annotations
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Corrupt annotation file skipped: {ann_path}")
        return []

    def save_annotations_for_image(self, image_path: str):
        """Saves annotations for a specific image to its JSON file."""
        if image_path is None: return
        
        ann_path = image_path + ".json"
        annotations = self.annotations_by_image.get(image_path, [])
        
        # Create a clean list for saving (without 'qt_items' or 'image_path')
        clean_annotations = [
            {k: v for k, v in ann.items() if k not in ['qt_items', 'image_path']}
            for ann in annotations
        ]

        if clean_annotations:
            with open(ann_path, 'w') as f:
                json.dump(clean_annotations, f, indent=4)
        elif os.path.exists(ann_path):
            os.remove(ann_path) # Clean up empty annotation files

    def save_annotations_for_current_image(self):
        """Convenience method to save annotations for the currently viewed image."""
        if self.current_image_path:
            self.save_annotations_for_image(self.current_image_path)
            self.status_label.setText(f"Annotations saved for {os.path.basename(self.current_image_path)}")

    def toggle_dpt_view(self, checked: bool):
        """Toggles between the real image and its DPT depth map."""
        self.is_dpt_view_active = checked
        if self.current_image_cv is None:
            self.show_dpt_btn.setChecked(False)
            return

        if checked:
            depth_map = generate_dpt_depth_map(self.current_image_cv, self.dpt_processor, self.dpt_model, self.device)
            if depth_map is not None:
                # Normalize for visualization
                norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_img_bgr = cv2.cvtColor(norm_depth, cv2.COLOR_GRAY2BGR)
                self.display_image(depth_img_bgr)
                self.show_dpt_btn.setText("Show Real Image")
            else:
                self.status_label.setText("Failed to generate DPT depth map.")
                self.show_dpt_btn.setChecked(False)
        else:
            self.display_image(self.current_image_cv)
            self.draw_existing_annotations()
            self.show_dpt_btn.setText("Show DPT Depth")

    def start_training(self):
        if not self.image_directory or not self.camera_id:
            QMessageBox.warning(self, "Not Ready", "Please load a directory and set a camera name first.")
            return
        if self.dpt_model is None:
            QMessageBox.warning(self, "DPT Model Error", "The DPT model is not loaded. Cannot start training.")
            return
            
        self.save_annotations_for_current_image()
        self.set_buttons_enabled(False)
        self.metrics_label.setText("Training in progress...")
        
        self.training_thread = TrainingThread(
            self.image_files, self.annotations_by_image, self.dpt_processor,
            self.dpt_model, self.device, self.image_directory, self.camera_id
        )
        self.training_thread.update_status.connect(self.status_label.setText)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.final_metrics_report.connect(self.metrics_label.setText)
        self.training_thread.plot_generated.connect(self.show_plot_window)
        self.training_thread.start()

    def on_training_finished(self):
        self.set_buttons_enabled(True)
        self.status_label.setText("Training process finished.")
        QMessageBox.information(self, "Training Complete", "Model training has finished. The model and scaler have been saved.")
        self.training_thread = None

    def show_plot_window(self, plot_path: str):
        if self.plot_window and self.plot_window.isVisible():
             self.plot_window.close()
        self.plot_window = PlotWindow(plot_path, self)
        self.plot_window.show()

    def set_buttons_enabled(self, enabled: bool):
        """Enables or disables all major UI controls."""
        self.load_dir_btn.setEnabled(enabled)
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.train_btn.setEnabled(enabled)
        self.show_dpt_btn.setEnabled(enabled)
        self.delete_annotation_btn.setEnabled(enabled)
        self.thumbnail_list_widget.setEnabled(enabled)

    def closeEvent(self, event):
        """Ensure annotations are saved when closing the application."""
        self.save_annotations_for_current_image()
        event.accept()

def main():
    """Main function to initialize and run the PyQt application."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = AnnotationTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()