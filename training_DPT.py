
import sys
import os
import json
import joblib
import tempfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from typing import Dict, List, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QFrame, QProgressBar, QTextEdit
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

# --- ML Dependencies ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model 
    from tensorflow.keras.layers import Input, Dense, Dropout 
    from tensorflow.keras.callbacks import Callback 
    from tensorflow.keras.metrics import MeanAbsoluteError 
except ImportError:
    tf = None

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
except ImportError:
    mean_squared_error = None

try:
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    import torch
except ImportError:
    torch = None

# --- Constants ---
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp')
IMAGE_CROP_FACTOR = 0.92593 
DPT_MODEL_NAME = "Intel/dpt-hybrid-midas"
MODEL_FILENAME_TEMPLATE = "{camera_id}_distance_model.joblib" # Simplified to just joblib for linear reg

# --- Helper Functions ---
def generate_dpt_depth_map(cv_img: np.ndarray, processor, model, device: str) -> Optional[np.ndarray]:
    """Generates a depth map from an OpenCV image using a DPT model."""
    # Bug F2 fix: check numpy array and other deps separately to avoid
    # 'ValueError: The truth value of an array is ambiguous' when cv_img is a valid array.
    if cv_img is None or processor is None or model is None or torch is None:
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

# --- Custom Widgets ---

class ScaledLabel(QLabel):
    """A QLabel that automatically scales its pixmap to fit its size while keeping aspect ratio."""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.original_pixmap = None
        self.setMinimumSize(200, 150)
        self.setAlignment(Qt.AlignCenter)

    def setPixmap(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        self.update_pixmap()

    def update_pixmap(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            scaled = self.original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event):
        self.update_pixmap()
        super().resizeEvent(event)


# --- Worker Threads ---

class TrainingThread(QThread):
    """Worker thread for training the model to prevent UI freezing."""
    update_status = pyqtSignal(str)
    progress_update = pyqtSignal(int, int) # (processed, total)
    training_finished = pyqtSignal()
    final_metrics_report = pyqtSignal(str)
    plot_generated = pyqtSignal(str)
    depth_plot_generated = pyqtSignal(str) 
    training_results_available = pyqtSignal(object)

    def __init__(self, image_files: List[str], annotations: Dict[str, List], dpt_processor, dpt_model, device: str, save_dir: str):
        super().__init__()
        self.image_files = image_files
        self.annotations = annotations
        self.processor = dpt_processor
        self.dpt_model = dpt_model
        self.device = device
        self.save_dir = save_dir
        self.is_running = True

    def get_depth_for_polygon(self, depth_map: np.ndarray, polygon_coords: List[List[int]]) -> Optional[float]:
        """Extracts a representative depth value (median) from a polygon area."""
        if not polygon_coords or len(polygon_coords) < 3:
            return None

        # Create a mask for the polygon
        h, w = depth_map.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(polygon_coords, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)

        # Extract depth values where mask is 1
        valid_depths = depth_map[mask == 1]
        valid_depths = valid_depths[np.isfinite(valid_depths)]

        return np.median(valid_depths) if valid_depths.size > 0 else None

    def run(self):
        """The main training logic executed in the worker thread."""
        if any(lib is None for lib in [mean_absolute_error, r2_score, joblib]):
            self.update_status.emit("Error: Scikit-learn or Joblib not installed. Training aborted.")
            self.training_finished.emit()
            return

        try:
            self.update_status.emit("Preparing data for training...")
            X_features, y_labels = [], []
            image_paths_for_features = [] # Track image paths for results
            depth_map_cache = {}

            total_annotations_count = sum(len(anns) for anns in self.annotations.values())
            if total_annotations_count == 0:
                self.update_status.emit("No annotations found. Please annotate images first.")
                self.training_finished.emit()
                return

            processed_count = 0
            for img_path, anns in self.annotations.items():
                for ann in anns:
                    if img_path in depth_map_cache:
                        depth_map = depth_map_cache[img_path]
                    else:
                        img_full = cv2.imread(img_path)
                        if img_full is None: 
                            processed_count += 1
                            self.progress_update.emit(processed_count, total_annotations_count)
                            continue
                        h, _, _ = img_full.shape
                        img_cropped = img_full[0:int(h * IMAGE_CROP_FACTOR), :]
                        depth_map = generate_dpt_depth_map(img_cropped, self.processor, self.dpt_model, self.device)
                        depth_map_cache[img_path] = depth_map

                    if depth_map is None: 
                        processed_count += 1
                        self.progress_update.emit(processed_count, total_annotations_count)
                        continue

                    # Handle both old (bbox) and new (polygon) formats
                    coords = ann["coordinates"]
                    bbox_depth = None
                    
                    if isinstance(coords[0], list): # Polygon: [[x,y], [x,y], ...]
                            bbox_depth = self.get_depth_for_polygon(depth_map, coords)
                    else: # Legacy BBox: [x1, y1, x2, y2]
                            x1, y1, x2, y2 = coords
                            poly_pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                            bbox_depth = self.get_depth_for_polygon(depth_map, poly_pts)

                    if bbox_depth is not None:
                        # DPT output is often inversely proportional to distance (disparity).
                        # We use 1/depth as the primary feature for Linear Regression.
                        inverse_depth = 1.0 / (bbox_depth + 1e-6)
                        X_features.append([inverse_depth]) 
                        y_labels.append(ann["distance_meters"])
                        image_paths_for_features.append(img_path)

                    processed_count += 1
                    self.progress_update.emit(processed_count, total_annotations_count)

            if not X_features:
                self.update_status.emit("No valid features could be extracted. Training aborted.")
                self.training_finished.emit()
                return

            X = np.array(X_features, dtype=np.float32)
            y = np.array(y_labels, dtype=np.float32)
            
            # Keep raw depth for plotting/table (re-calculate from inverse)
            X_raw_depth = 1.0 / (X[:, 0] + 1e-6)

            self.update_status.emit(f"Extracted {len(X)} features. Training Linear Regression model...")

            # Train Simple Linear Regression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            # Evaluation Metrics
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)

            metrics_text = f"MAE: {mae:.2f}m | RMSE: {rmse:.2f}m | R²: {r2:.2f}"
            self.final_metrics_report.emit(metrics_text)
            self.update_status.emit("Training complete. Generating plots...")

            # Generate Plot
            plot_path = None
            depth_plot_path = None

            if plt:
                # Plot 1: Actual vs. Predicted Distance
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    plot_path = temp_file.name
                
                plt.figure(figsize=(8, 6))
                plt.scatter(y, y_pred, alpha=0.7, label="Predictions")
                min_val = min(y.min(), y_pred.min())
                max_val = max(y.max(), y_pred.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")
                plt.title(f"Actual vs. Predicted Distance")
                plt.xlabel("Actual Distance (meters)")
                plt.ylabel("Predicted Distance (meters)")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                self.plot_generated.emit(plot_path)

                # Plot 2: Calibration: DPT Depth vs. Distance
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    depth_plot_path = temp_file.name

                plt.figure(figsize=(8, 6))
                plt.scatter(X_raw_depth, y, alpha=0.7, label="Actual Data", color='blue')
                plt.scatter(X_raw_depth, y_pred, alpha=0.7, label="Model Prediction", color='red', marker='x')
                
                sort_idx = np.argsort(X_raw_depth)
                plt.plot(X_raw_depth[sort_idx], y_pred[sort_idx], color='red', alpha=0.5, linestyle='--', label="Fit Curve")

                plt.title(f"Calibration: DPT Depth vs. Distance")
                plt.xlabel("DPT Depth Value")
                plt.ylabel("Distance (meters)")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(depth_plot_path)
                plt.close()
                self.depth_plot_generated.emit(depth_plot_path)

            # Emit detailed results
            results = {
                'y_true': y,
                'y_pred': y_pred,
                'X_raw': X_raw_depth, 
                'image_paths': image_paths_for_features,
                'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
                'plot_path': plot_path,
                'depth_plot_path': depth_plot_path
            }
            self.training_results_available.emit(results)

            # Bug J fix: MODEL_FILENAME_TEMPLATE contained an unformatted '{camera_id}' placeholder.
            # Use 'default' as the camera_id. Pass a real ID here if available in future.
            model_filename = MODEL_FILENAME_TEMPLATE.format(camera_id="default")
            model_path = os.path.join(self.save_dir, model_filename)
            print(f"Attempting to save model to: {model_path}")
            joblib.dump(model, model_path)
            
            self.update_status.emit(f"Model saved to {model_path}")
            self.training_finished.emit()

        except Exception as e:
            print(f"TRAINING ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.update_status.emit(f"Error: {str(e)}")
            self.training_finished.emit()


class TrainingTool(QWidget):
    """Main widget for the standalone Training Tool."""
    def __init__(self, parent=None, shared_dpt_processor=None, shared_dpt_model=None, shared_device=None):
        super().__init__(parent)
        
        self.image_directory = None
        self.image_files = []
        self.annotations_by_image = {}
        self.model_output_directory = None

        # Opt-1: Use shared model if provided to avoid loading a second copy.
        if shared_dpt_processor is not None and shared_dpt_model is not None:
            self.dpt_processor = shared_dpt_processor
            self.dpt_model = shared_dpt_model
            self.device = shared_device or "cpu"
        else:
            self.dpt_processor = None
            self.dpt_model = None
            self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        
        self.training_thread = None

        self.init_ui()
        # Only load DPT model internally if we did not receive a shared one.
        if self.dpt_model is None:
            self.load_dpt_inference_model()
        elif self.dpt_model == "async":
            self.status_label.setText("AI model is loading in the background...")
            self.train_btn.setEnabled(False)
        else:
            self.status_label.setText(f"DPT model ready (shared, device={self.device}).")
            self.train_btn.setEnabled(False)  # Still need images to enable

    def load_dpt_inference_model(self):
        """Loads the pre-trained DPT model from Hugging Face."""
        if torch is None:
            self.status_label.setText("DPT dependencies not installed.")
            return
        try:
            local_model = 'dpt-model'
            try:
                base_path = sys._MEIPASS
            except AttributeError:
                base_path = os.path.abspath(".")
            model_path = os.path.join(base_path, local_model)

            if os.path.exists(model_path):
                model_src = model_path
                self.status_label.setText(f"Loading DPT model locally from {model_src}...")
            else:
                model_src = DPT_MODEL_NAME
                self.status_label.setText(f"Loading DPT model '{DPT_MODEL_NAME}' from Hub...")

            self.dpt_processor = DPTImageProcessor.from_pretrained(model_src)
            self.dpt_model = DPTForDepthEstimation.from_pretrained(model_src).to(self.device)
            self.dpt_model.eval()
            self.status_label.setText(f"DPT model loaded successfully on {self.device}.")
            self.train_btn.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"Failed to load DPT model: {e}")
            self.train_btn.setEnabled(False)

    def create_step_card(self, step_num: str, title: str, button: QPushButton, info_label: QLabel) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            QLabel {
                border: none;
                background-color: transparent;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(8)
        
        # Step header
        header_layout = QHBoxLayout()
        step_lbl = QLabel(step_num)
        step_lbl.setStyleSheet("font-weight: bold; color: #c82828; font-size: 13px;")
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-weight: bold; color: #333333; font-size: 13px;")
        header_layout.addWidget(step_lbl)
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Button
        layout.addWidget(button)
        
        # Info Label
        info_label.setStyleSheet("color: #666666; font-size: 12px;")
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(info_label)
        
        return card

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- Title and Description + Clear Button ---
        header_widget = QWidget()
        header_outer = QHBoxLayout(header_widget)
        header_outer.setContentsMargins(0, 0, 0, 5)
        header_outer.setSpacing(0)

        # Left: title + description
        title_block = QWidget()
        header_layout = QVBoxLayout(title_block)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        title_label = QLabel("Calibration & Model Training")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1e1e1e;")
        desc_label = QLabel("Calibrate the DPT depth map output to predict actual distance using annotations as reference data.")
        desc_label.setStyleSheet("font-size: 13px; color: #666666;")
        header_layout.addWidget(title_label)
        header_layout.addWidget(desc_label)
        header_outer.addWidget(title_block, 1)

        # Right: Clear button
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setFixedSize(90, 32)
        self.clear_btn.clicked.connect(self.clear_all)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                color: #555555;
                border: 1px solid #cccccc;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
                color: #333333;
                border: 1px solid #bbbbbb;
            }
            QPushButton:pressed {
                background-color: #bbbbbb;
                color: #1e1e1e;
            }
        """)
        header_outer.addWidget(self.clear_btn, 0, Qt.AlignTop | Qt.AlignRight)

        main_layout.addWidget(header_widget)

        # --- Top Steps Config Panel ---
        steps_widget = QWidget()
        steps_layout = QHBoxLayout(steps_widget)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(15)

        # Step 1 Card: Load Directory
        self.load_dir_btn = QPushButton("Select Image Directory")
        self.load_dir_btn.clicked.connect(self.load_directory)
        self.dir_label = QLabel("No directory selected")
        card1 = self.create_step_card("STEP 1", "DATASET", self.load_dir_btn, self.dir_label)
        steps_layout.addWidget(card1, 1)

        # Step 2 Card: Set Output Folder
        self.set_model_out_btn = QPushButton("Select Model Output Folder")
        self.set_model_out_btn.clicked.connect(self.set_model_output_directory)
        self.model_out_label = QLabel("Output: Default (Image Dir)")
        card2 = self.create_step_card("STEP 2", "SAVE LOCATION", self.set_model_out_btn, self.model_out_label)
        steps_layout.addWidget(card2, 1)

        # Step 3 Card: Run Calibration
        self.train_btn = QPushButton("Start Model Training")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #c82828;
                color: white;
                font-weight: bold;
                border: 1px solid #c82828;
            }
            QPushButton:hover {
                background-color: #d83838;
                border: 1px solid #d83838;
            }
            QPushButton:disabled {
                background-color: #f0f0f0;
                color: #aaaaaa;
                border: 1px solid #dddddd;
            }
        """)
        self.train_status_label = QLabel("Awaiting dataset...")
        card3 = self.create_step_card("STEP 3", "CALIBRATION", self.train_btn, self.train_status_label)
        steps_layout.addWidget(card3, 1)

        main_layout.addWidget(steps_widget)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                text-align: center;
                background-color: #f9f9f9;
                height: 22px;
                font-weight: bold;
                color: #333333;
            }
            QProgressBar::chunk {
                background-color: rgb(80, 200, 120); /* Sleek Green */
                border-radius: 5px;
            }
        """)
        self.progress_bar.hide() # Hidden initially
        main_layout.addWidget(self.progress_bar)

        # --- Content Area (Splitter with Log and Plots) ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #e0e0e0;
                width: 2px;
            }
        """)
        
        # Log Panel
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(8)
        
        log_header = QLabel("Training Console Log")
        log_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #333333;")
        log_layout.addWidget(log_header)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Courier New", 12))
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #f7f7f7;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 10px;
                color: #2b2b2b;
            }
        """)
        self.log_output.setHtml("<span style='color: #888888;'>Console initialized. Waiting to start calibration...</span>")
        log_layout.addWidget(self.log_output)
        
        log_frame = QFrame()
        log_frame.setObjectName("GlassPanel")
        log_frame_layout = QVBoxLayout(log_frame)
        log_frame_layout.addWidget(log_widget)
        splitter.addWidget(log_frame)
        
        # Plot Panel
        plot_panel = QFrame()
        plot_panel.setObjectName("GlassPanel")
        plot_panel_layout = QVBoxLayout(plot_panel)
        plot_panel_layout.setContentsMargins(10, 10, 10, 10)
        plot_panel_layout.setSpacing(8)
        
        plot_header = QLabel("Calibration Results Visualizations")
        plot_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #333333;")
        plot_panel_layout.addWidget(plot_header)
        
        plots_container = QWidget()
        plots_container_layout = QHBoxLayout(plots_container)
        plots_container_layout.setContentsMargins(0, 0, 0, 0)
        plots_container_layout.setSpacing(15)
        
        # Plot 1 Card
        plot_card_1 = QFrame()
        plot_card_1.setStyleSheet("QFrame { background-color: #fafafa; border: 1px solid #e5e5e5; border-radius: 8px; }")
        pc_layout_1 = QVBoxLayout(plot_card_1)
        pc_layout_1.addWidget(QLabel("Distance Prediction Accuracy"))
        self.plot_label_1 = ScaledLabel("Prediction Plot")
        self.plot_label_1.setStyleSheet("color: #888888; font-size: 13px; border: none;")
        self.plot_label_1.setText("Accuracy plot will be generated here.\n(Awaiting training data...)")
        pc_layout_1.addWidget(self.plot_label_1, 1)
        plots_container_layout.addWidget(plot_card_1)
        
        # Plot 2 Card
        plot_card_2 = QFrame()
        plot_card_2.setStyleSheet("QFrame { background-color: #fafafa; border: 1px solid #e5e5e5; border-radius: 8px; }")
        pc_layout_2 = QVBoxLayout(plot_card_2)
        pc_layout_2.addWidget(QLabel("DPT Depth vs. Distance Calibration"))
        self.plot_label_2 = ScaledLabel("Calibration Curve")
        self.plot_label_2.setStyleSheet("color: #888888; font-size: 13px; border: none;")
        self.plot_label_2.setText("Calibration plot will be generated here.\n(Awaiting training data...)")
        pc_layout_2.addWidget(self.plot_label_2, 1)
        plots_container_layout.addWidget(plot_card_2)
        
        plot_panel_layout.addWidget(plots_container, 1)
        splitter.addWidget(plot_panel)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter, 1)

        # --- Status Bar ---
        self.status_label = QLabel("Please load a directory with annotated images.")
        self.status_label.setObjectName("StatusLabel")
        main_layout.addWidget(self.status_label, 0)

    def load_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if not directory:
            return

        self.image_directory = directory
        self.image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)])
        
        # Load annotations
        count = 0
        self.annotations_by_image.clear()
        for img_path in self.image_files:
            json_path = img_path + ".json"
            if os.path.exists(json_path):
                # Bug K fix: bare 'except: pass' was swallowing all errors silently.
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if data:
                            self.annotations_by_image[img_path] = data
                            count += 1
                except (json.JSONDecodeError, OSError) as e:
                    print(f"Warning: Skipped corrupt or unreadable annotation file '{json_path}': {e}")
        
        self.dir_label.setText(f"{os.path.basename(directory)} ({count} annotated images)")
        self.status_label.setText(f"Loaded {len(self.image_files)} images, {count} with annotations.")
        
        if count > 0 and self.dpt_model is not None:
            self.train_btn.setEnabled(True)
            self.train_status_label.setText("Ready to train.")
        else:
            self.train_btn.setEnabled(False)
            self.train_status_label.setText("No annotations found. Annotate first.")

    def start_training(self):
        if not self.image_directory:
            return
        
        self.train_btn.setEnabled(False)
        self.load_dir_btn.setEnabled(False)
        self.set_model_out_btn.setEnabled(False)
        
        self.log_output.clear()
        self.log_output.append("<span style='color: #c82828; font-weight: bold;'>Starting model training process...</span>")
        
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        output_dir = self.model_output_directory if self.model_output_directory else self.image_directory
        
        self.training_thread = TrainingThread(
            self.image_files, 
            self.annotations_by_image, 
            self.dpt_processor, 
            self.dpt_model, 
            self.device, 
            output_dir
        )
        self.training_thread.update_status.connect(self.update_log)
        self.training_thread.progress_update.connect(self.update_progress)
        self.training_thread.final_metrics_report.connect(self.update_log)
        self.training_thread.plot_generated.connect(lambda p: self.display_plot(p, self.plot_label_1))
        self.training_thread.depth_plot_generated.connect(lambda p: self.display_plot(p, self.plot_label_2))
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.start()

    def update_progress(self, current, total):
        if total > 0:
            pct = int((current / total) * 100)
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"Processing annotation {current}/{total} ({pct}%)")

    def update_log(self, message):
        self.log_output.append(message)
        self.status_label.setText(message)

    def display_plot(self, plot_path, label_widget):
        pixmap = QPixmap(plot_path)
        if not pixmap.isNull():
            label_widget.setPixmap(pixmap)

    def on_training_finished(self):
        self.train_btn.setEnabled(True)
        self.load_dir_btn.setEnabled(True)
        self.set_model_out_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Training completed.")
        self.train_status_label.setText("Training completed.")
        self.status_label.setText("Training finished.")

    def set_model_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Model Output Directory")
        if directory:
            self.model_output_directory = directory
            self.model_out_label.setText(f"Output: {os.path.basename(directory)}")

    def clear_all(self):
        """Resets all state and clears the UI back to its initial state."""
        # Stop any running training thread
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()

        # Reset state
        self.image_directory = None
        self.image_files = []
        self.annotations_by_image.clear()
        self.model_output_directory = None

        # Reset Step labels
        self.dir_label.setText("No directory selected")
        self.model_out_label.setText("Output: Default (Image Dir)")
        self.train_status_label.setText("Awaiting dataset...")

        # Reset buttons
        self.train_btn.setEnabled(False)
        self.load_dir_btn.setEnabled(True)
        self.set_model_out_btn.setEnabled(True)

        # Reset console log
        self.log_output.setHtml("<span style='color: #888888;'>Console initialized. Waiting to start calibration...</span>")

        # Reset progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.progress_bar.hide()

        # Reset plot placeholders
        self.plot_label_1.original_pixmap = None
        self.plot_label_1.clear()
        self.plot_label_1.setText("Accuracy plot will be generated here.\n(Awaiting training data...)")
        self.plot_label_2.original_pixmap = None
        self.plot_label_2.clear()
        self.plot_label_2.setText("Calibration plot will be generated here.\n(Awaiting training data...)")

        # Reset status bar
        self.status_label.setText("Cleared. Please load a directory with annotated images.")


