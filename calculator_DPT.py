########################################################
### Dev : Andaman Chankhao andmanchankhao@gmail.com ####
### Revised by Gemini ##################################
########################################################

import sys
import os
import cv2
import numpy as np
import csv
import joblib # Added joblib import
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGraphicsView, QGraphicsScene, QFileDialog,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QListWidget, QListWidgetItem, QGraphicsPixmapItem, QProgressDialog,
    QSplitter
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QIcon, QPen
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread

# --- Dependency Imports with User-Friendly Feedback ---
# These blocks provide clearer instructions if a library is missing.

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model # type: ignore
except ImportError:
    tf = None
    print("WARNING: TensorFlow not found. Please run 'pip install tensorflow'. Keras model loading will be disabled.")

try:
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    import torch
except ImportError:
    DPTForDepthEstimation, DPTImageProcessor, torch = None, None, None
    print("WARNING: 'transformers' or 'torch' not found. Please run 'pip install transformers torch'. DPT depth estimation will be disabled.")

try:
    import joblib
    from sklearn.preprocessing import StandardScaler
except ImportError:
    joblib, StandardScaler = None, None
    print("WARNING: 'scikit-learn' or 'joblib' not found. Please run 'pip install scikit-learn joblib'. Scaler loading will be disabled.")

# --- Constants ---
# Using constants makes the code cleaner and easier to modify.
APP_TITLE = "Wildlife Distance Calculator"
TABLE_HEADERS = ["ID", "Image", "Distance (m)", "Coordinates"]
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
MODEL_FILENAME_TEMPLATE = "{camera_id}_distance_model.joblib"

class CustomGraphicsView(QGraphicsView):
    """ A QGraphicsView subclass that emits signals for mouse press and move events. """
    mouse_pressed = pyqtSignal(object)
    mouse_moved = pyqtSignal(object)

    def mousePressEvent(self, event):
        self.mouse_pressed.emit(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.mouse_moved.emit(event)
        super().mouseMoveEvent(event)

    def resizeEvent(self, event):
        if self.scene():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)

class ResponsiveListWidget(QListWidget):
    """A QListWidget that automatically resizes its icons to fit the width."""
    def resizeEvent(self, event):
        width = event.size().width()
        # Calculate new icon size (subtracting scrollbar width and margins)
        new_size = width - 25 
        if new_size > 50: # Minimum size
            self.setIconSize(QSize(new_size, int(new_size * 0.75))) # Maintain aspect ratio roughly
        super().resizeEvent(event)

class DPTWorker(QThread):
    """ Worker thread exclusively for running the DPT model to generate a depth map. """
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, dpt_model, dpt_processor, device, image_cv):
        super().__init__()
        self.dpt_model = dpt_model
        self.dpt_processor = dpt_processor
        self.device = device
        self.image_cv = image_cv

    def run(self):
        try:
            img_rgb = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB)
            inputs = self.dpt_processor(images=img_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.dpt_model(**inputs)
                predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()
            
            # Resizing to original image dimensions for accurate coordinate mapping
            depth_map = cv2.resize(
                predicted_depth,
                (self.image_cv.shape[1], self.image_cv.shape[0]),
                interpolation=cv2.INTER_CUBIC # Use a higher quality interpolation
            )
            self.finished.emit(depth_map)
        except Exception as e:
            self.error.emit(f"Failed to generate depth map: {str(e)}")

class AutoDetectWorker(QThread):
    """Worker thread for auto-detecting animals and calculating distances."""
    progress_update = pyqtSignal(int, int, str) # current, total, status_message
    detection_found = pyqtSignal(str, list, float, float) # image_path, coords, distance, confidence
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, image_files, dpt_model, dpt_processor, distance_model, device):
        super().__init__()
        self.image_files = image_files
        self.dpt_model = dpt_model
        self.dpt_processor = dpt_processor
        self.distance_model = distance_model
        self.device = device
        self.yolo_model = None
        self.is_running = True

    def run(self):
        try:
            self.progress_update.emit(0, len(self.image_files), "Loading Object Detection Model...")
            # Load YOLOv5s from torch hub (pretrained on COCO)
            # COCO animal classes: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
            # Class IDs: 14-23 (roughly, need to check specific model mapping or use names)
            try:
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
                self.yolo_model.to(self.device)
                self.yolo_model.eval()
            except Exception as e:
                self.error.emit(f"Failed to load YOLOv5 model: {e}")
                return

            # COCO classes that are animals
            ANIMAL_CLASSES = [
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                'elephant', 'bear', 'zebra', 'giraffe'
            ]

            for i, img_path in enumerate(self.image_files):
                if not self.is_running: break
                
                self.progress_update.emit(i + 1, len(self.image_files), f"Processing {os.path.basename(img_path)}...")
                
                img_cv = cv2.imread(img_path)
                if img_cv is None: continue

                # 1. Object Detection
                results = self.yolo_model(img_cv)
                detections = results.pandas().xyxy[0] # Get results as pandas dataframe

                # Filter for animals
                animal_detections = detections[detections['name'].isin(ANIMAL_CLASSES)]
                
                if animal_detections.empty:
                    continue

                # 2. Generate Depth Map (Only if we have detections)
                # Re-use the generation logic (duplicated from DPTWorker for simplicity in this thread)
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                inputs = self.dpt_processor(images=img_rgb, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.dpt_model(**inputs)
                    predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()
                
                depth_map = cv2.resize(
                    predicted_depth,
                    (img_cv.shape[1], img_cv.shape[0]),
                    interpolation=cv2.INTER_CUBIC
                )

                # 3. Calculate Distance for each detection
                for _, row in animal_detections.iterrows():
                    # Calculate center of bounding box
                    x_center = int((row['xmin'] + row['xmax']) / 2)
                    y_center = int((row['ymin'] + row['ymax']) / 2)
                    
                    # Ensure within bounds
                    h, w = depth_map.shape
                    x_center = max(0, min(w-1, x_center))
                    y_center = max(0, min(h-1, y_center))

                    # Get depth and predict
                    depth_val = depth_map[y_center, x_center]
                    inverse_depth = 1.0 / (depth_val + 1e-6)
                    distance = self.distance_model.predict(np.array([[inverse_depth]], dtype=np.float32))[0]

                    self.detection_found.emit(img_path, [x_center, y_center], float(distance), row['confidence'])

            self.finished.emit()

        except Exception as e:
            self.error.emit(f"Error during auto-calculation: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        self.is_running = False
MODEL_FILENAME_TEMPLATE = "{camera_id}_distance_model.joblib"

class DistanceCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wildlife Distance Calculator (DPT)")
        self.setGeometry(100, 100, 1400, 900)

        # Model and data state
        # Model and data state
        self.distance_model = None
        self.image_files = []
        self.current_image_index = 0
        self.image_data = {} # Stores annotations: {image_path: [annotation_dict, ...]}
        self.next_annotation_id = 0
        self.camera_id = "default" # Default camera ID

        # Image and graphics state
        self.current_image_path = None
        self.current_image_cv = None
        self.current_depth_map = None # REVISED: Cache for the depth map
        self._pending_coords_for_prediction = None
        self.current_pixmap_item = None
        self.is_dpt_view_active = False

        # DPT model state
        self.dpt_processor = None
        self.dpt_model = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        
        # UI element references
        self.thumbnail_items = {}

        # Initialize UI and models
        self.init_ui()
        self.load_dpt_inference_model()
        self.update_navigation_buttons_state()

    def init_ui(self):
        """Initializes the main UI layout and widgets."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Create and add panels to the main layout ---
        # --- Create and add panels to the main layout ---
        splitter = QSplitter(Qt.Horizontal)
        
        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial sizes (approximate 1:4 ratio)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        main_layout.addWidget(splitter)

        self.status_label = QLabel("Ready. Please load a Keras model and a scaler file.")
        self.statusBar().addWidget(self.status_label)

    def _create_left_panel(self):
        """Creates the left panel widget containing controls and thumbnails."""
        panel = QWidget()
        panel.setObjectName("GlassPanel") # Apply glass style
        layout = QVBoxLayout(panel)
        # panel.setMaximumWidth(280) # Removed to allow resizing

        # Image Directory Controls
        self.load_dir_btn = QPushButton("Load Image Directory")
        self.load_dir_btn.clicked.connect(self.load_image_directory)
        layout.addWidget(self.load_dir_btn)

        # Model/Scaler Loading Group
        model_scaler_group = QWidget()
        model_scaler_layout = QVBoxLayout(model_scaler_group)
        # Removed inline style to use QSS


        self.load_model_btn = QPushButton("Load Distance Model")
        self.load_model_btn.clicked.connect(self.load_distance_model)
        self.model_status_label = QLabel("Model: <font color='red'>Not Loaded</font>")
        
        model_scaler_layout.addWidget(self.load_model_btn)
        model_scaler_layout.addWidget(self.model_status_label)
        layout.addWidget(model_scaler_group)

        # Auto-Calculate Button
        self.auto_calc_btn = QPushButton("Auto-Calculate All")
        self.auto_calc_btn.clicked.connect(self.auto_calculate_all)
        self.auto_calc_btn.setEnabled(False) # Disabled until everything is loaded
        layout.addWidget(self.auto_calc_btn)

        # Thumbnail List
        self.thumbnail_list_widget = ResponsiveListWidget()
        self.thumbnail_list_widget.setIconSize(QSize(100, 80))
        self.thumbnail_list_widget.itemClicked.connect(self.thumbnail_clicked)
        layout.addWidget(self.thumbnail_list_widget)

        return panel

    def _create_right_panel(self):
        """Creates the right panel containing the image viewer and results table."""
        panel = QWidget()
        panel.setObjectName("GlassPanel") # Apply glass style
        layout = QVBoxLayout(panel)

        content_layout = QHBoxLayout()
        image_view_widget = self._create_image_view_widget()
        table_widget = self._create_table_widget()

        content_layout.addWidget(image_view_widget, 3) # Image view takes 3 parts
        content_layout.addWidget(table_widget, 2)    # Table takes 2 parts
        layout.addLayout(content_layout)

        return panel

    def _create_image_view_widget(self):
        """Creates the widget for displaying the main image and its controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.current_image_name_label = QLabel("No Image Loaded")
        self.current_image_name_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.current_image_name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.current_image_name_label)

        self.graphics_view = CustomGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.mouse_pressed.connect(self.handle_mouse_press)
        self.graphics_view.mouse_moved.connect(self.handle_mouse_move)
        layout.addWidget(self.graphics_view)

        # Navigation Controls
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self.load_previous_image)
        self.show_dpt_btn = QPushButton("Toggle Depth Map")
        self.show_dpt_btn.clicked.connect(self.toggle_dpt_view)
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.load_next_image)

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.show_dpt_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)

        return widget

    def _create_table_widget(self):
        """Creates the widget for the results table and its controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(len(TABLE_HEADERS))
        self.results_table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.itemSelectionChanged.connect(self.update_delete_button_state)
        layout.addWidget(self.results_table)

        # Table buttons
        button_layout = QHBoxLayout()
        self.delete_row_btn = QPushButton("Delete Selected")
        self.delete_row_btn.clicked.connect(self.delete_selected_row)
        self.export_csv_btn = QPushButton("Export All to CSV")
        self.export_csv_btn.clicked.connect(self.export_to_csv)

        button_layout.addWidget(self.delete_row_btn)
        button_layout.addWidget(self.export_csv_btn)
        layout.addLayout(button_layout)

        return widget

    def load_dpt_inference_model(self):
        """Loads the DPT model from Hugging Face."""
        if DPTForDepthEstimation is None:
            self.status_label.setText("DPT dependencies not installed. Calculation disabled.")
            return

        try:
            model_name = "Intel/dpt-hybrid-midas"
            self.dpt_processor = DPTImageProcessor.from_pretrained(model_name)
            self.dpt_model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
            self.dpt_model.eval() # Set model to evaluation mode
            self.status_label.setText(f"DPT model loaded on {self.device}. Ready for use.")
        except Exception as e:
            self.status_label.setText(f"Failed to load DPT model: {e}")
            QMessageBox.critical(self, "DPT Model Error", f"Could not load the DPT model from Hugging Face. Please check your internet connection.\n\nError: {e}")

    def load_distance_model(self):
        """Loads the trained distance estimation model (Linear Regression)."""
        # Try to auto-load based on camera ID
        model_path = os.path.join(os.path.dirname(self.image_files[0]) if self.image_files else ".", MODEL_FILENAME_TEMPLATE.format(camera_id=self.camera_id))
        
        if os.path.exists(model_path):
            try:
                self.distance_model = joblib.load(model_path)
                self.model_status_label.setText(f"Model: <font color='green'><b>Loaded ({os.path.basename(model_path)})</b></font>")
                self.status_label.setText(f"Distance model loaded: {os.path.basename(model_path)}")
                self.auto_calc_btn.setEnabled(True)
                return
            except Exception as e:
                print(f"Auto-load failed: {e}")

        # Fallback to manual selection
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Distance Model", "", "Joblib Models (*.joblib)")
        if file_path:
            try:
                self.distance_model = joblib.load(file_path)
                self.model_status_label.setText(f"Model: <font color='green'><b>Loaded</b></font>")
                self.status_label.setText("Distance model loaded successfully.")
                self.auto_calc_btn.setEnabled(True)
            except Exception as e:
                self.distance_model = None
                self.model_status_label.setText("Model: <font color='red'>Load Failed</font>")
                QMessageBox.critical(self, "Load Error", f"Failed to load distance model:\n{str(e)}")
        else:
            self.distance_model = None
            self.model_status_label.setText("Model: <font color='red'>Not Loaded</font>")
            self.auto_calc_btn.setEnabled(False)

    def load_image_directory(self):
        """Loads all supported images from a user-selected directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if not directory: return

        self.image_files = sorted([
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
        ])

        # Reset state
        self.thumbnail_list_widget.clear()
        self.thumbnail_items.clear()
        self.image_data.clear()

        if self.image_files:
            for img_path in self.image_files:
                self.add_thumbnail(img_path)
            self.current_image_index = 0
            self.display_image(self.image_files[self.current_image_index])
            self.status_label.setText(f"Loaded {len(self.image_files)} images.")
            
            # Auto-detect model
            self.check_and_autoload_model(directory)
        else:
            self.status_label.setText("No supported image files found in the selected directory.")
        
        self.update_navigation_buttons_state()

    def check_and_autoload_model(self, directory):
        """Scans the directory for a .joblib distance model and auto-loads it."""
        try:
            files = os.listdir(directory)
            # Prioritize files ending with 'distance_model.joblib' (covers both 'distance_model.joblib' and '{id}_distance_model.joblib')
            model_files = [f for f in files if f.endswith('distance_model.joblib')]
            
            # If none found, look for any .joblib file
            if not model_files:
                model_files = [f for f in files if f.endswith('.joblib')]
            
            if model_files:
                # Load the first matching file
                model_path = os.path.join(directory, model_files[0])
                self.distance_model = joblib.load(model_path)
                self.model_status_label.setText(f"Model: <font color='green'><b>Loaded ({model_files[0]})</b></font>")
                self.status_label.setText(f"Auto-loaded model: {model_files[0]}")
                self.auto_calc_btn.setEnabled(True)
            else:
                self.status_label.setText("No model found in directory. Please load manually.")
        except Exception as e:
            print(f"Failed to auto-load model: {e}")
            self.status_label.setText(f"Auto-load failed: {e}")

    def add_thumbnail(self, image_path):
        """Creates and adds a thumbnail to the list widget."""
        pixmap = QPixmap(image_path)
        icon = QIcon(pixmap.scaled(self.thumbnail_list_widget.iconSize(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        item = QListWidgetItem(icon, os.path.basename(image_path))
        item.setData(Qt.UserRole, image_path)
        item.setToolTip(os.path.basename(image_path))
        self.thumbnail_list_widget.addItem(item)
        self.thumbnail_items[image_path] = item

    def thumbnail_clicked(self, item):
        """Handles clicks on a thumbnail item."""
        image_path = item.data(Qt.UserRole)
        if image_path and os.path.exists(image_path):
            try:
                index = self.image_files.index(image_path)
                if index != self.current_image_index:
                    self.current_image_index = index
                    self.display_image(self.image_files[self.current_image_index])
            except ValueError:
                self.status_label.setText("Error: Image path not found in list.")

    def display_image(self, image_path):
        """Displays the specified image in the graphics view and updates state."""
        self.graphics_scene.clear()
        self.is_dpt_view_active = False
        self.show_dpt_btn.setText("Show Depth Map")
        self.current_image_path = image_path
        self.current_depth_map = None # REVISED: Invalidate depth map cache

        self.current_image_cv = cv2.imread(image_path)
        if self.current_image_cv is None:
            self.status_label.setText(f"Error loading image: {os.path.basename(image_path)}")
            QMessageBox.critical(self, "Image Load Error", f"Could not read the image file:\n{image_path}")
            return

        h, w, ch = self.current_image_cv.shape
        q_image = QImage(self.current_image_cv.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.current_pixmap_item = self.graphics_scene.addPixmap(QPixmap.fromImage(q_image))
        self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        self.current_image_name_label.setText(f"{os.path.basename(image_path)}")
        self.status_label.setText("Click on a point to measure distance.")
        
        self.draw_all_annotations_for_current_image()
        self.update_results_table()
        self.update_thumbnail_status()
        self.update_navigation_buttons_state()

    def update_thumbnail_status(self):
        """Updates the visual state of thumbnails (selection and background color)."""
        for path, item in self.thumbnail_items.items():
            has_data = path in self.image_data and bool(self.image_data[path])
            item.setBackground(QColor(80, 200, 120) if has_data else Qt.transparent)
            
            if path == self.current_image_path:
                item.setSelected(True)
                self.thumbnail_list_widget.scrollToItem(item, QListWidget.PositionAtCenter)
            else:
                item.setSelected(False)

    def draw_annotation_on_scene(self, annotation_data):
        """Draws a single annotation (point and text) on the graphics scene."""
        coords = annotation_data["coords"]
        x, y = coords

        pen_color = QColor(255, 0, 0, 200) # Semi-transparent red
        text_color = QColor(255, 0, 0)
        
        ann_id = annotation_data["id"]
        distance = annotation_data["predicted_distance"]
        display_text = f"ID:{ann_id} | {distance:.2f}m"

        # Draw a cross instead of "X" for better centering
        pen = QPen(pen_color, 6)
        line1 = self.graphics_scene.addLine(x - 10, y, x + 10, y, pen)
        line2 = self.graphics_scene.addLine(x, y - 10, x, y + 10, pen)

        text_item = self.graphics_scene.addText(display_text)
        text_item.setDefaultTextColor(text_color)
        text_item.setFont(QFont("Arial", 52, QFont.Bold))
        text_item.setPos(x + 15, y - 30) # Position text relative to the point

        annotation_data['graphics_items'] = [line1, line2, text_item]

    def draw_all_annotations_for_current_image(self):
        """Clears and redraws all annotations for the currently displayed image."""
        if self.current_image_path in self.image_data:
            for annotation in self.image_data[self.current_image_path]:
                # First, ensure old items are removed if they exist
                for item in annotation.get('graphics_items', []):
                    if item in self.graphics_scene.items():
                        self.graphics_scene.removeItem(item)
                
                self.draw_annotation_on_scene(annotation)

    def handle_mouse_press(self, event):
        """Handles a left-click on the image to start a distance calculation."""
        if self.is_dpt_view_active:
            self.status_label.setText("Cannot select points on the depth map. Please switch to the real image.")
            return
        if not all([self.current_image_cv is not None, self.distance_model]):
            QMessageBox.warning(self, "Prerequisites Missing", "Please ensure an image directory and distance model are loaded.")
            return

        if event.button() == Qt.LeftButton:
            scene_pos = self.graphics_view.mapToScene(event.pos())
            coords = [int(scene_pos.x()), int(scene_pos.y())]

            # Check if click is within image bounds
            h, w, _ = self.current_image_cv.shape
            if not (0 <= coords[0] < w and 0 <= coords[1] < h):
                self.status_label.setText("Clicked outside image bounds.")
                return

            self.calculate_distance_at_point(coords)

    def calculate_distance_at_point(self, coords):
        """Initiates the distance calculation process for a given coordinate."""
        # REVISED: Major logic change. First get depth map, then predict.
        if self.current_depth_map is not None:
            # If depth map is cached, predict directly
            self.predict_distance_from_depth(coords, self.current_depth_map)
        else:
            # If not cached, run DPT worker first
            self.status_label.setText("Generating depth map (first click on this image)...")
            self.progress_dialog = QProgressDialog("Generating Depth Map...", "Cancel", 0, 0, self)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.show()

            self.dpt_worker = DPTWorker(self.dpt_model, self.dpt_processor, self.device, self.current_image_cv.copy())
            
            # Store coords to use after DPT is done
            self._pending_coords_for_prediction = coords 
            
            self.dpt_worker.finished.connect(self.on_depth_map_generated)
            self.dpt_worker.error.connect(self.on_dpt_error)
            self.dpt_worker.start()

    def on_depth_map_generated(self, depth_map):
        """Callback for when the DPT worker successfully finishes."""
        self.progress_dialog.close()
        self.current_depth_map = depth_map # Cache the result
        
        if self._pending_coords_for_prediction is not None:
            self.status_label.setText("Depth map generated. Predicting distance...")
            self.predict_distance_from_depth(self._pending_coords_for_prediction, depth_map)
            self._pending_coords_for_prediction = None # Clean up
        else:
            self.status_label.setText("Depth map generated.")
            # We triggered this from toggle_dpt_view, so now we should actually toggle the view
            self.toggle_dpt_view()

    def on_dpt_error(self, error_message):
        """Callback for when the DPT worker encounters an error."""
        self.progress_dialog.close()
        QMessageBox.critical(self, "DPT Error", error_message)
        self.status_label.setText("Failed to generate depth map.")

    def predict_distance_from_depth(self, coords, depth_map):
        """Uses the generated depth map and regression model to predict distance."""
        x, y = coords
        depth_feature = depth_map[y, x]

        # Use 1/depth feature
        inverse_depth = 1.0 / (depth_feature + 1e-6)
        input_data = np.array([[inverse_depth]], dtype=np.float32)
        
        try:
            predicted_distance = self.distance_model.predict(input_data)[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            self.status_label.setText(f"Prediction error: {e}")
            return

        # Create and store the new annotation
        new_annotation = {
            'id': self.next_annotation_id,
            'image_path': self.current_image_path,
            'coords': coords,
            'predicted_distance': float(predicted_distance),
            'graphics_items': []
        }
        self.next_annotation_id += 1

        if self.current_image_path not in self.image_data:
            self.image_data[self.current_image_path] = []
        self.image_data[self.current_image_path].append(new_annotation)

        # Update UI
        self.draw_annotation_on_scene(new_annotation)
        self.update_results_table()
        self.update_thumbnail_status()
        self.status_label.setText(f"Distance for ID {new_annotation['id']} at ({x}, {y}): {predicted_distance:.2f}m")

    # --- UI Update and Navigation Methods ---

    def handle_mouse_move(self, event):
        if self.current_image_cv is None: return
        scene_pos = self.graphics_view.mapToScene(event.pos())
        self.status_label.setText(f"Coordinates: ({int(scene_pos.x())}, {int(scene_pos.y())})")

    def load_previous_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image(self.image_files[self.current_image_index])

    def load_next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.display_image(self.image_files[self.current_image_index])
    
    def update_navigation_buttons_state(self):
        can_navigate = len(self.image_files) > 1
        self.prev_btn.setEnabled(can_navigate and self.current_image_index > 0)
        self.next_btn.setEnabled(can_navigate and self.current_image_index < len(self.image_files) - 1)
        self.export_csv_btn.setEnabled(any(self.image_data.values()))
        self.show_dpt_btn.setEnabled(self.current_image_cv is not None and self.dpt_model is not None)
        
        # Enable Auto-Calc only if we have images, model, and DPT
        can_auto_calc = (
            bool(self.image_files) and 
            self.distance_model is not None and 
            self.dpt_model is not None
        )
        self.auto_calc_btn.setEnabled(can_auto_calc)

    def update_results_table(self):
        self.results_table.setRowCount(0)
        
        # Collect all annotations from all images
        all_annotations = []
        for annotations in self.image_data.values():
            all_annotations.extend(annotations)
            
        # Sort by ID for consistent display
        all_annotations.sort(key=lambda x: x['id'])
        
        self.results_table.setRowCount(len(all_annotations))
        
        for i, ann in enumerate(all_annotations):
            # Create items
            id_item = QTableWidgetItem(str(ann['id']))
            name_item = QTableWidgetItem(os.path.basename(ann['image_path']))
            dist_item = QTableWidgetItem(f"{ann['predicted_distance']:.2f}")
            coord_item = QTableWidgetItem(f"({ann['coords'][0]}, {ann['coords'][1]})")
            
            # Highlight if it belongs to the current image
            if ann['image_path'] == self.current_image_path:
                highlight_color = QColor(80, 200, 120) # Light Green
                id_item.setBackground(highlight_color)
                name_item.setBackground(highlight_color)
                dist_item.setBackground(highlight_color)
                coord_item.setBackground(highlight_color)
            
            self.results_table.setItem(i, 0, id_item)
            self.results_table.setItem(i, 1, name_item)
            self.results_table.setItem(i, 2, dist_item)
            self.results_table.setItem(i, 3, coord_item)
        
        self.export_csv_btn.setEnabled(bool(all_annotations))
        self.update_delete_button_state()

    def update_delete_button_state(self):
        self.delete_row_btn.setEnabled(bool(self.results_table.selectedItems()))

    def delete_selected_row(self):
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows: return

        row_to_delete = selected_rows[0].row()
        ann_id_to_delete = int(self.results_table.item(row_to_delete, 0).text())

        # Find and remove annotation from data and scene
        found = False
        for image_path, annotations in self.image_data.items():
            ann_to_remove = next((ann for ann in annotations if ann['id'] == ann_id_to_delete), None)
            if ann_to_remove:
                # Remove graphics items if it's the current image
                if image_path == self.current_image_path:
                    for item in ann_to_remove.get('graphics_items', []):
                        if item in self.graphics_scene.items():
                            self.graphics_scene.removeItem(item)
                
                annotations.remove(ann_to_remove)
                self.status_label.setText(f"Deleted annotation ID {ann_id_to_delete}.")
                found = True
                break
        
        if found:
            self.update_results_table()
            self.update_thumbnail_status()

    def export_to_csv(self):
        """Exports all calculated distances to a single CSV file."""
        all_data = [ann for annotations in self.image_data.values() for ann in annotations]
        if not all_data:
            QMessageBox.warning(self, "No Data", "There are no calculated distances to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "wildlife_distances.csv", "CSV Files (*.csv)")
        if not path: return

        try:
            with open(path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['annotation_id', 'image_name', 'distance_m', 'x_coord', 'y_coord'])
                for ann in all_data:
                    writer.writerow([
                        ann['id'],
                        os.path.basename(ann['image_path']),
                        f"{ann['predicted_distance']:.4f}",
                        ann['coords'][0],
                        ann['coords'][1]
                    ])
            QMessageBox.information(self, "Export Successful", f"Exported {len(all_data)} records to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data to CSV:\n{str(e)}")

    def toggle_dpt_view(self):
        """Toggles the view between the real image and its DPT depth map."""
        if self.current_depth_map is None:
            # Generate it first
            self.status_label.setText("Generating depth map for visualization...")
            self.progress_dialog = QProgressDialog("Generating Depth Map...", "Cancel", 0, 0, self)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.show()

            self.dpt_worker = DPTWorker(self.dpt_model, self.dpt_processor, self.device, self.current_image_cv.copy())
            self._pending_coords_for_prediction = None # Signal that this is just for viewing
            
            self.dpt_worker.finished.connect(self.on_depth_map_generated)
            self.dpt_worker.error.connect(self.on_dpt_error)
            self.dpt_worker.start()
            return

        self.is_dpt_view_active = not self.is_dpt_view_active
        self.graphics_scene.clear()

        if self.is_dpt_view_active:
            # Normalize for visualization
            normalized_depth = cv2.normalize(self.current_depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_INFERNO)
            
            h, w, ch = depth_colormap.shape
            q_image = QImage(depth_colormap.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
            self.current_pixmap_item = self.graphics_scene.addPixmap(QPixmap.fromImage(q_image))
            
            self.show_dpt_btn.setText("Show Real Image")
            self.status_label.setText("Displaying DPT depth map.")
        else:
            # Show original image
            h, w, ch = self.current_image_cv.shape
            q_image = QImage(self.current_image_cv.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
            self.current_pixmap_item = self.graphics_scene.addPixmap(QPixmap.fromImage(q_image))
            self.draw_all_annotations_for_current_image() # Redraw annotations
            
            self.show_dpt_btn.setText("Show Depth Map")
            self.status_label.setText("Displaying real image.")

        self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)


    def auto_calculate_all(self):
        """Starts the auto-detection and calculation process."""
        if not self.image_files: return
        
        self.progress_dialog = QProgressDialog("Initializing Auto-Calculation...", "Cancel", 0, len(self.image_files), self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()

        self.auto_worker = AutoDetectWorker(
            self.image_files, self.dpt_model, self.dpt_processor, 
            self.distance_model, self.device
        )
        self.auto_worker.progress_update.connect(self.on_auto_calc_progress)
        self.auto_worker.detection_found.connect(self.on_auto_detection_found)
        self.auto_worker.finished.connect(self.on_auto_calc_finished)
        self.auto_worker.error.connect(self.on_auto_calc_error)
        self.auto_worker.start()

    def on_auto_calc_progress(self, current, total, message):
        self.progress_dialog.setLabelText(message)
        self.progress_dialog.setValue(current)
        if self.progress_dialog.wasCanceled():
            self.auto_worker.stop()

    def on_auto_detection_found(self, image_path, coords, distance, confidence):
        # Create annotation data structure
        new_annotation = {
            'id': self.next_annotation_id,
            'image_path': image_path,
            'coords': coords,
            'predicted_distance': distance,
            'graphics_items': []
        }
        self.next_annotation_id += 1

        if image_path not in self.image_data:
            self.image_data[image_path] = []
        self.image_data[image_path].append(new_annotation)

        # If this is the current image, draw it immediately
        if image_path == self.current_image_path:
            self.draw_annotation_on_scene(new_annotation)
            self.update_results_table()

    def on_auto_calc_finished(self):
        self.progress_dialog.close()
        self.update_thumbnail_status()
        self.update_results_table()
        QMessageBox.information(self, "Auto-Calculation Complete", "Finished processing all images.")

    def on_auto_calc_error(self, message):
        self.progress_dialog.close()
        QMessageBox.critical(self, "Auto-Calculation Error", message)

if __name__ == "__main__":
    from styles import apply_theme
    app = QApplication(sys.argv)
    apply_theme(app)
    window = DistanceCalculator()
    window.show()
    sys.exit(app.exec_())