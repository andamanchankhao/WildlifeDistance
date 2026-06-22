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
    QSplitter, QFrame, QStackedWidget
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QIcon, QPen
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread

from styles import UploadPlaceholder

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

    def __init__(self, image_files, dpt_model, dpt_processor, distance_model, device, yolo_model=None):
        super().__init__()
        self.image_files = image_files
        self.dpt_model = dpt_model
        self.dpt_processor = dpt_processor
        self.distance_model = distance_model
        self.device = device
        # Bug H fix: accept a pre-loaded YOLO model instead of downloading each run.
        self.yolo_model = yolo_model
        self.is_running = True

    def run(self):
        try:
            self.progress_update.emit(0, len(self.image_files), "Loading Object Detection Model...")
            # Bug H fix: only load YOLO from hub if one was not already provided.
            if self.yolo_model is None:
                try:
                    self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
                    self.yolo_model.to(self.device)
                    self.yolo_model.eval()
                except Exception as e:
                    self.error.emit(f"Failed to load YOLOv5 model: {e}")
                    return
            else:
                self.progress_update.emit(0, len(self.image_files), "Using cached YOLO model...")

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


class DistanceCalculator(QWidget):
    def __init__(self, parent=None, shared_dpt_processor=None, shared_dpt_model=None, shared_device=None):
        super().__init__(parent)

        # Model and data state
        self.distance_model = None
        self.image_files = []
        self.current_image_index = 0
        self.image_data = {}  # Stores annotations: {image_path: [annotation_dict, ...]}
        self.next_annotation_id = 0
        self.camera_id = "default"  # Default camera ID

        # Image and graphics state
        self.current_image_path = None
        self.current_image_cv = None
        # Opt-3: Cache stores both the depth map and the path it was generated for,
        # so we never serve a stale cache hit when the image changes.
        self.current_depth_map = None           # the cached numpy depth map
        self.current_depth_map_path = None      # which image path the cache belongs to
        self._pending_coords_for_prediction = None
        self.current_pixmap_item = None
        self.is_dpt_view_active = False

        # Opt-1: Use shared model if provided; otherwise load our own.
        if shared_dpt_processor is not None and shared_dpt_model is not None:
            self.dpt_processor = shared_dpt_processor
            self.dpt_model = shared_dpt_model
            self.device = shared_device or "cpu"
        else:
            self.dpt_processor = None
            self.dpt_model = None
            self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"

        # Bug H fix: cache the YOLO model so it is loaded only once per session.
        self.yolo_model = None

        # UI element references
        self.thumbnail_items = {}

        # Initialize UI and models
        self.init_ui()
        # Only load DPT model internally if we did not receive a shared one.
        if self.dpt_model is None:
            self.load_dpt_inference_model()
        else:
            self.status_label.setText(f"DPT model ready (shared, device={self.device}).")
        self.update_navigation_buttons_state()


    # ── Helper ────────────────────────────────────────────────────────────
    def _create_step_card(self, step_num: str, title: str, button: QPushButton, info_label: QLabel) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            QLabel { border: none; background-color: transparent; }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        step_lbl = QLabel(step_num)
        step_lbl.setStyleSheet("font-weight: bold; color: #c82828; font-size: 13px;")
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-weight: bold; color: #333333; font-size: 13px;")
        header_row.addWidget(step_lbl)
        header_row.addWidget(title_lbl)
        header_row.addStretch()
        layout.addLayout(header_row)

        layout.addWidget(button)

        info_label.setStyleSheet("color: #666666; font-size: 12px;")
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(info_label)

        return card

    def init_ui(self):
        """Initializes the main UI layout and widgets."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ── Header row (title + Clear button) ────────────────────────────
        header_widget = QWidget()
        header_outer = QHBoxLayout(header_widget)
        header_outer.setContentsMargins(0, 0, 0, 5)
        header_outer.setSpacing(0)

        title_block = QWidget()
        title_layout = QVBoxLayout(title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)
        title_lbl = QLabel("Distance Calculator")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #1e1e1e;")
        desc_lbl = QLabel("Click anywhere on an image to measure the distance to the camera using the loaded calibration model.")
        desc_lbl.setStyleSheet("font-size: 13px; color: #666666;")
        title_layout.addWidget(title_lbl)
        title_layout.addWidget(desc_lbl)
        header_outer.addWidget(title_block, 1)

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
            QPushButton:hover { background-color: #d0d0d0; color: #333333; border: 1px solid #bbbbbb; }
            QPushButton:pressed { background-color: #bbbbbb; }
        """)
        header_outer.addWidget(self.clear_btn, 0, Qt.AlignTop | Qt.AlignRight)
        main_layout.addWidget(header_widget)

        # ── Step Cards ────────────────────────────────────────────────────
        steps_widget = QWidget()
        steps_layout = QHBoxLayout(steps_widget)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(15)

        # Card 1: Load Images
        self.load_dir_btn = QPushButton("Load Image Directory")
        self.load_dir_btn.clicked.connect(self.load_image_directory)
        self.load_single_img_btn = QPushButton("Load Single Image")
        self.load_single_img_btn.clicked.connect(self.load_single_image)
        self.dir_info_label = QLabel("No images loaded")

        card1_frame = QFrame()
        card1_frame.setStyleSheet("""
            QFrame { background-color: white; border: 1px solid #e0e0e0; border-radius: 8px; }
            QLabel { border: none; background-color: transparent; }
        """)
        c1_layout = QVBoxLayout(card1_frame)
        c1_layout.setContentsMargins(15, 12, 15, 12)
        c1_layout.setSpacing(8)
        c1_header = QHBoxLayout()
        c1_step = QLabel("STEP 1")
        c1_step.setStyleSheet("font-weight: bold; color: #c82828; font-size: 13px;")
        c1_title = QLabel("IMAGES")
        c1_title.setStyleSheet("font-weight: bold; color: #333333; font-size: 13px;")
        c1_header.addWidget(c1_step)
        c1_header.addWidget(c1_title)
        c1_header.addStretch()
        c1_layout.addLayout(c1_header)
        c1_layout.addWidget(self.load_dir_btn)
        c1_layout.addWidget(self.load_single_img_btn)
        self.dir_info_label.setStyleSheet("color: #666666; font-size: 12px;")
        self.dir_info_label.setWordWrap(True)
        c1_layout.addWidget(self.dir_info_label)
        steps_layout.addWidget(card1_frame, 1)

        # Card 2: Load Model
        self.load_model_btn = QPushButton("Load Distance Model")
        self.load_model_btn.clicked.connect(self.load_distance_model)
        self.model_status_label = QLabel("Model: <font color='red'>Not Loaded</font>")
        card2 = self._create_step_card("STEP 2", "MODEL", self.load_model_btn, self.model_status_label)
        steps_layout.addWidget(card2, 1)

        # Card 3: Auto-Calculate
        self.auto_calc_btn = QPushButton("Auto-Calculate All Images")
        self.auto_calc_btn.setEnabled(False)
        self.auto_calc_btn.setStyleSheet("""
            QPushButton {
                background-color: #c82828; color: white; font-weight: bold;
                border: 1px solid #c82828;
            }
            QPushButton:hover { background-color: #d83838; }
            QPushButton:disabled { background-color: #f0f0f0; color: #aaaaaa; border: 1px solid #dddddd; }
        """)
        self.auto_calc_btn.clicked.connect(self.auto_calculate_all)
        auto_info = QLabel("Auto-detect animals & measure distances")
        card3 = self._create_step_card("STEP 3", "AUTO-DETECT", self.auto_calc_btn, auto_info)
        steps_layout.addWidget(card3, 1)

        main_layout.addWidget(steps_widget)

        # ── Main content splitter (left thumbnails | center image | right table) ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #e0e0e0; width: 1px; }")

        # Left: thumbnail strip
        left_panel = QFrame()
        left_panel.setObjectName("GlassPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)
        thumb_header = QLabel("Images")
        thumb_header.setStyleSheet("font-size: 11px; font-weight: bold; color: #888888; letter-spacing: 1px;")
        left_layout.addWidget(thumb_header)
        self.thumbnail_list_widget = ResponsiveListWidget()
        self.thumbnail_list_widget.setIconSize(QSize(100, 80))
        self.thumbnail_list_widget.itemClicked.connect(self.thumbnail_clicked)
        left_layout.addWidget(self.thumbnail_list_widget, 1)
        splitter.addWidget(left_panel)

        # Center: image viewer
        center_panel = QFrame()
        center_panel.setObjectName("GlassPanel")
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(6, 6, 6, 6)
        center_layout.setSpacing(6)

        self.current_image_name_label = QLabel("No Image Loaded")
        self.current_image_name_label.setFont(QFont("Arial", 13, QFont.Bold))
        self.current_image_name_label.setAlignment(Qt.AlignCenter)
        self.current_image_name_label.setStyleSheet("color: #555555; padding: 2px 0px;")
        center_layout.addWidget(self.current_image_name_label)

        # Stacked widget to switch between upload placeholder and graphics view
        self.center_stack = QStackedWidget()
        
        # Page 0: Upload Placeholder
        self.upload_placeholder = UploadPlaceholder("Drop Image File or Folder Here\n- or -\nClick to Browse")
        self.upload_placeholder.clicked.connect(self.load_image_directory)
        self.upload_placeholder.files_dropped.connect(self._on_files_dropped)
        self.center_stack.addWidget(self.upload_placeholder)
        
        # Page 1: Graphics View
        self.graphics_view = CustomGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.mouse_pressed.connect(self.handle_mouse_press)
        self.graphics_view.mouse_moved.connect(self.handle_mouse_move)
        self.center_stack.addWidget(self.graphics_view)
        
        center_layout.addWidget(self.center_stack, 1)

        # Bottom toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet("background-color: #f9f9f9; border-top: 1px solid #e8e8e8;")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(8)

        nav_btn_style = """
            QPushButton {
                background-color: #f5f5f5; border: 1px solid #ddd;
                border-radius: 4px; color: #555; padding: 4px 12px;
            }
            QPushButton:hover { background-color: #e8e8e8; border: 1px solid #bbb; }
            QPushButton:disabled { color: #cccccc; }
        """

        self.prev_btn = QPushButton("‹ Prev")
        self.prev_btn.setStyleSheet(nav_btn_style)
        self.prev_btn.clicked.connect(self.load_previous_image)
        toolbar_layout.addWidget(self.prev_btn)

        hint_lbl = QLabel("Click image to measure distance")
        hint_lbl.setStyleSheet("font-size: 11px; color: #999999;")
        hint_lbl.setAlignment(Qt.AlignCenter)
        toolbar_layout.addWidget(hint_lbl, 1)

        self.show_dpt_btn = QPushButton("Toggle Depth Map")
        self.show_dpt_btn.setStyleSheet(nav_btn_style)
        self.show_dpt_btn.clicked.connect(self.toggle_dpt_view)
        toolbar_layout.addWidget(self.show_dpt_btn)

        toolbar_layout.addWidget(QLabel(""))  # spacer
        self.next_btn = QPushButton("Next ›")
        self.next_btn.setStyleSheet(nav_btn_style)
        self.next_btn.clicked.connect(self.load_next_image)
        toolbar_layout.addWidget(self.next_btn)

        center_layout.addWidget(toolbar)
        splitter.addWidget(center_panel)

        # Right: results table
        right_panel = QFrame()
        right_panel.setObjectName("GlassPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 12, 10, 10)
        right_layout.setSpacing(8)

        results_header = QLabel("Results")
        results_header.setStyleSheet("font-size: 11px; font-weight: bold; color: #888888; letter-spacing: 1px;")
        right_layout.addWidget(results_header)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(len(TABLE_HEADERS))
        self.results_table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.results_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e5e5e5;
                border-radius: 6px;
                gridline-color: #f0f0f0;
            }
            QTableWidget::item:alternate { background-color: #fafafa; }
            QTableWidget::item:selected {
                background-color: rgb(80, 200, 120);
                color: black;
            }
        """)
        right_layout.addWidget(self.results_table, 1)

        # Table action buttons
        btn_divider = QFrame()
        btn_divider.setFrameShape(QFrame.HLine)
        btn_divider.setStyleSheet("color: #e5e5e5;")
        right_layout.addWidget(btn_divider)

        table_btn_layout = QHBoxLayout()
        table_btn_layout.setSpacing(8)

        self.delete_row_btn = QPushButton("Delete Selected")
        self.delete_row_btn.clicked.connect(self.delete_selected_row)
        self.delete_row_btn.setStyleSheet("""
            QPushButton {
                background-color: white; border: 1px solid #e0e0e0;
                border-radius: 4px; padding: 5px 10px; color: #c82828;
            }
            QPushButton:hover { background-color: #fff0f0; border: 1px solid #c82828; }
        """)
        table_btn_layout.addWidget(self.delete_row_btn)

        self.export_csv_btn = QPushButton("Export to CSV")
        self.export_csv_btn.clicked.connect(self.export_to_csv)
        self.export_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #c82828; color: white; border: 1px solid #c82828;
                border-radius: 4px; padding: 5px 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #d83838; }
            QPushButton:disabled { background-color: #f0f0f0; color: #aaaaaa; border: 1px solid #dddddd; }
        """)
        table_btn_layout.addWidget(self.export_csv_btn)

        right_layout.addLayout(table_btn_layout)
        splitter.addWidget(right_panel)

        # Stretch: narrow left | wide center | narrow right
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)

        main_layout.addWidget(splitter, 1)

        # ── Status Bar ─────────────────────────────────────────────────────
        self.status_label = QLabel("Ready. Load images and a distance model to begin.")
        self.status_label.setObjectName("StatusLabel")
        main_layout.addWidget(self.status_label, 0)

    def clear_all(self):
        """Reset all state and clear the UI back to its initial state."""
        self.image_files = []
        self.current_image_index = 0
        self.image_data.clear()
        self.next_annotation_id = 0
        self.current_image_path = None
        self.current_image_cv = None
        self.current_depth_map = None
        self.current_depth_map_path = None
        self._pending_coords_for_prediction = None
        self.current_pixmap_item = None
        self.is_dpt_view_active = False
        self.distance_model = None

        # Reset thumbnail strip
        self.thumbnail_list_widget.clear()
        self.thumbnail_items.clear()

        # Reset image viewer
        self.graphics_scene.clear()
        self.current_image_name_label.setText("No Image Loaded")

        # Reset results table
        self.results_table.setRowCount(0)

        # Reset labels
        self.dir_info_label.setText("No images loaded")
        self.model_status_label.setText("Model: <font color='red'>Not Loaded</font>")

        # Reset buttons
        self.auto_calc_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)

        self.status_label.setText("Cleared. Please load images and a distance model to begin.")
        self.center_stack.setCurrentIndex(0)
        self.update_navigation_buttons_state()



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
        if directory:
            self.load_directory_from_path(directory)

    def load_directory_from_path(self, directory: str):
        if not directory or not os.path.isdir(directory): return

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
            self.center_stack.setCurrentIndex(1) # Show graphics view
            self.display_image(self.image_files[self.current_image_index])
            self.dir_info_label.setText(f"{os.path.basename(directory)} ({len(self.image_files)} images)")
            self.status_label.setText(f"Loaded {len(self.image_files)} images.")
            
            # Auto-detect model
            self.check_and_autoload_model(directory)
        else:
            self.dir_info_label.setText("No images found")
            self.status_label.setText("No supported image files found in the selected directory.")
            self.center_stack.setCurrentIndex(0) # Show placeholder
        
        self.update_navigation_buttons_state()

    def load_single_image(self):
        """Loads a single image file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.load_image_from_path(file_path)

    def load_image_from_path(self, file_path: str):
        if not file_path or not os.path.isfile(file_path): return

        self.image_files = [file_path]
        
        self.thumbnail_list_widget.clear()
        self.thumbnail_items.clear()
        self.image_data.clear()
        
        self.add_thumbnail(file_path)
        self.current_image_index = 0
        self.center_stack.setCurrentIndex(1) # Show graphics view
        self.display_image(self.image_files[0])
        self.dir_info_label.setText(f"Single image: {os.path.basename(file_path)}")
        self.status_label.setText(f"Loaded single image: {os.path.basename(file_path)}")
        
        # Try to auto-load model from this image's directory
        self.check_and_autoload_model(os.path.dirname(file_path))
        
        self.update_navigation_buttons_state()

    def _on_files_dropped(self, paths):
        if not paths:
            return
        target_path = paths[0]
        if os.path.isdir(target_path):
            self.load_directory_from_path(target_path)
        elif os.path.isfile(target_path) and target_path.lower().endswith(VALID_IMAGE_EXTENSIONS):
            self.load_image_from_path(target_path)

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

        # Opt-3: Only invalidate depth cache if we are loading a different image.
        if self.current_depth_map_path != image_path:
            self.current_depth_map = None
            self.current_depth_map_path = None

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
        # Opt-3: store both the map and the path it belongs to.
        self.current_depth_map = depth_map
        self.current_depth_map_path = self.current_image_path
        
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
        self.results_table.blockSignals(True)
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
            
            # Store reference to the annotation object in the first item
            id_item.setData(Qt.UserRole, ann)
            
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
            
        # Highlight/select rows for current image and scroll to the first one
        self.results_table.clearSelection()
        scrolled = False
        for r_idx in range(self.results_table.rowCount()):
            item = self.results_table.item(r_idx, 0)
            if item:
                ann = item.data(Qt.UserRole)
                if ann and ann.get('image_path') == self.current_image_path:
                    for col_idx in range(self.results_table.columnCount()):
                        row_item = self.results_table.item(r_idx, col_idx)
                        if row_item:
                            row_item.setSelected(True)
                    if not scrolled:
                        self.results_table.scrollToItem(item)
                        scrolled = True
        
        self.export_csv_btn.setEnabled(bool(all_annotations))
        self.update_delete_button_state()
        self.results_table.blockSignals(False)

    def _on_table_selection_changed(self):
        """Slot to handle row selections in the results table."""
        selected_items = self.results_table.selectedItems()
        self.update_delete_button_state()
        if not selected_items:
            return
        
        # Get the annotation data from the first column of the selected rows
        for item in selected_items:
            row = item.row()
            first_col_item = self.results_table.item(row, 0)
            if first_col_item:
                ann = first_col_item.data(Qt.UserRole)
                if ann:
                    img_path = ann.get('image_path')
                    if img_path and img_path != self.current_image_path:
                        try:
                            index = self.image_files.index(img_path)
                            if index != self.current_image_index:
                                self.current_image_index = index
                                self.display_image(img_path)
                        except ValueError:
                            pass
                    break

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
            self.distance_model, self.device,
            yolo_model=self.yolo_model  # Bug H fix: pass cached model to avoid re-download.
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
        # Bug H fix: persist the YOLO model on the instance for the next run.
        if self.auto_worker.yolo_model is not None:
            self.yolo_model = self.auto_worker.yolo_model
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