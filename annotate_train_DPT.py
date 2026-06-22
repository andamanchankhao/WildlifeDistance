#################################################################
### Dev: Andaman Chankhao andmanchankhao@gmail.com            ####
### Rev: Gemini (Code Refinement and Feature Enhancement)     ####
#################################################################

import sys
import os
import json
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGraphicsView, QGraphicsScene, QInputDialog,
    QFileDialog, QMessageBox, QDialog, QSizePolicy, QListWidget,
    QListWidgetItem, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QStackedWidget
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont, QPolygonF, QBrush
from PyQt5.QtCore import Qt, QPointF, QRectF, QThread, pyqtSignal, QSize

from styles import UploadPlaceholder

# --- Dependency Imports with Graceful Fallbacks ---

# Matplotlib removed (moved to training tool)

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
ANNOTATION_FONT = QFont("Arial", 32, QFont.Bold) # Increased font size
TEMP_RECT_PEN = QPen(QColor("red"), 2, Qt.DashLine)
TEMP_LINE_PEN = QPen(QColor("red"), 1, Qt.DashLine)

# --- Helper Functions ---
def convert_cv_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    """Converts an OpenCV image (BGR) to a QPixmap."""
    if cv_img is None:
        return QPixmap()
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(q_img)

# generate_dpt_depth_map is still needed for "Show Depth" feature
def generate_dpt_depth_map(cv_img: np.ndarray, processor, model, device: str) -> Optional[np.ndarray]:
    """Generates a depth map from an OpenCV image using a DPT model."""
    # Bug F fix: check numpy array and other deps separately to avoid ValueError
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
            # Re-use the helper function logic but inside the thread
            depth_map = generate_dpt_depth_map(self.image_cv, self.dpt_processor, self.dpt_model, self.device)
            if depth_map is not None:
                self.finished.emit(depth_map)
            else:
                self.error.emit("Failed to generate depth map (returned None).")
        except Exception as e:
            self.error.emit(f"Failed to generate depth map: {str(e)}")

# --- PyQt GUI Classes ---


class CustomGraphicsView(QGraphicsView):
    """Subclassed QGraphicsView to emit signals for mouse events."""
    mouse_pressed = pyqtSignal(object)
    mouse_moved = pyqtSignal(object)
    mouse_released = pyqtSignal(object)
    mouse_double_clicked = pyqtSignal(object)

    def mousePressEvent(self, event):
        self.mouse_pressed.emit(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.mouse_moved.emit(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.mouse_released.emit(event)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.mouse_double_clicked.emit(event)
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        if self.scene():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)

class ResponsiveListWidget(QListWidget):
    """A QListWidget that automatically resizes its items to fit the width with the correct aspect ratio."""
    def resizeEvent(self, event):
        width = event.size().width()
        # Adjust for scrollbar and borders (approx 24px)
        item_width = max(50, width - 24)
        # 4:3 aspect ratio height plus 15px bottom spacing gap
        item_height = int(item_width * 0.75) + 15
        
        for i in range(self.count()):
            item = self.item(i)
            item.setSizeHint(QSize(item_width, item_height))
            
        super().resizeEvent(event)

class ThumbnailItemWidget(QWidget):
    """Custom widget for displaying an image thumbnail in the QListWidget."""
    def __init__(self, pixmap: QPixmap, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.original_pixmap = pixmap
        self.is_annotated = False

        layout = QVBoxLayout(self)
        # 5px margin on left/right/top, 15px margin on bottom for spacing
        layout.setContentsMargins(5, 5, 5, 15)
        
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.thumbnail_label)
        
        self.setAutoFillBackground(True)
        self.set_selected(False) # Set initial background

    def resizeEvent(self, event):
        # Subtract the margins (left+right=10, top+bottom=20)
        w = self.width() - 10
        h = self.height() - 20
        if w > 10 and h > 10:
            scaled_pixmap = self.original_pixmap.scaled(
                w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.thumbnail_label.setPixmap(scaled_pixmap)
        super().resizeEvent(event)

    def set_annotated_status(self, annotated: bool):
        self.is_annotated = annotated

    def set_selected(self, selected: bool):
        """Updates the selection status and background color."""
        palette = self.palette()
        color = QColor("lightblue") if selected else QColor("transparent")
        palette.setColor(self.backgroundRole(), color)
        self.setPalette(palette)


class AnnotationTool(QWidget):
    """Main application widget for the annotation tool."""
    def __init__(self, parent=None, shared_dpt_processor=None, shared_dpt_model=None, shared_device=None):
        super().__init__(parent)

        # State variables
        self.image_directory: Optional[str] = None
        self.image_files: List[str] = []
        self.current_image_path: Optional[str] = None
        self.current_image_cv: Optional[np.ndarray] = None
        self.annotations_by_image: Dict[str, List] = {}
        self.is_dpt_view_active = False
        self.json_output_directory: Optional[str] = None

        # Drawing state
        self.drawing_polygon = False
        self.current_polygon_points = []
        self.current_temp_items = []

        # DPT Model components
        # Opt-1: use a shared model if provided; otherwise load our own.
        if shared_dpt_processor is not None and shared_dpt_model is not None:
            self.dpt_processor = shared_dpt_processor
            self.dpt_model = shared_dpt_model
            self.device = shared_device or "cpu"
        else:
            self.dpt_processor = None
            self.dpt_model = None
            self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"

        # Bug E fix: TrainingThread and PlotWindow were moved to training_DPT.py.
        # Bug D fix: image_item_map was declared twice, keep only one.
        self.image_item_map: Dict[str, ThumbnailItemWidget] = {}
        self.image_list_item_map: Dict[str, QListWidgetItem] = {}

        # DPT Worker
        self.dpt_worker = None
        self.progress_dialog = None

        self.init_ui()
        # Only load DPT model internally if we did not receive a shared one.
        if self.dpt_model is None:
            self.load_dpt_inference_model()
        else:
            # Shared model is already loaded; just update the status label after UI init.
            self.status_label.setText(f"DPT model ready (shared, device={self.device}).")
    
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
    # ── Helper Methods ───────────────────────────────────────────────────
    def create_step_card(self, step_num: str, title: str, button: QPushButton, info_label: QLabel) -> QFrame:
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

        if button is not None:
            layout.addWidget(button)

        info_label.setStyleSheet("color: #666666; font-size: 12px;")
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(info_label)

        return card

    def clear_all(self):
        """Reset all state and clear the UI back to its initial state."""
        self.save_annotations_for_current_image()

        self.image_directory = None
        self.image_files = []
        self.annotations_by_image.clear()
        self.json_output_directory = None
        self.current_image_path = None
        self.current_image_cv = None
        self.current_polygon_points = []
        self.current_temp_items = []
        self.drawing_polygon = False

        # Reset list and map
        self.thumbnail_list_widget.clear()
        self.image_item_map.clear()
        self.image_list_item_map.clear()

        # Reset graphics view
        self.graphics_scene.clear()
        self.current_image_name_label.setText("No Image Loaded")

        # Reset table
        self.annotation_table.setRowCount(0)

        # Reset labels
        self.dir_label.setText("No directory selected")
        self.output_dir_label.setText("Output: Default (Image Dir)")
        self.show_dpt_btn.setChecked(False)
        self.show_dpt_btn.setText("Show DPT Depth")

        self.status_label.setText("Cleared. Choose an image directory to begin.")
        self.center_stack.setCurrentIndex(0)
        self.update_navigation_buttons_state()

    def update_navigation_buttons_state(self):
        if not self.image_files or self.current_image_path is None:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return
        try:
            current_index = self.image_files.index(self.current_image_path)
            self.prev_btn.setEnabled(current_index > 0)
            self.next_btn.setEnabled(current_index < len(self.image_files) - 1)
        except ValueError:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)

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
        title_layout = QVBoxLayout(title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)
        title_lbl = QLabel("Image Annotation Tool")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #1e1e1e;")
        desc_lbl = QLabel("Draw polygon annotations and measure distances to calibrate the distance estimation model.")
        desc_lbl.setStyleSheet("font-size: 13px; color: #666666;")
        title_layout.addWidget(title_lbl)
        title_layout.addWidget(desc_lbl)
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
            QPushButton:hover { background-color: #d0d0d0; color: #333333; border: 1px solid #bbbbbb; }
            QPushButton:pressed { background-color: #bbbbbb; }
        """)
        header_outer.addWidget(self.clear_btn, 0, Qt.AlignTop | Qt.AlignRight)
        main_layout.addWidget(header_widget)

        # --- Top Steps Config Panel ---
        steps_widget = QWidget()
        steps_layout = QHBoxLayout(steps_widget)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(15)

        # Step 1 Card: Load Directory
        self.load_dir_btn = QPushButton("Choose Image Directory")
        self.load_dir_btn.clicked.connect(self.load_directory)
        self.dir_label = QLabel("No directory selected")
        card1 = self.create_step_card("STEP 1", "DATASET", self.load_dir_btn, self.dir_label)
        steps_layout.addWidget(card1, 1)

        # Step 2 Card: Set Output Folder
        self.set_output_dir_btn = QPushButton("Set Output (.JSON)")
        self.set_output_dir_btn.clicked.connect(self.set_output_directory)
        self.output_dir_label = QLabel("Output: Default (Image Dir)")
        card2 = self.create_step_card("STEP 2", "SAVE LOCATION", self.set_output_dir_btn, self.output_dir_label)
        steps_layout.addWidget(card2, 1)

        # Step 3 Card: Actions & Instructions
        self.instruction_label = QLabel("• Left-click points to draw polygon\n• Right-click or double-click to finish\n• Enter distance in dialog")
        card3 = self.create_step_card("STEP 3", "DRAWING GUIDE", None, self.instruction_label)
        steps_layout.addWidget(card3, 1)

        main_layout.addWidget(steps_widget)

        # --- Main content splitter (left thumbnails | center image | right table) ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #e0e0e0; width: 1px; }")

        # Left Panel (Thumbnails)
        left_panel = QFrame()
        left_panel.setObjectName("GlassPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)
        thumb_header = QLabel("Images")
        thumb_header.setStyleSheet("font-size: 11px; font-weight: bold; color: #888888; letter-spacing: 1px;")
        left_layout.addWidget(thumb_header)
        
        self.thumbnail_list_widget = ResponsiveListWidget()
        self.thumbnail_list_widget.setViewMode(QListWidget.ListMode)
        self.thumbnail_list_widget.setIconSize(THUMBNAIL_SIZE)
        self.thumbnail_list_widget.setSpacing(5)
        self.thumbnail_list_widget.itemClicked.connect(self._on_thumbnail_clicked)
        left_layout.addWidget(self.thumbnail_list_widget, 1)
        splitter.addWidget(left_panel)

        # Center Panel (Image View & Navigation)
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
        self.upload_placeholder = UploadPlaceholder("Drop Image Folder Here\n- or -\nClick to Browse")
        self.upload_placeholder.clicked.connect(self.load_directory)
        self.upload_placeholder.files_dropped.connect(self._on_files_dropped)
        self.center_stack.addWidget(self.upload_placeholder)
        
        # Page 1: Graphics View
        self.graphics_view = CustomGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.mouse_pressed.connect(self.mouse_press_event)
        self.graphics_view.mouse_moved.connect(self.mouse_move_event)
        self.graphics_view.mouse_double_clicked.connect(self.mouse_double_click_event)
        self.center_stack.addWidget(self.graphics_view)
        
        center_layout.addWidget(self.center_stack, 1)
        
        # Navigation toolbar
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
        
        hint_lbl = QLabel("Draw polygon: click points; double-click / right-click to finish")
        hint_lbl.setStyleSheet("font-size: 11px; color: #999999;")
        hint_lbl.setAlignment(Qt.AlignCenter)
        toolbar_layout.addWidget(hint_lbl, 1)
        
        self.show_dpt_btn = QPushButton("Toggle Depth Map")
        self.show_dpt_btn.setStyleSheet(nav_btn_style)
        self.show_dpt_btn.setCheckable(True)
        self.show_dpt_btn.toggled.connect(self.toggle_dpt_view)
        toolbar_layout.addWidget(self.show_dpt_btn)
        
        self.next_btn = QPushButton("Next ›")
        self.next_btn.setStyleSheet(nav_btn_style)
        self.next_btn.clicked.connect(self.load_next_image)
        toolbar_layout.addWidget(self.next_btn)
        
        center_layout.addWidget(toolbar)
        splitter.addWidget(center_panel)

        # Right Panel (Annotations Table)
        right_panel = QFrame()
        right_panel.setObjectName("GlassPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 12, 10, 10)
        right_layout.setSpacing(8)
        
        results_header = QLabel("Annotations")
        results_header.setStyleSheet("font-size: 11px; font-weight: bold; color: #888888; letter-spacing: 1px;")
        right_layout.addWidget(results_header)
        
        self.annotation_table = QTableWidget()
        self.annotation_table.setColumnCount(3)
        self.annotation_table.setHorizontalHeaderLabels(["Image", "Coordinates", "Distance (m)"])
        self.annotation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.annotation_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.annotation_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.annotation_table.setSelectionMode(QTableWidget.SingleSelection)
        self.annotation_table.setAlternatingRowColors(True)
        self.annotation_table.verticalHeader().setVisible(False)
        self.annotation_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.annotation_table.setStyleSheet("""
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
        right_layout.addWidget(self.annotation_table, 1)
        
        # Table action buttons
        btn_divider = QFrame()
        btn_divider.setFrameShape(QFrame.HLine)
        btn_divider.setStyleSheet("color: #e5e5e5;")
        right_layout.addWidget(btn_divider)
        
        table_btn_layout = QHBoxLayout()
        table_btn_layout.setSpacing(8)
        
        self.delete_annotation_btn = QPushButton("Delete Selected")
        self.delete_annotation_btn.clicked.connect(self.delete_selected_annotation)
        self.delete_annotation_btn.setStyleSheet("""
            QPushButton {
                background-color: white; border: 1px solid #e0e0e0;
                border-radius: 4px; padding: 5px 10px; color: #c82828;
            }
            QPushButton:hover { background-color: #fff0f0; border: 1px solid #c82828; }
        """)
        table_btn_layout.addWidget(self.delete_annotation_btn)
        
        self.save_btn = QPushButton("Save Annotations")
        self.save_btn.clicked.connect(self.save_annotations_for_current_image)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #c82828; color: white; border: 1px solid #c82828;
                border-radius: 4px; padding: 5px 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #d83838; }
            QPushButton:disabled { background-color: #f0f0f0; color: #aaaaaa; border: 1px solid #dddddd; }
        """)
        table_btn_layout.addWidget(self.save_btn)
        
        right_layout.addLayout(table_btn_layout)
        splitter.addWidget(right_panel)

        # Stretch: narrow left | wide center | narrow right
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)

        main_layout.addWidget(splitter, 1)

        # --- Status Bar ---
        self.status_label = QLabel("Ready. Choose an image directory to begin.")
        self.status_label.setObjectName("StatusLabel")
        main_layout.addWidget(self.status_label, 0)

    def _on_thumbnail_clicked(self, item: QListWidgetItem):
        """Slot to handle clicks on thumbnail items."""
        thumbnail_widget = self.thumbnail_list_widget.itemWidget(item)
        if thumbnail_widget and thumbnail_widget.image_path != self.current_image_path:
            if self.show_dpt_btn.isChecked():
                self.show_dpt_btn.setChecked(False) # This will trigger toggle_dpt_view to switch back
            self.load_image(thumbnail_widget.image_path)

    def load_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if directory:
            self.load_directory_from_path(directory)

    def load_directory_from_path(self, directory: str):
        if not directory or not os.path.isdir(directory):
            return

        self.save_annotations_for_current_image() # Save work before switching directory
        self.image_directory = directory

        self.image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)])

        # Reset UI and state
        self.thumbnail_list_widget.clear()
        self.image_item_map.clear()
        self.image_list_item_map.clear()
        self.annotations_by_image.clear()
        self.annotation_table.setRowCount(0)
        self.graphics_scene.clear()

        if not self.image_files:
            self.dir_label.setText("No images found")
            self.status_label.setText("No supported image files found in the selected directory.")
            self.center_stack.setCurrentIndex(0) # Show placeholder
            return

        # Default JSON output to image directory if not set
        if not self.json_output_directory:
            self.json_output_directory = self.image_directory

        for img_path in self.image_files:
            # Load annotations
            self.annotations_by_image[img_path] = self.load_annotations_from_file(img_path)

            # Create and add thumbnail
            pixmap = QPixmap(img_path)
            item = QListWidgetItem()
            widget = ThumbnailItemWidget(pixmap, img_path)
            widget.set_annotated_status(bool(self.annotations_by_image[img_path]))
            
            # Initial size hint calculation based on current list width
            w = max(50, self.thumbnail_list_widget.width() - 24)
            h = int(w * 0.75) + 15
            item.setSizeHint(QSize(w, h))
            
            self.thumbnail_list_widget.addItem(item)
            self.thumbnail_list_widget.setItemWidget(item, widget)
            self.image_item_map[img_path] = widget
            self.image_list_item_map[img_path] = item
            
        self.dir_label.setText(f"{os.path.basename(directory)} ({len(self.image_files)} images)")
        self.output_dir_label.setText(f"Output: {os.path.basename(self.json_output_directory)}")
        self.status_label.setText(f"Loaded {len(self.image_files)} images.")
        
        self.center_stack.setCurrentIndex(1) # Show graphics view
        self.load_image(self.image_files[0])
        self.update_annotation_table()

    def _on_files_dropped(self, paths):
        if not paths:
            return
        target_path = paths[0]
        if os.path.isdir(target_path):
            self.load_directory_from_path(target_path)
        elif os.path.isfile(target_path) and target_path.lower().endswith(SUPPORTED_IMAGE_FORMATS):
            parent_dir = os.path.dirname(target_path)
            self.load_directory_from_path(parent_dir)
            self.select_image_by_path(target_path)

    def select_image_by_path(self, target_path):
        if target_path in self.image_list_item_map:
            item = self.image_list_item_map[target_path]
            self.thumbnail_list_widget.setCurrentItem(item)
            self._on_thumbnail_clicked(item)

    def set_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select JSON Output Directory")
        if directory:
            self.json_output_directory = directory
            self.output_dir_label.setText(f"Output: {os.path.basename(directory)}")
            self.status_label.setText(f"Output directory set to: {os.path.basename(directory)}")
            # Reload annotations from new directory? 
            if self.image_files:
                reply = QMessageBox.question(self, "Reload Annotations?", 
                                           "Do you want to reload annotations from this new directory? Unsaved changes will be lost.",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.annotations_by_image.clear()
                    for img_path in self.image_files:
                         self.annotations_by_image[img_path] = self.load_annotations_from_file(img_path)
                    
                    # Update UI for current image
                    if self.current_image_path:
                        self.load_image(self.current_image_path)

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
        if image_path in self.image_list_item_map:
            item = self.image_list_item_map[image_path]
            self.thumbnail_list_widget.blockSignals(True)
            self.thumbnail_list_widget.setCurrentItem(item)
            self.thumbnail_list_widget.scrollToItem(item)
            self.thumbnail_list_widget.blockSignals(False)
        self.status_label.setText(f"Displaying: {os.path.basename(image_path)}")
        self.current_image_name_label.setText(os.path.basename(image_path))
        self.update_navigation_buttons_state()

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
        if self.current_image_cv is None or self.current_image_path is None:
            self.status_label.setText("No image loaded to annotate.")
            return

        if not self.is_dpt_view_active:
            scene_pos = self.graphics_view.mapToScene(event.pos())
            
            # Check if click is within image bounds
            h, w, _ = self.current_image_cv.shape
            if not (0 <= scene_pos.x() < w and 0 <= scene_pos.y() < h):
                self.status_label.setText("Clicked outside image bounds.")
                return

            if event.button() == Qt.LeftButton:
                # Add point to polygon
                self.drawing_polygon = True
                self.current_polygon_points.append(scene_pos)
                
                # Draw a small circle for the point
                r = 3
                dot = self.graphics_scene.addEllipse(scene_pos.x()-r, scene_pos.y()-r, r*2, r*2, ANNOTATION_PEN, QBrush(Qt.red))
                self.current_temp_items.append(dot)
                
                # Draw line from previous point if exists
                if len(self.current_polygon_points) > 1:
                    line = self.graphics_scene.addLine(
                        self.current_polygon_points[-2].x(), self.current_polygon_points[-2].y(),
                        scene_pos.x(), scene_pos.y(),
                        ANNOTATION_PEN
                    )
                    self.current_temp_items.append(line)

            elif event.button() == Qt.RightButton and self.drawing_polygon:
                # Finish polygon
                if len(self.current_polygon_points) < 3:
                    self.status_label.setText("Polygon must have at least 3 points.")
                    return
                
                self.finish_polygon_annotation()

    def mouse_move_event(self, event):
        if self.current_image_cv is None: return
        scene_pos = self.graphics_view.mapToScene(event.pos())
        self.status_label.setText(f"Coordinates: ({int(scene_pos.x())}, {int(scene_pos.y())})")
        
        # Optional: Draw rubber band line from last point to cursor
        # (Implementation omitted for brevity/cleanliness, but can be added if requested)

    def mouse_double_click_event(self, event):
        """Handle double click to finish polygon."""
        if self.current_image_cv is None or self.current_image_path is None: return
        if self.drawing_polygon:
            print("DEBUG: Double click detected, finishing polygon.")
            self.finish_polygon_annotation()

    def finish_polygon_annotation(self):
        print("DEBUG: Finishing polygon annotation.")
        self.drawing_polygon = False
        
        # Clean up temp items
        for item in self.current_temp_items:
            self.graphics_scene.removeItem(item)
        self.current_temp_items = []
        
        distance, ok = QInputDialog.getDouble(self, "Enter Distance", "Distance (meters):", 0.0, 0.0, 1000.0, 2)
        if ok:
            # Convert QPointF to list of [x, y]
            coords = [[int(p.x()), int(p.y())] for p in self.current_polygon_points]
            
            annotation = {
                "type": "polygon",
                "coordinates": coords,
                "distance_meters": distance,
                "image_path": self.current_image_path
            }
            self.annotations_by_image[self.current_image_path].append(annotation)
            self.draw_annotation(annotation)
            self.update_annotation_table()
            self.image_item_map[self.current_image_path].set_annotated_status(True)
        
        self.current_polygon_points = []

    def draw_annotation(self, annotation: Dict):
        """Draws a single annotation on the scene and stores its Qt items."""
        coords = annotation["coordinates"]
        
        qt_items = []
        
        # Handle legacy bbox [x1, y1, x2, y2]
        if isinstance(coords[0], int):
             x1, y1, x2, y2 = coords
             rect = QRectF(x1, y1, x2-x1, y2-y1)
             rect_item = self.graphics_scene.addRect(rect, ANNOTATION_PEN)
             text_pos = rect.topLeft()
             qt_items.append(rect_item)
        else:
            # Polygon [[x,y], ...]
            polygon = QPolygonF([QPointF(pt[0], pt[1]) for pt in coords])
            poly_item = self.graphics_scene.addPolygon(polygon, ANNOTATION_PEN)
            text_pos = polygon.boundingRect().topLeft()
            qt_items.append(poly_item)

        text_item = self.graphics_scene.addText(f"{annotation['distance_meters']:.2f}m", ANNOTATION_FONT)
        text_item.setDefaultTextColor(ANNOTATION_PEN.color())
        text_item.setPos(text_pos - QPointF(0, ANNOTATION_FONT.pointSize() * 1.5))
        qt_items.append(text_item)
        
        # Store Qt graphics items with the annotation for easy removal
        annotation['qt_items'] = qt_items

    def draw_existing_annotations(self):
        if self.current_image_path:
            for ann in self.annotations_by_image.get(self.current_image_path, []):
                self.draw_annotation(ann)

    def update_annotation_table(self):
        """Populates the table with all annotations from all loaded images in chronological order."""
        self.annotation_table.blockSignals(True)
        self.annotation_table.setRowCount(0)
        
        sorted_annotations = []
        for img_path in self.image_files:
            if img_path in self.annotations_by_image:
                for ann in self.annotations_by_image[img_path]:
                    sorted_annotations.append(ann)
        
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
                    item.setBackground(QColor(80, 200, 120))
                else:
                    item.setBackground(Qt.transparent)

                self.annotation_table.setItem(row_idx, col_idx, item)

        # Highlight/select rows for current image and scroll to the first one
        self.annotation_table.clearSelection()
        scrolled = False
        for r_idx in range(self.annotation_table.rowCount()):
            item = self.annotation_table.item(r_idx, 0)
            if item:
                ann = item.data(Qt.UserRole)
                if ann and ann.get('image_path') == self.current_image_path:
                    for col_idx in range(self.annotation_table.columnCount()):
                        row_item = self.annotation_table.item(r_idx, col_idx)
                        if row_item:
                            row_item.setSelected(True)
                    if not scrolled:
                        self.annotation_table.scrollToItem(item)
                        scrolled = True

        self.annotation_table.blockSignals(False)

    def _on_table_selection_changed(self):
        """Slot to handle row selections in the annotation table."""
        selected_items = self.annotation_table.selectedItems()
        if not selected_items:
            return
        
        # Get the annotation data from the first column of the selected rows
        for item in selected_items:
            ann = item.data(Qt.UserRole)
            if ann:
                img_path = ann.get('image_path')
                if img_path and img_path != self.current_image_path:
                    self.load_image(img_path)
                break

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
        # Determine valid JSON path
        basename = os.path.basename(image_path)
        
        # Priority 1: Check in json_output_directory
        if self.json_output_directory:
            ann_path = os.path.join(self.json_output_directory, basename + ".json")
            if os.path.exists(ann_path):
                 return self._read_json_file(ann_path, image_path)

        # Priority 2: Check in same directory as image (Legacy)
        ann_path = image_path + ".json"
        if os.path.exists(ann_path):
            return self._read_json_file(ann_path, image_path)
            
        return []

    def _read_json_file(self, filepath: str, image_path: str) -> List[Dict]:
        try:
            with open(filepath, 'r') as f:
                annotations = json.load(f)
                # Add image_path to each loaded annotation for consistency
                for ann in annotations:
                    ann['image_path'] = image_path
                return annotations
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: Corrupt annotation file skipped: {filepath}")
            return []

    def save_annotations_for_image(self, image_path: str):
        """Saves annotations for a specific image to its JSON file."""
        if image_path is None: return
        
        basename = os.path.basename(image_path)
        # Use output dir if set, otherwise image dir
        dir_to_use = self.json_output_directory if self.json_output_directory else os.path.dirname(image_path)
        
        if not os.path.exists(dir_to_use):
            try:
                os.makedirs(dir_to_use)
            except OSError:
                self.status_label.setText(f"Error: Cannot create directory {dir_to_use}")
                return

        ann_path = os.path.join(dir_to_use, basename + ".json")
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
            # Use Worker Thread to prevent freezing
            from PyQt5.QtWidgets import QProgressDialog
            self.progress_dialog = QProgressDialog("Generating Depth Map...", "Cancel", 0, 0, self)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.show()

            self.dpt_worker = DPTWorker(self.dpt_model, self.dpt_processor, self.device, self.current_image_cv)
            self.dpt_worker.finished.connect(self.on_dpt_generated)
            self.dpt_worker.error.connect(self.on_dpt_error)
            self.dpt_worker.start()
        else:
            self.display_image(self.current_image_cv)
            self.draw_existing_annotations()
            self.show_dpt_btn.setText("Show DPT Depth")

    def on_dpt_generated(self, depth_map):
        self.progress_dialog.close()
        if depth_map is not None:
            # Normalize for visualization
            norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_img_bgr = cv2.cvtColor(norm_depth, cv2.COLOR_GRAY2BGR)
            self.display_image(depth_img_bgr)
            self.show_dpt_btn.setText("Show Real Image")
        else:
            self.status_label.setText("Failed to generate DPT depth map.")
            self.show_dpt_btn.setChecked(False)

    def on_dpt_error(self, msg):
        self.progress_dialog.close()
        self.status_label.setText(f"Error: {msg}")
        self.show_dpt_btn.setChecked(False)
        QMessageBox.critical(self, "DPT Error", msg)

    # Training logic has been moved to training_DPT.py

    def set_buttons_enabled(self, enabled: bool):
        """Enables or disables all major UI controls."""
        self.load_dir_btn.setEnabled(enabled)
        self.set_output_dir_btn.setEnabled(enabled)
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        # self.train_btn.setEnabled(enabled)
        self.show_dpt_btn.setEnabled(enabled)
        self.delete_annotation_btn.setEnabled(enabled)
        self.thumbnail_list_widget.setEnabled(enabled)

    def closeEvent(self, event):
        """Ensure annotations are saved when closing the application."""
        self.save_annotations_for_current_image()
        event.accept()

def main():
    """Main function to initialize and run the PyQt application."""
    from styles import apply_theme
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = AnnotationTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()