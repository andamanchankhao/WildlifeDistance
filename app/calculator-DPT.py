########################################################
### Dev : Andaman Chankhao andmanchankhao@gmail.com ####
### Modified by Gemini #################################
########################################################

import sys
import os
import cv2
import numpy as np
import csv
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGraphicsView, QGraphicsScene, QFileDialog,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QInputDialog,
    QListWidget, QListWidgetItem, QGraphicsPixmapItem, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont, QIcon
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize, pyqtSignal, QThread

# Import TensorFlow for model loading and prediction
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model # type: ignore
except ImportError:
    tf = None
    print("TensorFlow not installed. Please install it with 'pip install tensorflow' to use the model.")

# Import DPT related libraries for depth estimation
try:
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    import torch
    print("Hugging Face Transformers and PyTorch installed.")
except ImportError:
    DPTForDepthEstimation = None
    DPTImageProcessor = None
    torch = None
    print("Hugging Face Transformers or PyTorch not installed. Install with 'pip install transformers torch' to enable DPT-based calculation.")
    print("Note: PyTorch is required for DPT models from Hugging Face.")

# Import joblib and StandardScaler for loading the scaler
try:
    import joblib
    from sklearn.preprocessing import StandardScaler
except ImportError:
    joblib = None
    StandardScaler = None
    print("Scikit-learn or joblib not installed. Install with 'pip install scikit-learn joblib' to use the scaler.")


class CustomGraphicsView(QGraphicsView):
    # Custom signals for mouse events
    mouse_pressed = pyqtSignal(object)
    mouse_moved = pyqtSignal(object)

    def mousePressEvent(self, event):
        self.mouse_pressed.emit(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.mouse_moved.emit(event)
        super().mouseMoveEvent(event)


class CalculationWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, dpt_model, dpt_processor, regression_model, scaler, device, image_cv, point_coords):
        super().__init__()
        self.dpt_model = dpt_model
        self.dpt_processor = dpt_processor
        self.regression_model = regression_model
        self.scaler = scaler
        self.device = device
        self.image_cv = image_cv
        self.point_coords = point_coords

    def get_depth_for_point(self, depth_map, coords):
        x, y = coords
        h, w = depth_map.shape
        if 0 <= x < w and 0 <= y < h:
            return depth_map[y, x]
        return None

    def run(self):
        try:
            # DPT processing
            img_rgb = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB)
            inputs = self.dpt_processor(images=img_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.dpt_model(**inputs)
                predicted_depth_unscaled = outputs.predicted_depth.squeeze().cpu().numpy()
                depth_map = cv2.resize(predicted_depth_unscaled,
                                       (self.image_cv.shape[1], self.image_cv.shape[0]),
                                       interpolation=cv2.INTER_AREA)

            # Feature extraction
            point_depth_feature = self.get_depth_for_point(depth_map, self.point_coords)
            if point_depth_feature is None:
                self.error.emit("Could not extract a valid depth feature from the point.")
                return

            # Scaling and Prediction
            scaled_feature = self.scaler.transform(np.array([[point_depth_feature]]))[0][0]
            input_data = np.array([[scaled_feature]], dtype=np.float32)
            predicted_distance = self.regression_model.predict(input_data)[0][0]

            result = {
                'coords': self.point_coords,
                'predicted_distance': float(predicted_distance) # Ensure it's a standard float
            }
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"An error occurred during calculation: {str(e)}")


class DistanceCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wildlife Distance Calculator")
        self.setGeometry(100, 100, 1400, 900)

        self.model = None
        self.scaler = None
        self.image_files = []
        self.current_image_index = -1
        self.current_image_path = None
        self.current_image_cv = None
        self.current_pixmap_item = None

        self.clicked_point_data = None
        self.current_drawn_graphics_items = []

        self.image_data = {}
        self.next_annotation_id = 0

        self.dpt_processor = None
        self.dpt_model = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.is_dpt_view_active = False

        self.thumbnail_items = {}

        self.current_image_name_label = QLabel("No Image Loaded")
        self.status_label = QLabel("Ready")

        self.worker = None
        self.calculating_dialog = None

        self.init_ui()
        self.load_dpt_inference_model()
        # REMOVED: self.load_regression_model() - This is now done manually by the user

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_h_layout = QHBoxLayout(central_widget)

        # Left section: Thumbnail view and model/scaler loading
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_widget.setMinimumWidth(200)
        left_panel_widget.setMaximumWidth(250)

        # --- Model/Scaler Loading Controls ---
        model_scaler_group = QWidget()
        model_scaler_layout = QVBoxLayout(model_scaler_group)
        model_scaler_group.setStyleSheet("QWidget { border: 1px solid #cccccc; border-radius: 5px; }")

        self.load_keras_model_btn = QPushButton("Load Keras Model")
        self.load_keras_model_btn.clicked.connect(self.load_keras_model_file)
        self.model_status_label = QLabel("Model: <font color='red'>Not Loaded</font>")
        model_scaler_layout.addWidget(self.load_keras_model_btn)
        model_scaler_layout.addWidget(self.model_status_label)

        self.load_scaler_btn = QPushButton("Load Scaler File")
        self.load_scaler_btn.clicked.connect(self.load_joblib_scaler_file)
        self.scaler_status_label = QLabel("Scaler: <font color='red'>Not Loaded</font>")
        model_scaler_layout.addWidget(self.load_scaler_btn)
        model_scaler_layout.addWidget(self.scaler_status_label)
        left_panel_layout.addWidget(model_scaler_group)

        # --- Image Directory Controls ---
        self.load_dir_btn = QPushButton("Load Image Directory")
        self.load_dir_btn.clicked.connect(self.load_image_directory)
        left_panel_layout.addWidget(self.load_dir_btn)

        self.thumbnail_list_widget = QListWidget()
        self.thumbnail_list_widget.setViewMode(QListWidget.ListMode)
        self.thumbnail_list_widget.setIconSize(QSize(80, 60))
        self.thumbnail_list_widget.setResizeMode(QListWidget.Adjust)
        self.thumbnail_list_widget.setMovement(QListWidget.Static)
        self.thumbnail_list_widget.setWordWrap(True)
        self.thumbnail_list_widget.setTextElideMode(Qt.ElideMiddle)
        self.thumbnail_list_widget.itemClicked.connect(self.thumbnail_clicked)
        left_panel_layout.addWidget(self.thumbnail_list_widget)

        main_h_layout.addWidget(left_panel_widget)

        # Right section: Main content (Image display + Table)
        right_content_widget = QWidget()
        right_content_layout = QVBoxLayout(right_content_widget)

        top_layout = QHBoxLayout()

        # Image display area
        image_display_widget = QWidget()
        image_layout = QVBoxLayout(image_display_widget)

        self.current_image_name_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.current_image_name_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.current_image_name_label)

        self.graphics_view = CustomGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setMouseTracking(True)
        image_layout.addWidget(self.graphics_view)

        # Connect mouse events to custom methods
        self.graphics_view.mouse_pressed.connect(self.mouse_press_event)
        self.graphics_view.mouse_moved.connect(self.mouse_move_event)

        # Image navigation controls
        nav_controls_layout = QHBoxLayout()

        self.prev_btn = QPushButton("<")
        self.prev_btn.setFixedSize(30, 30)
        self.prev_btn.clicked.connect(self.load_previous_image)
        self.prev_btn.setEnabled(False)
        nav_controls_layout.addWidget(self.prev_btn)
        nav_controls_layout.addStretch(1)

        self.show_dpt_btn = QPushButton("Show DPT Image")
        self.show_dpt_btn.clicked.connect(self.toggle_dpt_view)
        nav_controls_layout.addWidget(self.show_dpt_btn)
        nav_controls_layout.addStretch(1)

        self.next_btn = QPushButton(">")
        self.next_btn.setFixedSize(30, 30)
        self.next_btn.clicked.connect(self.load_next_image)
        self.next_btn.setEnabled(False)
        nav_controls_layout.addWidget(self.next_btn)

        image_layout.addLayout(nav_controls_layout)
        top_layout.addWidget(image_display_widget, 2)

        # Table display and controls
        table_controls_widget = QWidget()
        table_layout = QVBoxLayout(table_controls_widget)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["ID", "Image", "Distance (m)", "Point Coords"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        table_layout.addWidget(self.results_table, 1)

        table_buttons_layout = QHBoxLayout()
        self.delete_row_btn = QPushButton("Delete Selected Row")
        self.delete_row_btn.clicked.connect(self.delete_selected_row)
        self.delete_row_btn.setEnabled(False)
        table_buttons_layout.addWidget(self.delete_row_btn)

        self.export_csv_btn = QPushButton("Export All to CSV")
        self.export_csv_btn.clicked.connect(self.export_to_csv)
        self.export_csv_btn.setEnabled(False)
        table_buttons_layout.addWidget(self.export_csv_btn)

        table_layout.addLayout(table_buttons_layout)
        table_layout.addStretch(1)
        top_layout.addWidget(table_controls_widget, 1)

        right_content_layout.addLayout(top_layout)
        main_h_layout.addWidget(right_content_widget, 3)

        self.status_label = QLabel("Ready. Please load Keras model and scaler.")
        self.statusBar().addWidget(self.status_label)
        self.results_table.itemSelectionChanged.connect(self.update_delete_button_state)
        self.update_navigation_buttons_state()

    def load_dpt_inference_model(self):
        if DPTForDepthEstimation is None or DPTImageProcessor is None or torch is None:
            self.status_label.setText("DPT dependencies not installed. DPT functionality disabled.")
            return

        try:
            model_name = "Intel/dpt-hybrid-midas"
            self.dpt_processor = DPTImageProcessor.from_pretrained(model_name)
            self.dpt_model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
            self.dpt_model.eval()
            self.status_label.setText(f"DPT model loaded on {self.device}. Please load Keras model and scaler.")
        except Exception as e:
            self.status_label.setText(f"Failed to load DPT model: {str(e)}. Check internet/dependencies.")
            print(f"DPT Inference Model Load Error: {e}")

    def load_keras_model_file(self):
        if tf is None:
            QMessageBox.critical(self, "Error", "TensorFlow is not installed. Cannot load Keras model.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Select Keras Model File", "", "Keras Models (*.keras *.h5)")
        if file_path:
            try:
                self.model = load_model(file_path)
                self.model_status_label.setText(f"Model: <font color='green'>{os.path.basename(file_path)}</font>")
                self.status_label.setText("Keras model loaded successfully.")
                QMessageBox.information(self, "Success", "Keras model loaded.")
            except Exception as e:
                self.model = None
                self.model_status_label.setText("Model: <font color='red'>Load Failed</font>")
                self.status_label.setText(f"Error loading Keras model.")
                QMessageBox.critical(self, "Load Error", f"Failed to load Keras model:\n{str(e)}")

    def load_joblib_scaler_file(self):
        if joblib is None:
            QMessageBox.critical(self, "Error", "Joblib is not installed. Cannot load scaler file.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Select Joblib Scaler File", "", "Joblib Files (*.joblib)")
        if file_path:
            try:
                self.scaler = joblib.load(file_path)
                # Quick check to see if it's a scaler
                if not hasattr(self.scaler, 'transform'):
                    raise ValueError("Loaded object is not a valid scaler (missing 'transform' method).")
                self.scaler_status_label.setText(f"Scaler: <font color='green'>{os.path.basename(file_path)}</font>")
                self.status_label.setText("Scaler file loaded successfully.")
                QMessageBox.information(self, "Success", "Scaler file loaded.")
            except Exception as e:
                self.scaler = None
                self.scaler_status_label.setText("Scaler: <font color='red'>Load Failed</font>")
                self.status_label.setText(f"Error loading scaler file.")
                QMessageBox.critical(self, "Load Error", f"Failed to load scaler file:\n{str(e)}")

    def load_image_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if directory:
            self.image_files = sorted([
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
            ])

            self.thumbnail_list_widget.clear()
            self.thumbnail_items = {}
            self.image_data = {}

            if self.image_files:
                for img_path in self.image_files:
                    self.add_thumbnail(img_path)

                self.current_image_index = 0
                self._display_image(self.image_files[self.current_image_index])
                self.status_label.setText(f"Loaded {len(self.image_files)} images. Click a point to measure distance.")
            else:
                self.status_label.setText("No supported image files found in directory.")
            self.update_navigation_buttons_state()
        else:
            self.status_label.setText("No directory selected.")

    def add_thumbnail(self, image_path):
        thumbnail_size = self.thumbnail_list_widget.iconSize()
        pixmap = QPixmap(image_path)
        icon = QIcon(pixmap.scaled(thumbnail_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        item = QListWidgetItem(icon, "")
        item.setData(Qt.UserRole, image_path)
        self.thumbnail_list_widget.addItem(item)
        self.thumbnail_items[image_path] = item
        self.update_thumbnail_status(image_path)

    def update_thumbnail_status(self, image_path):
        if image_path in self.thumbnail_items:
            item = self.thumbnail_items[image_path]
            has_data = image_path in self.image_data and bool(self.image_data[image_path])

            if has_data:
                item.setBackground(QColor(180, 255, 180))
                item.setToolTip(f"{os.path.basename(image_path)}\nDistances calculated.")
            else:
                item.setBackground(QColor(Qt.transparent))
                item.setToolTip(os.path.basename(image_path))

            if image_path == self.current_image_path:
                item.setSelected(True)
                self.thumbnail_list_widget.scrollToItem(item)
            else:
                item.setSelected(False)

    def thumbnail_clicked(self, item):
        image_path = item.data(Qt.UserRole)
        if image_path and os.path.exists(image_path):
            try:
                index = self.image_files.index(image_path)
                if index != self.current_image_index:
                    self.current_image_index = index
                    self._display_image(self.image_files[self.current_image_index])
                    self.update_navigation_buttons_state()
            except ValueError:
                self.status_label.setText(f"Error: Image path not found in list.")
        else:
            self.status_label.setText(f"Error: Invalid thumbnail path.")

    def _display_image(self, image_path):
        self.graphics_scene.clear()
        self.current_pixmap_item = None
        self.clear_pending_graphics()

        self.is_dpt_view_active = False
        self.show_dpt_btn.setText("Show DPT Image")

        self.current_image_path = image_path
        self.current_image_cv = cv2.imread(image_path)
        if self.current_image_cv is None:
            self.status_label.setText(f"Error loading image: {image_path}")
            return

        h, w, ch = self.current_image_cv.shape
        bytes_per_line = ch * w
        q_image = QImage(self.current_image_cv.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.current_pixmap_item = self.graphics_scene.addPixmap(pixmap)
        self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        self.current_image_name_label.setText(f"Current Image: {os.path.basename(image_path)}")
        self.status_label.setText(f"Displaying: {os.path.basename(image_path)}. Click a point to measure distance.")

        self.draw_all_annotations_for_current_image()
        self.update_results_table()
        self.update_thumbnail_status(image_path)

        try:
            self.current_image_index = self.image_files.index(image_path)
        except ValueError:
            self.current_image_index = -1

    def clear_pending_graphics(self):
        for item in self.current_drawn_graphics_items:
            if item in self.graphics_scene.items():
                self.graphics_scene.removeItem(item)
        self.current_drawn_graphics_items = []
        self.clicked_point_data = None

    def draw_annotation_on_scene(self, annotation_data, is_pending=False):
        coords = annotation_data["coords"]
        x, y = coords

        if is_pending:
            pen_color = QColor(0, 200, 0)
            text_color = QColor(255, 255, 0)
            display_text = "Calculating..."
        else:
            pen_color = QColor(255, 0, 0)
            text_color = QColor(255, 0, 0)
            ann_id = annotation_data["id"]
            predicted_distance = annotation_data["predicted_distance"]
            display_text = f"ID:{ann_id} | {predicted_distance:.2f}m"

        point_item = self.graphics_scene.addText("X")
        point_item.setDefaultTextColor(pen_color)
        point_item.setFont(QFont("Arial", 20, QFont.Bold))
        rect = point_item.boundingRect()
        point_item.setPos(x - rect.width() / 2, y - rect.height() / 2)

        text_item = self.graphics_scene.addText(display_text)
        text_item.setDefaultTextColor(text_color)
        text_item.setFont(QFont("Arial", 30, QFont.Bold))
        text_item.setPos(x, y - 40)

        annotation_data['graphics_items'] = [point_item, text_item]
        return point_item, text_item

    def draw_all_annotations_for_current_image(self):
        if self.current_image_path in self.image_data:
            for annotation in self.image_data[self.current_image_path]:
                for item in annotation.get('graphics_items', []):
                    if item in self.graphics_scene.items():
                        self.graphics_scene.removeItem(item)
                annotation['graphics_items'] = []

                point_item, text_item = self.draw_annotation_on_scene(annotation, is_pending=False)
                annotation['graphics_items'] = [point_item, text_item]

    def update_results_table(self):
        self.results_table.setRowCount(0)
        if self.current_image_path in self.image_data:
            for row_idx, annotation in enumerate(self.image_data[self.current_image_path]):
                self.results_table.insertRow(row_idx)

                self.results_table.setItem(row_idx, 0, QTableWidgetItem(str(annotation['id'])))
                self.results_table.setItem(row_idx, 1, QTableWidgetItem(os.path.basename(annotation['image_path'])))
                self.results_table.setItem(row_idx, 2, QTableWidgetItem(f"{annotation['predicted_distance']:.2f}"))
                point_str = f"({annotation['coords'][0]}, {annotation['coords'][1]})"
                self.results_table.setItem(row_idx, 3, QTableWidgetItem(point_str))

        self.export_csv_btn.setEnabled(any(self.image_data.values()))
        self.update_delete_button_state()

    def update_delete_button_state(self):
        self.delete_row_btn.setEnabled(len(self.results_table.selectedIndexes()) > 0)

    def delete_selected_row(self):
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows: return

        row_to_delete_idx = selected_rows[0].row()
        ann_id_item = self.results_table.item(row_to_delete_idx, 0)
        if ann_id_item is None: return

        ann_id_to_delete = int(ann_id_item.text())

        if self.current_image_path in self.image_data:
            ann_to_remove = next((ann for ann in self.image_data[self.current_image_path] if ann['id'] == ann_id_to_delete), None)

            if ann_to_remove:
                for item in ann_to_remove.get('graphics_items', []):
                    if item in self.graphics_scene.items():
                        self.graphics_scene.removeItem(item)

                self.image_data[self.current_image_path].remove(ann_to_remove)
                QMessageBox.information(self, "Deletion Successful", f"Annotation ID {ann_id_to_delete} deleted.")

        self.update_results_table()
        self.update_thumbnail_status(self.current_image_path)

    def load_previous_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self._display_image(self.image_files[self.current_image_index])
            self.update_navigation_buttons_state()

    def load_next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self._display_image(self.image_files[self.current_image_index])
            self.update_navigation_buttons_state()

    def update_navigation_buttons_state(self):
        can_navigate = len(self.image_files) > 0
        self.prev_btn.setEnabled(can_navigate and self.current_image_index > 0)
        self.next_btn.setEnabled(can_navigate and self.current_image_index < len(self.image_files) - 1)
        self.export_csv_btn.setEnabled(any(self.image_data.values()))

    def mouse_press_event(self, event):
        if self.current_image_cv is None:
            self.status_label.setText("Please load an image directory first.")
            return
        if self.is_dpt_view_active:
            self.status_label.setText("Cannot select point on DPT depth map. Switch to real image view first.")
            return

        # Check if model and scaler are loaded before allowing a click
        if self.model is None or self.scaler is None:
            QMessageBox.warning(self, "Files Not Loaded", "Please load both the Keras model and the scaler file before calculating distances.")
            return

        if event.button() == Qt.LeftButton:
            scene_pos = self.graphics_view.mapToScene(event.pos())
            self.clear_pending_graphics()

            x, y = int(scene_pos.x()), int(scene_pos.y())
            self.clicked_point_data = {'coords': [x, y]}

            pending_annotation = {'coords': [x, y]}
            point_item, text_item = self.draw_annotation_on_scene(pending_annotation, is_pending=True)
            self.current_drawn_graphics_items = [point_item, text_item]

            self.status_label.setText(f"Point selected at ({x}, {y}). Calculating distance...")
            self.calculate_distance()

    def mouse_move_event(self, event):
        if self.current_image_cv is None: return
        scene_pos = self.graphics_view.mapToScene(event.pos())
        self.status_label.setText(f"Mouse: ({int(scene_pos.x())}, {int(scene_pos.y())})")

    def calculate_distance(self):
        # Redundant check, but good for safety
        if self.model is None or self.scaler is None:
            QMessageBox.warning(self, "Model/Scaler Not Loaded", "The Keras model or scaler is not loaded.")
            self.clear_pending_graphics()
            return
        if self.current_image_cv is None:
            QMessageBox.warning(self, "No Image", "Please load an image.")
            self.clear_pending_graphics()
            return
        if self.clicked_point_data is None:
            self.clear_pending_graphics()
            return
        if self.dpt_model is None or self.dpt_processor is None:
            QMessageBox.warning(self, "DPT Model Not Loaded", "DPT model is not loaded.")
            self.clear_pending_graphics()
            return

        self.calculating_dialog = QMessageBox(self)
        self.calculating_dialog.setWindowTitle("Processing")
        self.calculating_dialog.setText("Calculating distance, please wait...")
        self.calculating_dialog.setIcon(QMessageBox.Information)
        self.calculating_dialog.setStandardButtons(QMessageBox.NoButton)
        self.calculating_dialog.setModal(True)
        self.calculating_dialog.open()

        x, y = self.clicked_point_data['coords']

        self.worker = CalculationWorker(
            dpt_model=self.dpt_model,
            dpt_processor=self.dpt_processor,
            regression_model=self.model,
            scaler=self.scaler,
            device=self.device,
            image_cv=self.current_image_cv.copy(),
            point_coords=[x, y]
        )
        self.worker.finished.connect(self.handle_calculation_finished)
        self.worker.error.connect(self.handle_calculation_error)
        self.worker.start()

    def handle_calculation_finished(self, result):
        if self.calculating_dialog:
            self.calculating_dialog.accept()

        self.clear_pending_graphics()

        x, y = result['coords']
        predicted_distance = result['predicted_distance']

        new_annotation = {
            'id': self.next_annotation_id,
            'image_path': self.current_image_path,
            'coords': [x, y],
            'predicted_distance': predicted_distance,
            'graphics_items': []
        }
        self.next_annotation_id += 1

        if self.current_image_path not in self.image_data:
            self.image_data[self.current_image_path] = []
        self.image_data[self.current_image_path].append(new_annotation)

        self.draw_all_annotations_for_current_image()
        self.update_results_table()
        self.update_thumbnail_status(self.current_image_path)

        self.status_label.setText(f"Distance for ID {new_annotation['id']}: {predicted_distance:.2f} meters.")

    def handle_calculation_error(self, error_message):
        if self.calculating_dialog:
            self.calculating_dialog.accept()

        self.clear_pending_graphics()
        QMessageBox.critical(self, "Calculation Error", error_message)
        self.status_label.setText(f"Calculation failed.")

    def export_to_csv(self):
        all_data_for_export = []
        for image_path, annotations in self.image_data.items():
            for ann in annotations:
                row = {
                    'image_name': os.path.basename(image_path),
                    'predicted_distance': ann['predicted_distance'],
                    'x': ann['coords'][0],
                    'y': ann['coords'][1]
                }
                all_data_for_export.append(row)

        if not all_data_for_export:
            QMessageBox.warning(self, "No Data", "No distances have been calculated yet.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Predicted Distances", "wildlife_predictions.csv", "CSV Files (*.csv)")

        if file_path:
            try:
                fieldnames = ['image_name', 'predicted_distance', 'x', 'y']
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_data_for_export)
                self.status_label.setText(f"Successfully exported {len(all_data_for_export)} predictions.")
                QMessageBox.information(self, "Export Complete", f"Data exported to:\n{os.path.abspath(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"An error occurred during CSV export: {str(e)}")

    def toggle_dpt_view(self):
        if self.current_image_cv is None or self.dpt_model is None: return

        self.graphics_scene.clear()
        self.current_pixmap_item = None
        self.clear_pending_graphics()

        if self.current_image_path in self.image_data:
            for pred in self.image_data[self.current_image_path]:
                pred['graphics_items'] = []

        if not self.is_dpt_view_active:
            self.status_label.setText("Generating DPT depth map...")
            try:
                img_rgb = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2RGB)
                inputs = self.dpt_processor(images=img_rgb, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.dpt_model(**inputs)
                    predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()

                normalized_depth = cv2.normalize(predicted_depth, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8U)

                h, w = normalized_depth.shape
                q_image = QImage(normalized_depth.data, w, h, w, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(q_image)
                self.current_pixmap_item = self.graphics_scene.addPixmap(pixmap)

                self.show_dpt_btn.setText("Show Real Image")
                self.status_label.setText("DPT depth map displayed.")
                self.is_dpt_view_active = True

            except Exception as e:
                self.status_label.setText(f"Error generating DPT map: {str(e)}")
                self.is_dpt_view_active = False
        else:
            h, w, ch = self.current_image_cv.shape
            bytes_per_line = ch * w
            q_image = QImage(self.current_image_cv.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            self.current_pixmap_item = self.graphics_scene.addPixmap(pixmap)
            self.draw_all_annotations_for_current_image()

            self.show_dpt_btn.setText("Show DPT Image")
            self.status_label.setText("Real image displayed.")
            self.is_dpt_view_active = False

        self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DistanceCalculator()
    window.show()
    sys.exit(app.exec_())