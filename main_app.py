
import sys
import os
import platform

if platform.system() == "Darwin":
    # Prevent OpenMP and MKL thread conflicts causing segfaults on macOS
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QTabWidget, QLabel, QHBoxLayout
)
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Import Refactored Components
from annotate_train_DPT import AnnotationTool
from calculator_DPT import DistanceCalculator
from training_DPT import TrainingTool
from styles import apply_theme

# --- Opt-1 & Opt-2: Shared DPT model loader ---
# The previous design loaded three independent copies of Intel/dpt-hybrid-midas
# (one per tab), tripling RAM usage and startup time.  We now load it ONCE here
# and pass the shared (processor, model, device) tuple into every tool.
try:
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    import torch
    _DPT_AVAILABLE = True
except ImportError:
    DPTForDepthEstimation = DPTImageProcessor = torch = None
    _DPT_AVAILABLE = False

DPT_MODEL_NAME = "Intel/dpt-hybrid-midas"


def load_shared_dpt_model():
    """
    Load and return (processor, model, device) once for the whole application.

    Opt-2: Converts the model to FP16 when a CUDA GPU is available,
    which roughly doubles inference throughput and halves VRAM use with
    negligible accuracy loss for depth estimation.

    Returns (None, None, 'cpu') if dependencies are missing or loading fails.
    """
    if not _DPT_AVAILABLE:
        print("WARNING: 'transformers' or 'torch' not installed. DPT disabled.")
        return None, None, "cpu"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = DPTImageProcessor.from_pretrained(DPT_MODEL_NAME)
        model = DPTForDepthEstimation.from_pretrained(DPT_MODEL_NAME).to(device)
        model.eval()

        # Opt-2: FP16 reduces VRAM by ~50 % and speeds up inference on CUDA.
        if device == "cuda":
            model = model.half()
            print("INFO: DPT model converted to FP16 for CUDA acceleration.")

        print(f"INFO: Shared DPT model loaded on '{device}'.")
        return processor, model, device
    except Exception as e:
        print(f"ERROR: Could not load shared DPT model: {e}")
        return None, None, device


class DPTModelLoaderThread(QThread):
    model_loaded = pyqtSignal(object, object, str)

    def run(self):
        processor, model, device = load_shared_dpt_model()
        self.model_loaded.emit(processor, model, device)


def resource_path(relative_path):
    """Get absolute path to resource — works for dev and for PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class WildlifeDistanceApp(QMainWindow):
    """
    The main application window hosting specific tools in tabs.
    Theme: Minimal Red.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wildlife Distance")
        self.setGeometry(100, 100, 1600, 950)

        # Opt-1: Store shared DPT components to inject into each tool.
        self._dpt_processor = "async"
        self._dpt_model = "async"
        self._device = "cpu"

        icon_path = resource_path('icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.init_ui()
        self.start_model_loader()

    def init_ui(self):
        # Central Widget & Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Header (Logo + Title) ---
        header_widget = QWidget()
        header_widget.setStyleSheet("background-color: white;")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 20, 20, 20)

        # Logo
        logo_label = QLabel()
        logo_path = resource_path('icon.png')
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        header_layout.addWidget(logo_label)

        # Title
        title_label = QLabel("Wildlife Distance")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(title_label)

        # Status of DPT Model Loading
        self.ai_status_label = QLabel(" ⚪ Loading AI model...")
        self.ai_status_label.setStyleSheet("color: #777777; font-size: 12px; margin-left: 10px;")
        header_layout.addWidget(self.ai_status_label, 0, Qt.AlignVCenter)

        header_layout.addStretch()

        # Version Badge (Top Right)
        version_label = QLabel("Demo Version")
        version_label.setStyleSheet("""
            color: #888888;
            font-size: 12px;
            font-weight: 500;
            background-color: #f0f0f0;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 4px 14px;
        """)
        header_layout.addWidget(version_label, 0, Qt.AlignVCenter)

        main_layout.addWidget(header_widget)

        # --- Tab Widget ---
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)  # Cleaner look for tabs

        # Opt-1: Pass the single shared DPT model to each tool so they do NOT
        # each load their own independent copy.
        self.annotation_tool = AnnotationTool(
            self,
            shared_dpt_processor=self._dpt_processor,
            shared_dpt_model=self._dpt_model,
            shared_device=self._device,
        )
        self.training_tool = TrainingTool(
            self,
            shared_dpt_processor=self._dpt_processor,
            shared_dpt_model=self._dpt_model,
            shared_device=self._device,
        )
        self.calculator_tool = DistanceCalculator(
            self,
            shared_dpt_processor=self._dpt_processor,
            shared_dpt_model=self._dpt_model,
            shared_device=self._device,
        )

        # Add Tabs
        self.tabs.addTab(self.annotation_tool, "Annotate")
        self.tabs.addTab(self.training_tool, "Train Model")
        self.tabs.addTab(self.calculator_tool, "Distance Calculator")

        main_layout.addWidget(self.tabs)

        # Note: Each tool has its own local status bar.

    def start_model_loader(self):
        self.loader_thread = DPTModelLoaderThread()
        self.loader_thread.model_loaded.connect(self.on_model_loaded)
        self.loader_thread.start()

    def on_model_loaded(self, processor, model, device):
        self._dpt_processor = processor
        self._dpt_model = model
        self._device = device

        # Distribute model references dynamically to the tabs
        self.annotation_tool.dpt_processor = processor
        self.annotation_tool.dpt_model = model
        self.annotation_tool.device = device

        self.training_tool.dpt_processor = processor
        self.training_tool.dpt_model = model
        self.training_tool.device = device

        self.calculator_tool.dpt_processor = processor
        self.calculator_tool.dpt_model = model
        self.calculator_tool.device = device

        # Update sub-status bars
        self.annotation_tool.status_label.setText("AI model loaded. Ready for annotations.")
        self.training_tool.status_label.setText("AI model loaded. Ready for training.")
        self.calculator_tool.status_label.setText("AI model loaded. Ready for calculations.")

        # Update header status
        if model is not None:
            self.ai_status_label.setText(" 🟢 AI Model Ready")
            self.ai_status_label.setStyleSheet("color: #2e7d32; font-weight: bold; font-size: 12px; margin-left: 10px;")
        else:
            self.ai_status_label.setText(" 🔴 AI Model Load Failed (Offline?)")
            self.ai_status_label.setStyleSheet("color: #c82828; font-weight: bold; font-size: 12px; margin-left: 10px;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_theme(app)

    window = WildlifeDistanceApp()
    window.show()

    sys.exit(app.exec_())
