
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
    QTabWidget, QLabel, QHBoxLayout, QFrame, QProgressBar, QDesktopWidget,
    QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor
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
    status_changed = pyqtSignal(str)

    def run(self):
        self.status_changed.emit("Initializing environment...")
        self.msleep(150)  # Soft delay for smooth visual transition
        self.status_changed.emit("Loading DPT AI weights (470 MB)...")
        processor, model, device = load_shared_dpt_model()
        if model is not None:
            self.status_changed.emit("AI model ready!")
        else:
            self.status_changed.emit("Failed to load AI model.")
        self.msleep(300)  # Let the user see the success status briefly
        self.model_loaded.emit(processor, model, device)


class WildlifeDistanceSplashScreen(QWidget):
    """
    A beautiful, modern, frameless splash screen shown during application startup.
    Features the app logo, bold title, dynamic loading status, and a sleek red progress bar.
    """
    def __init__(self):
        super().__init__()
        # Frameless, stays on top, splash window flags
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SplashScreen)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.init_ui()
        self.center_on_screen()

    def init_ui(self):
        # Main background container frame
        self.container = QFrame(self)
        self.container.setObjectName("SplashContainer")
        # Apply modern minimal/card look with rounded borders
        self.container.setStyleSheet("""
            QFrame#SplashContainer {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
            }
        """)
        
        # Shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setOffset(0, 6)
        self.container.setGraphicsEffect(shadow)

        # Layout inside container
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(40, 45, 40, 45)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignCenter)

        # Logo
        self.logo_label = QLabel()
        logo_path = resource_path('icon.png')
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(pixmap)
        layout.addWidget(self.logo_label, 0, Qt.AlignCenter)

        # Title
        self.title_label = QLabel("Wildlife Distance")
        self.title_label.setFont(QFont("Inter", 22, QFont.Bold))
        self.title_label.setStyleSheet("color: #c82828; margin-top: 5px;")
        layout.addWidget(self.title_label, 0, Qt.AlignCenter)

        # Subtitle / Version
        self.version_label = QLabel("Demo Version")
        self.version_label.setFont(QFont("Inter", 10, QFont.Medium))
        self.version_label.setStyleSheet("""
            color: #888888;
            font-weight: 500;
            background-color: #f0f0f0;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 2px 10px;
        """)
        layout.addWidget(self.version_label, 0, Qt.AlignCenter)

        layout.addSpacing(10)

        # Loading message
        self.status_label = QLabel("Loading AI models...")
        self.status_label.setFont(QFont("Inter", 11))
        self.status_label.setStyleSheet("color: #555555;")
        layout.addWidget(self.status_label, 0, Qt.AlignCenter)

        # Custom red progress bar (sleek thin line)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate/busy state
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #f0f0f0;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #c82828;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Set size and outer layout
        self.setFixedSize(450, 390)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(15, 15, 15, 15)
        outer_layout.addWidget(self.container)

    def set_status_text(self, text: str):
        self.status_label.setText(text)
        QApplication.processEvents()  # Ensure UI updates immediately

    def center_on_screen(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


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
        self.prepare_model_loader()

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

    def prepare_model_loader(self):
        self.loader_thread = DPTModelLoaderThread()
        self.loader_thread.model_loaded.connect(self.on_model_loaded)

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

        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_theme(app)

    # 1. Create and show Splash Screen
    splash = WildlifeDistanceSplashScreen()
    splash.show()
    app.processEvents()

    # 2. Instantiate main app (initially hidden)
    window = WildlifeDistanceApp()

    # 3. Connect thread status signals to splash screen
    window.loader_thread.status_changed.connect(splash.set_status_text)

    # 4. Connect finish signal to close splash and show main window
    def launch_main_app(processor, model, device):
        splash.close()
        window.show()
        window.raise_()
        window.activateWindow()

    window.loader_thread.model_loaded.connect(launch_main_app)

    # 5. Start the background model loader thread
    window.loader_thread.start()

    sys.exit(app.exec_())
