# -*- mode: python ; coding: utf-8 -*-
import platform
import os

# --- Determine Icon Path ---
# This section checks for the correct icon file based on the operating system.
# If the icon file is not found, it defaults to None.
icon_path = None
if platform.system() == "Windows":
    if os.path.exists('icon.ico'):
        icon_path = 'icon.ico'
elif platform.system() == "Darwin": # 'Darwin' is the system name for macOS
    if os.path.exists('icon.icns'):
        icon_path = 'icon.icns'

# --- Data files ---
# This list includes all non-python files needed for your application to run.
datas = []
if icon_path:
    # Add the icon to the datas list only if it was found.
    datas.append((os.path.basename(icon_path), '.'))

# --- Main Analysis ---
# THIS IS THE CRITICAL PART: All of your python scripts must be listed here
# so PyInstaller can find all their dependencies (TensorFlow, PyTorch, etc.)
a = Analysis(
    ['launcher.py', 'annotate_train_DPT.py', 'calculator_DPT.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

# --- Executable ---
# This defines the main executable file and sets the icon if it exists.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WildlifeDistance',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon=icon_path
)

# --- Platform-Specific Bundling ---
# This "if/else" block makes the script work on BOTH macOS and Windows.
if platform.system() == "Darwin":
    # Create a .app bundle for macOS
    app = BUNDLE(
        exe,
        name='WildlifeDistance.app',
        icon=icon_path,
        bundle_identifier=None,
    )
else:
    # For Windows, create a standard folder containing the executable.
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='WildlifeDistance'
    )

