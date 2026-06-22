import sys
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Determine OS
is_mac = sys.platform == 'darwin'
is_win = sys.platform == 'win32'

# Icon selection
icon_file = 'icon.icns' if is_mac else 'icon.ico'

# Collect all data, binaries, and hidden imports for ultralytics
tmp_ret = collect_all('ultralytics')
datas = tmp_ret[0]
binaries = tmp_ret[1]
hiddenimports = tmp_ret[2]

# Add other manual data files
datas += [
    ('annotate_train_DPT.py', '.'),
    ('calculator_DPT.py', '.'),
    ('training_DPT.py', '.'), # Include the new training module
    ('styles.py', '.'), 
    ('icon.png', '.'),
]

a = Analysis(
    ['main_app.py'], # Entry point changed to main_app.py
    pathex=[], 
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='WildlifeDistance',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file
)

# BUNDLE is only for macOS .app bundles
if is_mac:
    app = BUNDLE(
        exe,
        name='WildlifeDistance.app',
        icon='icon.icns',
        bundle_identifier=None,
    )
