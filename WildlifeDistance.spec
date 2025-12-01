import sys
import os

block_cipher = None

# Determine OS
is_mac = sys.platform == 'darwin'
is_win = sys.platform == 'win32'

# Icon selection
icon_file = 'icon.icns' if is_mac else 'icon.ico'

a = Analysis(
    ['launcher.py'],
    pathex=[], 
    binaries=[],
    datas=[
        ('annotate_train_DPT.py', '.'),
        ('calculator_DPT.py', '.'),
        ('styles.py', '.'), # Include the new styles file
        ('icon.png', '.'),
    ],
    hiddenimports=[],
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
