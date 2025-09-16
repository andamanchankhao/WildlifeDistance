# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['/Users/andamanchankhao/Workspace/Distance-calculator/app/laucher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('/Users/andamanchankhao/Workspace/Distance-calculator/app/annotate-train-DPT.py', '.'),
        ('/Users/andamanchankhao/Workspace/Distance-calculator/app/calculator-DPT.py', '.')
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
    name='WildlifeTool',
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
)
app = BUNDLE(
    exe,
    name='WildlifeTool.app',
    icon='/Users/andamanchankhao/Workspace/Distance-calculator/app/icon.icns',
    bundle_identifier=None,
)
