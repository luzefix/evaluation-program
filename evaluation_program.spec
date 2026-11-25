# PyInstaller spec: build a stand-alone app with
#     pyinstaller evaluation_program.spec
# The result is in dist/EvaluationProgram/ (Windows: EvaluationProgram.exe, macOS: .app).
import sys
from pathlib import Path

root = Path(SPECPATH)

a = Analysis(
    [str(root / "run.py")],
    pathex=[str(root / "src")],
    datas=[(str(root / "src/evalprog/materials/data"), "evalprog/materials/data")],
    hiddenimports=["matplotlib.backends.backend_qtagg"],
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="EvaluationProgram", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="EvaluationProgram")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="EvaluationProgram.app", bundle_identifier=None)
