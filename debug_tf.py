
import sys
import os

print(f"Python Version: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    import tensorflow as tf
    print(f"TensorFlow Version: {tf.__version__}")
    try:
        import keras
        print(f"Keras Version: {keras.__version__}")
    except ImportError:
        print("Keras import failed")
except ImportError as e:
    print(f"TensorFlow import failed: {e}")

# Test file writing
try:
    with open("test_write.txt", "w") as f:
        f.write("test")
    print("File write successful")
    os.remove("test_write.txt")
except Exception as e:
    print(f"File write failed: {e}")
