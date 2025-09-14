import os

path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(path):
    print("✅ Found tesseract.exe at:", path)
else:
    print("❌ tesseract.exe not found at:", path)
