
try:
    from pypdf import PdfReader
    reader = PdfReader("template/Template_Jurnal_JIT_2025.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(text)
except ImportError:
    print("pypdf not installed")
except Exception as e:
    print(f"Error: {e}")
