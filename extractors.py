import os
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from readability import Document
    import lxml.html
except ImportError:
    Document = None

try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
except ImportError:
    pytesseract = None

from db import DocVersion # Just for type hinting if needed

def extract_text_from_file(file_path: str, mime_type: str = None) -> tuple[str, bool]:
    """
    Extract text from file.
    Returns: (text, used_ocr)
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext in [".html", ".htm"]:
        return extract_html(file_path)
    elif ext == ".txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read(), False
    else:
        # Fallback or simple read
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(), False
        except:
            return "", False

def extract_pdf(file_path: str) -> tuple[str, bool]:
    if not PdfReader:
        return "Missing pypdf library", False
        
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            t = page.extract_text() or "" # Fix None return
            text += t + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return "", False
        
    text = text.strip()
    
    # OCR Fallback Check
    # If text is extremely short relative to file size or page count, assume scan.
    # Simple heuristic: averaged < 50 chars per page
    if len(text) / max(len(reader.pages), 1) < 50:
        print(f"PDF {file_path} seems to be scanned. Attempting OCR...")
        ocr_text = run_ocr_pdf(file_path)
        if ocr_text:
            return ocr_text, True
            
    return text, False

def run_ocr_pdf(file_path: str) -> str:
    if not pytesseract:
        print("pytesseract not installed. Skipping OCR.")
        return ""
        
    # This requires pdf2image and poppler installed on system
    try:
        images = convert_from_path(file_path)
        text = ""
        for img in images:
            # lang='vie' for Vietnamese
            text += pytesseract.image_to_string(img, lang='vie') + "\n"
        return text
    except Exception as e:
        print(f"OCR Failed: {e}")
        return ""

def extract_html(file_path: str) -> tuple[str, bool]:
    if not Document:
        # Basic fallback
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read(), False
            
    with open(file_path, 'r', encoding='utf-8') as f:
        html = f.read()
        
    # Readability for content extraction (removes menu/sidebar)
    try:
        doc = Document(html)
        summary_html = doc.summary() 
        # Convert summary HTML to text using lxml
        cleaned_text = lxml.html.document_fromstring(summary_html).text_content()
    except Exception as e:
        print(f"Readability failed: {e}, using basic extraction")
        soup = lxml.html.document_fromstring(html)
        cleaned_text = soup.text_content()
    
    return cleaned_text, False
