import unicodedata
import re
import config


def normalize_text(text: str) -> str:
    """
    Normalize text: NFC, strip whitespace BUT preserve paragraphs.
    """
    if not text:
        return ""
    
    # Unicode NFC
    text = unicodedata.normalize('NFC', text)
    
    # Replace zero-width spaces
    text = text.replace('\u200b', '')
    
    # Normalize newline characters
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Replace horizontal whitespace (tabs) with single space
    # BUT we want to keep \n. 
    # Current regex in old code `re.sub(r'[ \t\f\v]+', ' ', text)` MIGHT replace \n if included?
    # Usually \s includes \n. [ \t\f\v] does NOT include \n.
    text = re.sub(r'[ \t\f\v]+', ' ', text)
    
    # Fix multiple newlines -> max 2 newlines (paragraph break)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    return text.strip()

def split_text_with_overlap(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """
    Word-based sliding window preserving some structure if possible.
    """
    words = text.split()
    chunks = []
    if not words:
        return []
        
    step = chunk_size - overlap
    if step <= 0: step = 1
    
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        
    return chunks

def smart_legal_chunking(text: str) -> list[str]:
    """
    Heuristic: Split by 'Điều', 'Khoản', 'Chương'.
    """
    # 1. Normalize (preserving paragraphs)
    text = normalize_text(text)
    
    # 2. Heuristic split by "Điều \d+"
    # Use lookahead to keep the delimiter
    # Also support "Chương \w+" if wanted, but Điều is most granular useful unit.
    pattern = r'(?=(?:^|\n|\s)Điều\s+\d+\.)'
    parts = re.split(pattern, text)
    
    final_chunks = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # If part is huge (e.g. whole chapter), split it further
        if len(part.split()) > config.CHUNK_SIZE * 1.5:
            sub_chunks = split_text_with_overlap(part)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(part)
            
    return final_chunks
