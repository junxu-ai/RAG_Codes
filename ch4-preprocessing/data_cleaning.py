from unstructured.cleaners.core import clean_text, replace_unicode_quotes
import re

def clean_document_text(text):
    """
    Comprehensive document cleaning pipeline
    """
    # Basic text cleaning
    text = clean_text(text)
    text = replace_unicode_quotes(text)
    
    # Remove special characters and normalize whitespace
    text = re.sub(r'[^\w\s.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Deduplicate paragraphs
    paragraphs = text.split('\n')
    unique_paragraphs = list(dict.fromkeys(paragraphs))
    
    # Remove boilerplate content
    cleaned_paragraphs = [
        p for p in unique_paragraphs 
        if len(p.split()) > 5  # Remove very short segments
        and not any(boilerplate in p.lower() for boilerplate in 
                   ['cookie policy', 'terms of service', 'all rights reserved'])
    ]
    
    return '\n'.join(cleaned_paragraphs)

def validate_document_quality(text):
    """
    Validate document quality metrics
    """
    words = text.split()
    word_count = len(words)
    
    metrics = {
        'length': len(text),
        'avg_word_length': sum(len(word) for word in words) / word_count if word_count > 0 else 0,
        'special_char_ratio': len(re.findall(r'[^\w\s]', text)) / len(text) if len(text) > 0 else 0
    }    
    return metrics

if __name__ == '__main__':
    # Example usage
    text = "This is a sample document with some special characters like !@#$%^&*() and some boilerplate content."
    cleaned_text = clean_document_text(text)
    quality_metrics = validate_document_quality(cleaned_text)
    print(f"Cleaned text: {cleaned_text}")
    print(f"Quality metrics: {quality_metrics}")