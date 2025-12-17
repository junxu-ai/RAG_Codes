from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTRect

def extract_text_with_style(pdf_path):
    # Extract pages with layout information
    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                # Extract text with font information
                fonts = {}
                for text_line in element:
                    for char in text_line:
                        if isinstance(char, LTChar):
                            font_name = char.fontname
                            font_size = char.size
                            fonts[(font_name, font_size)] = fonts.get((font_name, font_size), '') + char.get_text()
                
                # Apply rules based on font properties
                for (font_name, font_size), text in fonts.items():
                    if font_size > 14:  # Assume headers
                        print(f"Header: {text}")
                    else:  # Assume body text
                        print(f"Body: {text}")


if __name__ == "__main__":
    extract_text_with_style("path_to_your_pdf.pdf")



