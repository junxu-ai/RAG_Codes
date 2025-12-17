import openai
import base64
import fitz  # PyMuPDF
from io import BytesIO

def analyze_pdf(pdf_file_path):
    """
    Analyzes a PDF document using the OpenAI Vision API and GPT-4v.

    Args:
      pdf_file_path: Path to the PDF file.

    Returns:
      A dictionary containing the extracted information.
    """
    
    # Initialize OpenAI client
    client = openai.OpenAI()
    
    # 1. Convert PDF to images
    images = convert_pdf_to_images(pdf_file_path)
    
    # 2. Call OpenAI Vision API for each image
    extracted_data = {}
    for i, image_bytes in enumerate(images):
        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the key information from this document, including the title, author, date, and any important findings or conclusions."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=1000
        )
        extracted_data[f"page_{i+1}"] = response.choices[0].message.content
    
    # 3. Aggregate and process the extracted information
    # This would depend on your specific requirements
    
    return extracted_data

def convert_pdf_to_images(pdf_path):
    """
    Converts PDF pages to images using PyMuPDF
    """
    images = []
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_data = pix.tobytes("png")
        images.append(img_data)
    
    pdf_document.close()
    return images

if __name__ == "__main__":
    # Example usage
    pdf_file = "example.pdf"
    extracted_info = analyze_pdf(pdf_file)
    print(extracted_info)