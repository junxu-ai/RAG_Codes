# write a python code to merge the pdf files given in the list.
# add one given string as a header to the first page of the merged pdf
# add one given string as a foot to the first page of the merged pdf

# write a python code to merge the pdf files given in the list.
# add one given string as a header to the first page of the merged pdf
# add one given string as a foot to the first page of the merged pdf

import os
from PyPDF2 import PdfReader, PdfWriter, PageObject
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

def merge_pdfs_with_header_footer(pdf_list, output_path, header_text="", footer_text=""):
    """
    Merge PDF files and add header and footer to the first page.
    
    Args:
        pdf_list (list): List of PDF file paths to merge
        output_path (str): Path for the output merged PDF
        header_text (str): Header text for the first page
        footer_text (str): Footer text for the first page
    """
    # Create PDF writer object
    pdf_writer = PdfWriter()
    
    # Check if pdf_list is empty
    if not pdf_list:
        raise ValueError("PDF list is empty")
    
    # Process the first PDF to add header and footer
    first_pdf = PdfReader(pdf_list[0])
    
    # Add header and footer to the first page of the first PDF
    for i, page in enumerate(first_pdf.pages):
        if i == 0:  # Only add header/footer to the first page
            # Create a new page with header and footer
            modified_page = add_header_footer_to_page(page, header_text, footer_text)
            pdf_writer.add_page(modified_page)
        else:
            pdf_writer.add_page(page)
    
    # Add remaining pages from the first PDF
    for pdf_path in pdf_list[1:]:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)
    
    # Write the merged PDF to output file
    with open(output_path, 'wb') as output_file:
        pdf_writer.write(output_file)

def add_header_footer_to_page(page, header_text, footer_text):
    """
    Add header and footer text to a PDF page.
    
    Args:
        page (PageObject): PDF page object
        header_text (str): Header text to add
        footer_text (str): Footer text to add
    
    Returns:
        PageObject: Modified page with header and footer
    """
    # Get page dimensions
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    
    # Create header/footer overlay using reportlab
    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=(width, height))
    
    # Add header (near top)
    can.setFont("Helvetica", 12)
    can.drawString(50, height - 30, header_text)
    
    # Add footer (near bottom)
    can.setFont("Helvetica", 12)
    can.drawString(50, 30, footer_text)
    
    can.save()
    
    # Move to the beginning of the StringIO buffer
    packet.seek(0)
    
    # Create a new PDF with the header/footer
    new_pdf = PdfReader(packet)
    
    # Merge the header/footer with the original page
    page.merge_page(new_pdf.pages[0])
    
    return page

if __name__ == '__main__':
    pdf_files = [r'D:\Writing\llm_rag\word_version\Ch1 Introduction.pdf', 
                 r'D:\Writing\llm_rag\word_version\Ch2 LLMOps.pdf', 
                 r'D:\Writing\llm_rag\word_version\Ch3 RAG.pdf',
                 r'D:\Writing\llm_rag\word_version\Ch4 data process.pdf', 
                 r'D:\Writing\llm_rag\word_version\Ch5 Vector.pdf', ]
    output_file = r'D:\Writing\llm_rag\word_version\RAG_LLM_CH1-5.pdf'
    header_text = 'Draft version for Mo\'s review'
    footer_text = 'Author: Jun Xu, xujun@ieee.org'
    merge_pdfs_with_header_footer(pdf_files, output_file, header_text, footer_text)

