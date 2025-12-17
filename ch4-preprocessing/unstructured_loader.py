
from langchain_community.document_loaders import UnstructuredPDFLoader
def load_pdf_with_tables(pdf_path):
    try:
        loader = UnstructuredPDFLoader(
            pdf_path,
            mode="hi_res",
            strategy="hi_res",
            infer_table_structure=True
        )
        documents = loader.load()
        
        # Optional: keep the print statements for debugging
        for doc in documents:
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            
        return documents  # Return for further processing
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

if __name__ == "__main__":
    pdf_path = "path/to/your/pdf.pdf"
    documents = load_pdf_with_tables(pdf_path)
    print(f"Loaded {len(documents)} documents.")    
