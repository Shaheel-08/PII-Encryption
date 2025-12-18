import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import PyPDF2
import re
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# Set Tesseract path (default installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Define the RAGPipeline class
class RAGPipeline:
    def __init__(self, file_path, model_name='all-MiniLM-L6-v2', chunk_size=500):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.model = SentenceTransformer(model_name)
        self.text = self.extract_text_from_file()
        self.chunks = self.preprocess_text(self.text)
        self.chunk_embeddings = self.encode_chunks(self.chunks)

    def extract_text_from_file(self):
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        # Handle PDF files
        if file_ext == '.pdf':
            return self.extract_text_from_pdf()
        
        # Handle text files
        elif file_ext == '.txt':
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        
        # Handle image files
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            try:
                image = Image.open(self.file_path)
                text = pytesseract.image_to_string(image)
                if not text.strip():
                    raise ValueError("No text found in image")
                return text
            except Exception as e:
                raise ValueError(f"Could not extract text from image: {str(e)}")
        
        # Handle Word documents
        elif file_ext in ['.docx', '.doc']:
            try:
                import docx
                doc = docx.Document(self.file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                if not text.strip():
                    raise ValueError("Document contains no text")
                return text
            except ImportError:
                raise ValueError("python-docx library not installed. Run: pip install python-docx")
            except Exception as e:
                raise ValueError(f"Could not extract text from document: {str(e)}")
        
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def extract_text_from_pdf(self):
        # Try regular text extraction first
        with open(self.file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        
        # If no text found, try OCR
        if not text.strip():
            try:
                images = convert_from_path(self.file_path)
                text = ""
                for image in images:
                    text += pytesseract.image_to_string(image) + "\n"
                
                if not text.strip():
                    raise ValueError("PDF contains no extractable text even with OCR")
            except Exception as e:
                raise ValueError(f"Could not extract text from PDF: {str(e)}")
        
        return text

    def preprocess_text(self, text):
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)

        processed_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return processed_chunks

    def encode_chunks(self, chunks):
        return self.model.encode(chunks, convert_to_tensor=True)

    def retrieve_relevant_chunks(self, query, top_k=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, self.chunk_embeddings)[0]

        # Ensure top_k does not exceed the number of chunks
        top_k = min(top_k, len(self.chunks))

        top_results = scores.topk(top_k)  # Retrieve top k results
        return [self.chunks[idx] for idx in top_results.indices]



    def find_pii_information(self, query):
        # Search in the entire text instead of just retrieved chunks
        text = ' '.join(self.chunks)
        
        pii_info = {}
        
        # Aadhaar regex pattern
        aadhaar_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
        aadhaar_matches = re.findall(aadhaar_pattern, text)
        if aadhaar_matches:
            pii_info['Aadhaar'] = aadhaar_matches
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        if email_matches:
            pii_info['Email Address'] = email_matches
        
        # Phone pattern (Indian format)
        phone_pattern = r'\b(?:\+91|91)?[-.\s]?[6-9]\d{9}\b'
        phone_matches = re.findall(phone_pattern, text)
        if phone_matches:
            pii_info['Phone number'] = phone_matches
        
        # PAN card pattern
        pan_pattern = r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'
        pan_matches = re.findall(pan_pattern, text)
        if pan_matches:
            pii_info['PAN card'] = pan_matches
        
        # Check for other PII keywords in text
        for q in query:
            if q not in pii_info and q.lower() in text.lower():
                pii_info[q] = "Found in document"
        
        if len(pii_info) == 0:
            return "There is no PII information present in your PDF."
        else:
            return f"PII information found: {pii_info}"
st.set_page_config(page_title="PDF Upload", page_icon="📄", layout="centered")
st.markdown("""
    <style>
    body {
        background-color: #000;  /* Black background */
        color: #fff;  /* White text */
        font-family: 'Arial', sans-serif;
    }
    .drag-drop-area {
        border: 3px dashed #1e3a8a;
        padding: 50px;
        text-align: center;
        color: #1e3a8a;
        background-color: #111;
        border-radius: 15px;
        transition: background-color 0.3s ease;
    }
    .drag-drop-area:hover {
        background-color: #222;  /* Slightly lighter black */
    }
    .upload-button {
        background-color: #1e3a8a;
        color: #fff;
        border: none;
        padding: 12px 24px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        margin-top: 20px;
    }
    .upload-button:hover {
        background-color: #3b82f6;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #111;
        color: #888;
        text-align: center;
        padding: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📄 Upload your document to find PII")
st.markdown("*Drag and drop your files here, or click to upload. Supports PDF, DOCX, TXT, and images.*")

# File uploader widget - accepts multiple file types
uploaded_file = st.file_uploader("", type=["pdf", "txt", "docx", "doc", "png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    st.success('File uploaded successfully!')

    # Create "tmp" directory if it doesn't exist
    tmp_dir = "tmp"
    try:
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            print(f'Directory "{tmp_dir}" created.')
        else:
            print(f'Directory "{tmp_dir}" already exists.')

        # Save uploaded file to the "tmp" directory
        file_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    # Initialize the RAGPipeline with the uploaded file
    try:
        pipeline = RAGPipeline(file_path)

        # Define your queries
        query = ["Aadhaar", "Passport number", "Account Number ", "Driving License Number", "PAN card", "Application Number", "Email Address", "Phone number","Biomtric data","IP address","Voter identity","Date of birth"]

        # Retrieve and display the PII information
        pii_info = pipeline.find_pii_information(query)
        st.write(pii_info)

        # Determine MIME type based on file extension
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        mime_types = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        mime = mime_types.get(file_ext, 'application/octet-stream')
        
        st.download_button(label=f"Download {uploaded_file.name}", data=uploaded_file, file_name=uploaded_file.name, mime=mime, key="download_button")
    
    except ValueError as ve:
        st.error(f"⚠️ {str(ve)}")
        st.info("Please make sure the file contains readable text or install required dependencies.")
    except Exception as e:
        st.error(f"An error occurred while processing: {str(e)}")

# Optional footer
# st.markdown('<div class="footer">© 2024 Your Company. All rights reserved.</div>', unsafe_allow_html=True)

