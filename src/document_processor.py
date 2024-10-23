import os
import tempfile
import requests
from typing import Union, Optional
from PyPDF2 import PdfReader
from docx import Document
import csv
from bs4 import BeautifulSoup
import logging
import io

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
        self.supported_extensions = {
            '.txt': self._process_text_file,
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.csv': self._process_csv
        }

    def process_input(self, input_source: Union[str, bytes, io.IOBase]) -> str:
        """Process different types of input sources and return path to processed text."""
        try:
            if isinstance(input_source, str):
                if input_source.startswith('http'):
                    self._process_url(input_source)
                elif os.path.isfile(input_source):
                    self._process_file(input_source)
                else:
                    self._process_text(input_source)
            elif hasattr(input_source, 'read'):
                self._process_file_object(input_source)
            else:
                raise ValueError("Unsupported input type")

            return self.temp_file.name
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}", exc_info=True)
            raise

    def _process_url(self, url: str):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' in content_type:
                soup = BeautifulSoup(response.content, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text(separator='\n', strip=True)
            else:
                text = response.text
                
            self._write_to_temp_file(text)
            
        except requests.RequestException as e:
            logger.error(f"Error fetching URL: {str(e)}")
            raise

    def _process_file(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {ext}")
        
        self.supported_extensions[ext](file_path)

    def _process_file_object(self, file_object: io.IOBase):
        file_name = getattr(file_object, 'name', '')
        ext = os.path.splitext(file_name)[1].lower()
        
        try:
            if ext == '.pdf':
                self._process_pdf_object(file_object)
            else:
                content = file_object.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                    
                if ext == '.csv':
                    self._process_csv_content(content)
                else:
                    self._write_to_temp_file(content)
                    
        except UnicodeDecodeError:
            raise ValueError(f"Unsupported file encoding for {ext}")

    def _process_text_file(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            self._write_to_temp_file(file.read())

    def _process_pdf(self, file_path: str):
        with open(file_path, 'rb') as file:
            self._process_pdf_object(file)

    def _process_pdf_object(self, file_object: io.IOBase):
        pdf_reader = PdfReader(file_object)
        text = '\n'.join(page.extract_text() for page in pdf_reader.pages)
        self._write_to_temp_file(text)

    def _process_docx(self, file_path: str):
        doc = Document(file_path)
        text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
        self._write_to_temp_file(text)

    def _process_csv(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            self._process_csv_content(file.read())

    def _process_csv_content(self, content: str):
        csv_reader = csv.reader(io.StringIO(content))
        text = '\n'.join(','.join(row) for row in csv_reader)
        self._write_to_temp_file(text)

    def _process_text(self, text: str):
        self._write_to_temp_file(text)

    def _write_to_temp_file(self, text: str):
        self.temp_file.write(text)
        self.temp_file.flush()

    def __del__(self):
        try:
            self.temp_file.close()
            if os.path.exists(self.temp_file.name):
                os.unlink(self.temp_file.name)
        except Exception as e:
            logger.error(f"Error cleaning up temporary file: {str(e)}")