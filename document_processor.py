import os
import tempfile
import requests
from PyPDF2 import PdfReader
from docx import Document
import csv
from bs4 import BeautifulSoup
import io

class DocumentProcessor:
    def __init__(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')

    def process_input(self, input_source):
        if isinstance(input_source, str):
            if input_source.startswith('http'):
                self._process_url(input_source)
            elif os.path.isfile(input_source):
                self._process_file(input_source)
            else:
                self._process_text(input_source)
        elif hasattr(input_source, 'read'):  # File-like object
            self._process_file_object(input_source)
        else:
            raise ValueError("Unsupported input type")

        return self.temp_file.name

    def _process_url(self, url):
        response = requests.get(url)
        if 'text/html' in response.headers.get('Content-Type', ''):
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
        else:
            text = response.text
        self._write_to_temp_file(text)

    def _process_file(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        if file_extension == '.pdf':
            self._process_pdf(file_path)
        elif file_extension == '.docx':
            self._process_docx(file_path)
        elif file_extension == '.csv':
            self._process_csv(file_path)
        elif file_extension == '.txt':
            with open(file_path, 'r') as file:
                self._write_to_temp_file(file.read())
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _process_file_object(self, file_object):
        # Try to get the file name or extension
        file_name = getattr(file_object, 'name', '')
        _, file_extension = os.path.splitext(file_name)
        
        if file_extension.lower() == '.pdf':
            self._process_pdf_object(file_object)
        elif file_extension.lower() == '.csv':
            content = file_object.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            self._process_csv_content(content)
        else:
            # For other file types, try to read as text
            try:
                content = file_object.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                self._write_to_temp_file(content)
            except UnicodeDecodeError:
                raise ValueError(f"Unsupported file type or encoding: {file_extension}")

    def _process_pdf_object(self, file_object):
        pdf_reader = PdfReader(file_object)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        self._write_to_temp_file(text)

    def _process_pdf(self, file_path):
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        self._write_to_temp_file(text)

    def _process_docx(self, file_path):
        doc = Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        self._write_to_temp_file(text)

    def _process_csv(self, file_path):
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            text = '\n'.join([','.join(row) for row in csv_reader])
        self._write_to_temp_file(text)

    def _process_csv_content(self, content):
        csv_reader = csv.reader(io.StringIO(content))
        text = '\n'.join([','.join(row) for row in csv_reader])
        self._write_to_temp_file(text)

    def _process_text(self, text):
        self._write_to_temp_file(text)

    def _write_to_temp_file(self, text):
        self.temp_file.write(text)
        self.temp_file.flush()

    def __del__(self):
        self.temp_file.close()
        os.unlink(self.temp_file.name)