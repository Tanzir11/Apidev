from io import BytesIO
import fitz
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch
from .params import (num_lists, dimensions, 
                    similarity_algorithm, kind, 
                    m, ef_construction
                    )
from vendors.Openai import embeddings
from langchain_core.documents import Document
from vendors.Mongodb import collection
import requests
from bs4 import BeautifulSoup


def create_embeddings(Data):
    """
    Creates embeddings for the given data and stores them in an Azure Cosmos DB vector search index.

    Parameters:
    ---------------
    session_id : str
        A unique identifier for the current session, used as the index name in Azure Cosmos DB.
    Data : List[Document]
        A list of documents for which embeddings are to be created. Each document contains text content and associated metadata.

    Returns:
    -------
    str
        A confirmation message that the vector store has been successfully created.

    Process:
    -----
    1. Initializes an AzureCosmosDBVectorSearch instance using the provided Data and pre-configured embeddings.
    2. The session_id is used as the index name in the vector store to organize the embeddings for the specific session.
    3. Calls create_index() on the vector store to create the vector index with specific configurations, including:
        - num_lists: The number of clusters in the index.
        - dimensions: The number of dimensions of the embeddings.
        - similarity_algorithm: The algorithm used to measure similarity between vectors.
        - kind: Specifies the type of index being created.
        - m and ef_construction: Hyperparameters related to HNSW (Hierarchical Navigable Small World) indexing.
    4. Once the index is created, returns a confirmation message that the vector store has been created.
    """
    
    vectorstore = AzureCosmosDBVectorSearch.from_documents(
        Data,
        embeddings,
        collection=collection,
        index_name="lol",
    )
    
    # Create the vector search index with specific parameters
    vectorstore.create_index(
        num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
    )
    
    return "vectorstore has been created"


def extract_text_from_pdf(pdf_bytes, file):
    """
    Extracts text from a PDF file and returns it as structured data.

    Parameters:
    ----------
    pdf_bytes : bytes
        The binary content of the PDF file, which is passed as a byte stream.
    file : File
        The file object containing metadata such as the filename. Used to identify the source of the document.

    Returns:
    -------
    data : List[Document]
        A list containing a Document object with metadata (source as filename and page number) and the extracted text content from the PDF.
    
    Process:
    --------
    1. Opens the PDF byte stream using fitz.open()` (from PyMuPDF).
    2. Iterates through all pages of the PDF.
    3. Extracts the text from each page and appends it to extracted_text`.
    4. After text extraction, creates a Document` object with the extracted text and metadata such as source filename and page number.
    5. Closes the PDF document and returns the Document` object.
    """
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_text = ""
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        extracted_text += page.get_text()
    data = [Document(page_content=extracted_text, metadata={'source': f'data/{file}.pdf', 'page': 0})]
    pdf_document.close()
    
    return data


def scrape_website_to_txt(url):
    """
    Scrapes the website content using BeautifulSoup and returns it as text.

    Parameters:
    - url: str, The URL to scrape.
    
    Returns:
    - data: List[Document], scraped data with the source metadata.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        page_content = soup.find('body').get_text(separator=' ', strip=True)
        data = [Document(metadata={'source': url, 'page': 0}, page_content=page_content)]
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None
