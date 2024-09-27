from .utils import create_embeddings, scrape_website_to_txt, extract_text_from_pdf
import re


class Scrapper:
    """
    A class used to perform scraping tasks and generate embeddings from either a website or a PDF file.

    Methods
    -------
    embedding_generator(input_str, file=None, session_id):
        Generates embeddings from the content of a URL or a PDF file.
    """
    @staticmethod
    def embedding_generator_url(input_str):
        """
        Generates embeddings based on input URL or file content.

        Parameters:
        ----------
        input_str : str
            The input string that could be a URL.
        session_id : str
            A session ID for creating the embedding index.
        file : Optional
            If provided, the file content will be used to generate embeddings.

        Returns:
        -------
        str
            A success message or an error message if the input is not a valid URL.
        """
        data = scrape_website_to_txt(input_str)
        data_push = create_embeddings(data)
        return data_push

    
    @staticmethod
    def embedding_generator_pdf(input_str, file):
        if file.endswith('.pdf'):
            data = extract_text_from_pdf(input_str, file)
            # print(type(data))
            data_push = create_embeddings(data)
        else:
            data_push = "File not a pdf"
        
        return data_push



Scrapper_obj = Scrapper()
