import os

from langchain.document_loaders import PyPDFLoader, DirectoryLoader, RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document


pdf_data_path = "data/pdf"
faiss_db_path = "vectordb/python"

device = os.environ["DEVICE"]


def load_pdf_documents() -> list[Document]:
    pdf_doc_loader = DirectoryLoader(
        path=pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader
    )
    pdf_documents = pdf_doc_loader.load()
    return pdf_documents


def load_url_documents() -> list[Document]:
    urls_and_max_depths = [
        ("https://docs.python.org/3/", -1),
        ("https://www.python.org", 1),
        ("https://www.w3schools.com/python/", 1),
        ("https://en.wikipedia.org/wiki/Python_(programming_language)", 1),
        ("https://www.geeksforgeeks.org/python-programming-language/", 1),
    ]
    url_documents = []
    for url, max_depth in urls_and_max_depths:
        curr_url_docs = RecursiveUrlLoader(url=url, max_depth=max_depth).load()
        url_documents += curr_url_docs
    return url_documents


def create_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    pdf_documents = load_pdf_documents()
    url_documents = load_url_documents()

    all_documents = pdf_documents + url_documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=64, chunk_overlap=16)
    texts = text_splitter.split_documents(documents=all_documents)

    db = FAISS.from_documents(documents=texts, embedding=embeddings)
    db.save_local(folder_path=faiss_db_path)


if __name__ == "__main__":
    create_vector_db()
