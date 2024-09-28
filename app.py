import os
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_PATHS = ["docs/ProjectMentorshipDAVV.pdf","docs/Taiyo.AI - Data Engineering (Web Scraping)Trial Task.pdf"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = 'models/embedding-001'
GENERATIVE_MODEL = 'gemini-pro'

# Set up Gemini API
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it before running the script.")

genai.configure(api_key=GOOGLE_API_KEY)

class PDFLoader:
    @staticmethod
    def load_pdfs(pdf_paths: List[str]) -> List[Document]:
        documents = []
        for path in pdf_paths:
            try:
                loader = PyPDFLoader(path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading PDF {path}: {e}")
        if not documents:
            raise ValueError("No documents were successfully loaded. Please check your PDF paths and file permissions.")
        return documents

class DocumentSplitter:
    @staticmethod
    def split_documents(documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return text_splitter.split_documents(documents)

class VectorStore:
    @staticmethod
    def create(documents: List[Document]) -> FAISS:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.from_documents(documents, embeddings)

    @staticmethod
    def query(vector_store: FAISS, query: str, k: int = 4) -> List[Document]:
        return vector_store.similarity_search(query, k=k)

class ResponseGenerator:
    @staticmethod
    def generate(query: str, context: str) -> str:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = llm.invoke(prompt)
        return response.content

class ChatBot:
    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    def get_general_response(self, query: str) -> str:
        prompt = f"Answer the following question to the best of your ability. If you're not sure or the question requires specific information not provided, respond with 'I need more context to answer that accurately.'\n\nQuestion: {query}\n\nAnswer:"
        response = self.llm.invoke(prompt)
        return response.content

    def get_pdf_specific_response(self, query: str) -> str:
        relevant_docs = VectorStore.query(self.vector_store, query)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = self.llm.invoke(prompt)
        return response.content

    def chat_loop(self):
        print("Chat with your AI assistant! (Type 'exit' to quit)")
        while True:
            query = input("You: ")
            if query.lower() == 'exit':
                break
            
            try:
                general_response = self.get_general_response(query)
                if "I need more context" in general_response:
                    response = self.get_pdf_specific_response(query)
                else:
                    response = general_response
                print("AI:", response)
            except Exception as e:
                print(f"An error occurred: {e}")

def main():
    try:
        # Load and process documents
        documents = PDFLoader.load_pdfs(PDF_PATHS)
        if not documents:
            raise ValueError("No documents were loaded. Please check your PDF paths and file permissions.")
        
        split_docs = DocumentSplitter.split_documents(documents)
        
        # Create vector store
        vector_store = VectorStore.create(split_docs)
        
        # Start chat bot
        chat_bot = ChatBot(vector_store)
        chat_bot.chat_loop()
    except Exception as e:
        print(f"An error occurred during setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()