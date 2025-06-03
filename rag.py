from langgraph.graph import StateGraph , END
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest" , temperature=0.1, api_key=api_key )


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

pdf_path = "Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path) : 
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try : 
    pages = pdf_loader.load()
    print(f"PDF loaded with {len(pages)} pages.")
except Exception as e :
    print(f"Error loading PDF: {str(e)}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
)

persist_directory = r"C:\Users\FRANS\PycharmProjects\langgraph"
collection_name = "stock_market"

if not os.path.exists(persist_directory) : 
    os.makedirs(persist_directory)