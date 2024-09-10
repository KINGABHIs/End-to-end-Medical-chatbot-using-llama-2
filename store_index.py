from src.helper import load_pdf, text_split, download_hugging_face_embeddings


from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PP
import os
import pinecone
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_pinecone import PineconeVectorStore 
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

print(PINECONE_API_KEY)
print(PINECONE_API_ENV)

extracted_data = load_pdf("D:\projects\End-to-end-Medical-chatbot-using-llama-2\data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

from langchain_pinecone import PineconeVectorStore 


pc = PP(api_key = os.environ.get('PINECONE_API_KEY'))
index_name ="mdi-chatbot"
docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
print("Successfully data loaded.....")