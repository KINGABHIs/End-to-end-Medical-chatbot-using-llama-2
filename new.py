from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone, FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.llms import CTransformers
from langchain_community.docstore.in_memory import InMemoryDocstore

import pinecone
import os


# Ensure Pinecone is initialized with correct API details
PINECONE_API_KEY = '7281e437-7288-457d-9f17-2724b37a6180'
# PINECONE_API_KEY = '3383ca87-f6e3-4e50-8937-93f2211eb00f'
PINECONE_API_ENV = 'us-east-1'
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)


# Load PDF documents
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

extracted_data = load_pdf("data/")
print(extracted_data)  # Check if PDF data is loaded correctly


# Text splitting
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


text_chunks = text_split(extracted_data)
print("length of my chunk:", len(text_chunks))


# Download embeddings
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


embeddings = download_hugging_face_embeddings()


# Initialize Pinecone vector store
index_name = "mdi-chatbot"
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)


# LLM setup (ensure the correct model path and authentication)
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama", config={'max_new_tokens': 512, 'temperature': 0.8})

# Retrieval QA Chain
def retrieval_qa_chain():
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    return qa_chain

qa_chain = retrieval_qa_chain()

while True:
    user_input = input("Input Prompt:")
    result = qa_chain({"query": user_input})
    print("Response:", result["result"])
