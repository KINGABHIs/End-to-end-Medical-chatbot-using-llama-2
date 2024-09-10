from flask import Flask, render_template, jsonify, request
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
from src.prompt import *



app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = download_hugging_face_embeddings()


pc = PP(api_key = os.environ.get('PINECONE_API_KEY'))
index_name ="mdi-chatbot"
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"]
)
chain_type_kwargs={"prompt": PROMPT}
print(chain_type_kwargs)

llm = CTransformers(model="D:\projects\End-to-end-Medical-chatbot-using-llama-2\model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens':512,
                            'temperature':0.8})


qa = RetrievalQA.from_chain_type(  
    llm=llm,  
    chain_type="stuff",  
    retriever=docsearch.as_retriever(search_kwargs ={'k':2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs 
) 


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    query = request.form["query"]
    print(query)
    result = qa.invoke({"query": query})
    print("Response:", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(debug = True)