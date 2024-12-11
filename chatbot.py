from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from warnings import filterwarnings

load_dotenv()
filterwarnings('ignore')
groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

class Chatbot:
    def __init__(self):
        self.llm = ChatGroq(model='llama-3.1-8b-instant', api_key=groq_api_key)
        self.vectordb = self.load_vector_base()
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})
        self.qa_chain = self.load_qa_chain()
        
    def load_qa_chain(self):
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
            return qa_chain
        except Exception as e:
            print(f"Error creating QA chain: {str(e)}")
            raise
    
    def load_vector_base(self):
        try:
            embedding_model = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": 'cpu'},
                encode_kwargs={"normalize_embeddings": True}
            )
            vectorstore = FAISS.load_local(
                'embeddings/faiss_index_each_doc',
                embedding_model,
                allow_dangerous_deserialization=True
            )
            return vectorstore
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            raise

    def process_llm_response(self, llm_response):
        try:
            all_source = set()
            for source in llm_response["source_documents"]:
                all_source.add(source.metadata['source'])
            llm_response['source_documents'] = list(all_source)
            return llm_response
        except Exception as e:
            print(f"Error processing LLM response: {str(e)}")
            raise

    def ask_question(self, query): 
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            llm_response = self.qa_chain(query)
            print(llm_response['result'])
            return llm_response
            # return self.process_llm_response(llm_response)
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            raise

# Commented out training code preserved for reference
# def train(self):
#     text_folder = os.path.join(os.getcwd(), 'data/extracted')
#     for text_file in os.listdir(text_folder):
#         file_path = os.path.join(text_folder, text_file)
#         document = ""
#         with open(file_path, 'r', encoding='utf-8') as f:
#             document = f.read()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=10)
#         chunks = text_splitter.split_text(document)
#         documents = [
#             {
#                 "id": f"{text_file}_{i}",
#                 "text": chunk,
#                 "metadata": {
#                     "source": os.path.splitext(os.path.basename(text_file))[0],
#                 }
#             }
#             for i, chunk in enumerate(chunks)
#         ]
#         for doc in documents:
#             self.vectordb.add_texts([doc['text']], metadatas=[doc['metadata']])