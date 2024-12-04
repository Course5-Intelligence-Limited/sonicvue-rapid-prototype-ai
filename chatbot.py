# import os
# from typing import List, Dict
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain_groq import ChatGroq
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from dotenv import load_dotenv

# load_dotenv()

# class Chatbot:
#     def __init__(self, documents: List[str]):
#         self.groq_api_key = os.environ.get("GROQ_API_KEY")
#         if not self.groq_api_key:
#             raise ValueError("GROQ_API_KEY environment variable not set")

#         self.embeddings = HuggingFaceEmbeddings()
#         self.vector_store = FAISS.from_texts(documents, self.embeddings)
#         self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         self.qa_chain = self._create_qa_chain()

#     def _create_qa_chain(self):
#         llm = ChatGroq(temperature=0, groq_api_key=self.groq_api_key)
#         return ConversationalRetrievalChain.from_llm(
#             llm,
#             retriever=self.vector_store.as_retriever(),
#             memory=self.memory
#         )

#     def chat(self, user_input: str) -> str:
#         response = self.qa_chain({"question": user_input})
#         return response['answer']

#     def get_chat_history(self) -> List[Dict[str, str]]:
#         return self.memory.chat_memory.messages

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

class Chatbot:
    def __init__(self):
        self.llm = llm = ChatGroq(model = 'llama-3.1-8b-instant')
        self.vectordb = self.load_vector_base()
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})
        self.qa_chain = self.load_qa_chain()
        # self.train()
        
    def load_qa_chain(self):
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                        chain_type="stuff",
                                        retriever=self.retriever,
                                        return_source_documents=True)
        return qa_chain
    
    def load_vector_base(self):
        embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": 'cpu'},
        encode_kwargs={"normalize_embeddings": True}
        )
        #loading embeddings from local
        vectorstore = FAISS.load_local('embeddings/faiss_index_each_doc',embedding_model,allow_dangerous_deserialization= True)
        return vectorstore

    # def train(self) : 
    #     text_folder = os.path.join(os.getcwd(), 'data//extracted')
    #     for text_file in os.listdir(text_folder) : 
    #         file_path = os.path.join(text_folder, text_file)
    #         document = ""
    #         with open(file_path, 'r', encoding='utf-8') as f : 
    #             document = f.read()

    #         ## Here is the nmew embeddings being used
    #         text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=10)
    #         chunks = text_splitter.split_text(document)
    #         documents = [
    #                     {
    #                         "id": f"{text_file}_{i}",  # Unique ID for each chunk
    #                         "text": chunk,
    #                         "metadata": {
    #                             "source": os.path.splitext(os.path.basename(text_file))[0],  # Store the filename as metadata
    #                         }
    #                     }
    #                     for i, chunk in enumerate(chunks)
    #         ]

    #         for doc in documents : 
    #             self.vectordb.add_texts([doc['text']], metadatas=[doc['metadata']])
            
            # print('done')
            

    def process_llm_response(self, llm_response):
        # print(llm_response['result'])
        # print('Sources:')
        all_source = set()
        for source in llm_response["source_documents"]:
            all_source.add(source.metadata['source'])
        llm_response['source_documents'] = list(all_source)
        return llm_response

    def ask_question(self, query) : 
        llm_response = self.qa_chain(query)
        return self.process_llm_response(llm_response)
