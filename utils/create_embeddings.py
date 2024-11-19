from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.document_loaders import AzureBlobStorageFileLoader

# Connection details for Azure Blob Storage
connection_string = os.getenv('connection_string')
container_name = "sonicvue-rapid-prototype"
blob_list = ["new/audio__1.txt", "new/audio__2.txt"]  # List of blobs to process
data_folder = './data/extracted'  # Local directory to load files from

# This will store all document chunks
all_splits = []

# Process local files
def process_local_files():
    global all_splits
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                docs = file.read()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2200,
                chunk_overlap=110
            )
            chunks_list = text_splitter.create_documents([docs])
            all_splits.extend(chunks_list)
            print(f'Local file {filename} processed.')

# Process Azure Blob Storage files
def process_blob_files():
    global all_splits
    for blob_name in blob_list:
        loader = AzureBlobStorageFileLoader(conn_str=connection_string, container=container_name, blob_name=blob_name)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2200,
            chunk_overlap=110
        )
        chunks_list = text_splitter.split_documents(docs)
        all_splits.extend(chunks_list)
        print(f'Blob file {blob_name} processed.')

# First process local files
process_local_files()

# Then process blob files
# process_blob_files()

# Set up embedding model
DEVICE = 'cpu'
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embedding_model)

# Save the FAISS index locally
vectorstore.save_local('embeddings/faiss_index_each_doc')

print('Embedding and vector store creation complete.')
