import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient


load_dotenv()
connection_string = os.getenv('connection_string')
container_name = 'sonicvue-rapid-prototype'
folder_path = 'raw-audios/'

raw_data_folder = os.path.join(os.getcwd(), 'data//raw_data')
os.makedirs(raw_data_folder, exist_ok=True)

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)


blobs = container_client.list_blobs(name_starts_with=folder_path)
for blob in blobs : 
    download_file_path = os.path.join(raw_data_folder, blob.name[len(folder_path):])
    if not os.path.exists(download_file_path) : 
        with open(download_file_path, 'wb') as download_file : 
            download_file.write(container_client.download_blob(blob.name).readall())

# with open()
# print(blobs)
