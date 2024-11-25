import os, asyncio, json, shutil
from fastapi import File, UploadFile, HTTPException, BackgroundTasks, APIRouter
from azure.storage.blob import BlobServiceClient
from pydantic import BaseModel
from chatbot import Chatbot
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
from utils.create_transcripts import get_transcripts
# from utils.notused.process_metadata import process_metadata_for_file
from utils.feature_extraction import extract_features

router = APIRouter()

# Azure Blob Storage configuration
connection_string = os.getenv('connection_string')
container_name = "sonicvue-rapid-prototype"
groq_api_key = os.getenv('GROQ_API_KEY')

# Ensure the local data directory exists
local_data_dir = "data/raw"
local_transcripts_dir = "data//extracted"
local_metadata_dir = "metadata"
for dir_path in [local_data_dir, local_metadata_dir, local_transcripts_dir] : 
    os.makedirs(dir_path, exist_ok=True)

chatbot = Chatbot()

# processing_status: Dict[str, Dict[str, str]] = {}
latest_uploaded_files: List[str] = []

latest_uploaded_files = ['audio__1.wav', 'audio__2.wav', 'audio__3.wav', 'audio__5.wav']

def get_dashboard_dataframe() -> pd.DataFrame:
    """Convert JSON metadata files to a DataFrame"""
    data_list = []
     
    #debug
    print(latest_uploaded_files)
    
    for filename in latest_uploaded_files:
        json_path = os.path.join(local_metadata_dir, f"{filename[:-4]}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                data['filename'] = filename
                data_list.append(data)
    
    if not data_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(data_list)
    
    # Convert time columns to numeric
    time_columns = ['call_time', 'hold_time', 'route_time', 'resolution_time']
    for col in time_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def calculate_dashboard_metrics(df: pd.DataFrame) -> Dict:
    """Calculate metrics for dashboard"""
    if df.empty:
        return {}

    total_calls = len(df)
    
    metrics = {
        "summary": {
            "total_calls": total_calls,
            "call_routing_accuracy": (df['route_time'] == 0).mean() * 100,
            "multiple_agents": (df['multiple_agents'] == "Yes").mean() * 100,
            "call_hold_percentage": (df['hold'] == "Yes").mean() * 100,
            "escalated_calls": (df['escalation'] == "Yes").mean() * 100,
            "resolution_confirmation": (df['resolution_confirmation'] == "Yes").mean() * 100,
            "cs_portal_recommended": (df['digital_service'] == "Yes").mean() * 100
        },
        "calls_complexity": {
            "easy": len(df[df['complexity'] == "Easy"]),
            "medium": len(df[df['complexity'] == "Medium"]),
            "difficult": len(df[df['complexity'] == "Difficult"])
        },
        "call_hygiene": {
            "greeting": (df['greeting'] == "Yes").mean() * 100,
            "phone_number": (df['phone_number'] == "Yes").mean() * 100,
            "email": (df['email_address'] == "Yes").mean() * 100
        },
        "tone_conversation": {
            "positive": len(df[df['call_tone'] == "Positive"]),
            "neutral": len(df[df['call_tone'] == "Neutral"]),
            "negative": len(df[df['call_tone'] == "Negative"])
        },
        "event_type": df['call_type'].value_counts().to_dict(),
        "customer_service": {
            "parts_request": (df['part_request'] == "Yes").mean() * 100,
            "digital_service": (df['digital_service'] == "Yes").mean() * 100,
            "field_visits": (df['field_service'] == "Yes").mean() * 100
        }
    }
    
    return metrics

task_status = {}  # Store task status as {filename: status}

# Background task to generate transcripts
async def get_transcript_func(file_path: str, filename: str, background_tasks: BackgroundTasks = BackgroundTasks()):
    # Set status to 'processing' when starting the task
    task_status[filename] = "processing"
    try:
        # Call your existing transcript generation function
        result = get_transcripts(file_path)
        text_file_path = os.path.join(local_transcripts_dir, f'{filename[:-4]}.txt')
        background_tasks.add_task(extract_features, text_file_path)
        task_status[filename] = "completed"
    except Exception as e:
        task_status[filename] = "failed"
        print(f"Error generating transcript for {filename}: {e}")

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    # Create local directories if they don't exist
    raw_dir = os.path.join(os.getcwd(), 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    results = []
    latest_uploaded_files.clear()
    for file in files:
        # Save to local directory
        file_path = os.path.join(raw_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Upload to blob storage
        # blob_client = container_client.get_blob_client(f"raw_data/{file.filename}")
        # with open(file_path, "rb") as data:
        #     blob_client.upload_blob(data, overwrite=True)
        
        task_status[file.filename] = "queued"

        latest_uploaded_files.append(file.filename)
        print(f'{file.filename} added to latest_upload_files')

        background_tasks.add_task(get_transcript_func, file_path, file.filename)
        # background_tasks.add_task(extract_features, file_path)
            
        results.append({
            "filename": file.filename,
            "status": "uploaded"
        })
    
    return results

@router.get("/transcripts-status/{filename}")
async def get_transcript_status(filename: str):
    # Get the current status of the transcript task for the given file
    status = task_status.get(filename, "not found")
    return {"filename": filename, "status": status}

@router.get("/generate_transcripts")
async def generate_transcripts(files: str, background_tasks: BackgroundTasks = BackgroundTasks()):
    # Convert comma-separated string to list
    file_list = files.split(',')
    results = []
    
    for filename in file_list:
        file_path = os.path.join(os.getcwd(), 'data', 'raw', filename)
        transcript_filename = f"{os.path.splitext(filename)[0]}.txt"
        transcript_path = os.path.join(os.getcwd(), 'data', 'extracted', transcript_filename)
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        
        # Initialize status and extracted text
        if os.path.exists(transcript_path):
            # If the transcript already exists, read it
            with open(transcript_path, 'r', encoding='utf-8') as txt_file:
                extracted_text = txt_file.read()
            status = "completed"
        else:
            # If no transcript file exists, start processing it in the background
            background_tasks.add_task(get_transcripts, file_path, filename)
            extracted_text = ""  # No transcript yet
            status = "processing"

        # Add metadata extraction task if transcript is already completed
        if status == "completed":
            # Assuming extract_features is another background task
            background_tasks.add_task(extract_features, transcript_path)
        
        # Store the result for this file
        results.append({
            "filename": filename,
            "transcript": extracted_text if status == "completed" else None,
            "transcript_file": transcript_filename,
            "status": status
        })
    
    return results

# async def update_status_on_completion(filename: str, status: str):
#     await asyncio.sleep(1)  # Simulate some async task delay
#     task_status[filename] = status  # Update status

class ChatInput(BaseModel):
    message: str

@router.post("/chat")
async def chat(chat_input: ChatInput):
    try:
        results = chatbot.ask_question(chat_input.message)
        if results:
            response_content = results['result']
        else :
            response_content = "No results found."
        return {"response": response_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @router.get("/processing-status/{file_name}")
# async def get_processing_status(file_name: str):
#     if file_name not in processing_status:
#         return {"status": "not_found"}
#     return processing_status[file_name]

@router.get("/latest-uploads")
async def get_latest_uploads():
    return {"files": latest_uploaded_files}

@router.get("/dashboard-data")
async def get_dashboard_data():
    df = get_dashboard_dataframe()
    metrics = calculate_dashboard_metrics(df)
    return metrics

@router.get("/raw-data")
async def get_raw_data():
    df = get_dashboard_dataframe()
    return df.to_dict(orient='records')

class TranscriptRequest(BaseModel):
    files: List[str]



# @router.get("/generate_transcripts")
# async def generate_transcripts(files: str, background_tasks: BackgroundTasks = BackgroundTasks()):
#     # Convert comma-separated string to list
#     file_list = files.split(',')
#     results = []
    
#     for filename in file_list:
#         # print(filename)
#         file_path = os.path.join(os.getcwd(), 'data', 'raw', filename)
#         transcript_filename = f"{os.path.splitext(filename)[0]}.txt"
#         transcript_path = os.path.join(os.getcwd(), 'data', 'extracted', transcript_filename)
#         os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        
#         if os.path.exists(transcript_path):
#             # If transcript already exists, read it
#             with open(transcript_path, 'r', encoding='utf-8') as txt_file:
#                 extracted_text = txt_file.read()
#             status = "completed"
#         else:
#             # Add transcript generation to background tasks
#             background_tasks.add_task(
#                 get_transcripts,
#                 file_path
#             )
#             extracted_text = ""
#             status = "processing"
#             # task_status[filename] = status
#         # Add metadata processing to background tasks
#         if status == "completed":
#             background_tasks.add_task(
#                 extract_features,
#                 filename
#             )
            
#         results.append({
#             "filename": filename,
#             "transcript": extracted_text if status == "completed" else None,
#             "transcript_file": transcript_filename,
#             "status": status
#         })
    
#     return results