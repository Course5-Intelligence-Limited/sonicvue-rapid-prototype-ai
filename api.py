import os, asyncio, json, shutil
from fastapi import File, UploadFile, HTTPException, BackgroundTasks, APIRouter
from azure.storage.blob import BlobServiceClient
from pydantic import BaseModel
from chatbot import Chatbot
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
from utils.create_transcripts import get_transcripts
from utils.feature_extraction import extract_features
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import json
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import AzureBlobStorageFileLoader

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(model='llama-3.1-8b-instant', api_key=groq_api_key, temperature=0, seed=42)
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

# chatbot = Chatbot()

latest_uploaded_files: List[str] = ['final_record_1.mp3', 'final_record_2.mp3', 'final_record_3.mp3', 'final_record_4.mp3', 'final_record_5.mp3']
call_type_analysis = []

from fastapi import APIRouter, Query
import pandas as pd
from typing import Dict

router = APIRouter()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_dashboard_dataframe(complexity: str = 'All', event_type: str = 'All') -> pd.DataFrame:
    """
    Convert JSON metadata files to a DataFrame and filter by complexity and event type.
    
    Args:
        complexity (str): Filter for call complexity level ('All', 'Easy', 'Intermediate', 'Difficult')
        event_type (str): Filter for type of call event ('All' or specific event types)
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing call metadata
        
    Raises:
        Exception: If there are issues reading JSON files or processing data
    """
    data_list = []
    
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

    # Apply filters if not 'All'
    if complexity != 'All':
        df = df[df['complexity'] == complexity]
    
    if event_type != 'All':
        df = df[df['call_type'] == event_type]
    
    return df

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=40))
def calculate_dashboard_metrics(df: pd.DataFrame) -> Dict:
    """Calculate metrics for dashboard"""
    if df.empty:
        return {}
 
    total_calls = len(df)
    total_time = 0
    for index, rows in df.iterrows():
        total_time += rows['hold_time'] + rows['route_time'] + rows['resolution_time']
 
    metrics = {
        "summary": {
            "total_calls": total_calls,
            "call_routing_accuracy": (df['route_time'] != 0).mean() * 100,
            "multiple_agents": (df['multiple_agents'] == "Yes").mean() * 100,
            "call_hold_percentage": (df['hold'] == "Yes").mean() * 100,
            "escalated_calls": (df['escalation'] == "Yes").mean() * 100,
            "resolution_confirmation": (df['resolution_confirmation'] == "Yes").mean() * 100,
            "cs_portal_recommended": (df['digital_service'] == "Yes").mean() * 100
        },
        "complexity": {
            "easy": len(df[df['complexity'] == "Easy"]),
            "intermediate": len(df[df['complexity'] == "Intermediate"]),
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
        },
        "root_cause_analysis": {
            "hold_time": (df['hold_time']).sum()/total_time * 100,
            "resolution_time": (df['resolution_time']).sum()/total_time * 100,
            "route_time": (df['route_time']).sum()/total_time * 100,
        }
    }
    print('no issues here')
    return metrics

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=40))
# def calculate_dashboard_metrics(df: pd.DataFrame) -> Dict:
    
#     """
#     Calculate various metrics for the dashboard based on call data.
    
#     Args:
#         df (pd.DataFrame): DataFrame containing call metadata
        
#     Returns:
#         Dict: Dictionary containing calculated metrics across different categories:
#             - summary stats
#             - complexity distribution
#             - call hygiene metrics
#             - conversation tone analysis
#             - event type distribution
#             - customer service metrics
#             - root cause analysis
            
#     Raises:
#         Exception: If there are calculation errors
#     """
#     if df.empty:
#         return {}

#     total_calls = len(df)
#     total_time = 0
#     for index, rows in df.iterrows():
#         total_time += rows['hold_time'] + rows['route_time'] + rows['resolution_time']
#     print('starting_metrics')
#     metrics = {
#         "summary": {
#             "total_calls": total_calls,
#             "call_routing_accuracy": (df['route_time'] != 0).mean() * 100,
#             "multiple_agents": (df['multiple_agents'].lower() == "yes").mean() * 100,
#             "call_hold_percentage": (df['hold'] == "Yes").mean() * 100,
#             "escalated_calls": (df['escalation'] == "Yes").mean() * 100,
#             "resolution_confirmation": (df['resolution_confirmation'] == "Yes").mean() * 100,
#             "cs_portal_recommended": (df['digital_service'] == "Yes").mean() * 100,
#             "csat" : df['csat'].mean() * 25,
#         },
#         "complexity": {
#             "easy": len(df[df['complexity'] == "Easy"]),
#             "intermediate": len(df[df['complexity'] == "Intermediate"]),
#             "difficult": len(df[df['complexity'] == "Difficult"])
#         },
#         "call_hygiene": {
#             "greeting": (df['greeting'] == "Yes").mean() * 100,
#             "phone_number": (df['phone_number'] == "Yes").mean() * 100,
#             "email": (df['email_address'] == "Yes").mean() * 100
#         },
#         "tone_conversation": {
#             "positive": len(df[df['call_tone'] == "Positive"]),
#             "neutral": len(df[df['call_tone'] == "Neutral"]),
#             "negative": len(df[df['call_tone'] == "Negative"])
#         },
#         "event_type": df['call_type'].value_counts().to_dict(),
#         "customer_service": {
#             "parts_request": (df['part_request'] == "Yes").mean() * 100,
#             "digital_service": (df['digital_service'] == "Yes").mean() * 100,
#             "field_visits": (df['field_service'] == "Yes").mean() * 100
#         },
#         "root_cause_analysis": {
#             "hold_time": (df['hold_time']).mean(),
#             "resolution_time": (df['resolution_time']).mean(),
#             "route_time": (df['route_time']).mean(),
#         }
#     }
#     print('no issues here')
#     return metrics

@router.get("/dashboard-data")
async def get_dashboard_data(complexity: str = Query('All', alias='complexity', description="Filter by complexity (All, Easy, Intermediate, Difficult)"),
                             event_type: str = Query('All', alias='eventType', description="Filter by event type (All, installation, proactive, etc.)")):
    """
    Endpoint to get dashboard data with optional filters.
    
    Args:
        complexity (str): Filter for call complexity
        event_type (str): Filter for call event type
        
    Returns:
        Dict: Calculated metrics for the dashboard
        
    Raises:
        HTTPException: If there are errors processing the request
    """
    try:
        df = get_dashboard_dataframe(complexity, event_type)
        metrics = calculate_dashboard_metrics(df)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_table_dataframe() -> pd.DataFrame:
    """
    Convert metadata JSON files to a DataFrame for table view with specific KPIs.
    
    Returns:
        pd.DataFrame: DataFrame containing KPI data for table display
        
    Raises:
        Exception: If there are issues reading or processing the data
    """
    data_list = []
    
    kpi_columns = [
        'call_time', 'hold_time', 'route_time', 'resolution_time', 
        'greeting', 'phone_number', 'email_address', 'call_quality',
        'case_type', 'resolution_confirmation', 'hold', 'hold_satisfaction',
        'multiple_agents', 'escalation', 'call_tone', 'issue_discussed',
        'complexity', 'issue_type', 'status_query', 'call_type',
        'replacement_required', 'part_request', 'parts_dispatch',
        'refund_required', 'field_service', 'digital_service'
    ]
    
    for filename in latest_uploaded_files:
        json_path = os.path.join(local_metadata_dir, f"{filename[:-4]}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                filtered_data = {k: data.get(k, '') for k in kpi_columns}
                filtered_data['filename'] = filename
                data_list.append(filtered_data)
                data_list.append({
                    'call_type': data.get('call_type', 'Other'),
                    'issue_discussed': data.get('issue_discussed', '')
                })
    
    if not data_list:
        return pd.DataFrame()
        
    df = pd.DataFrame(data_list)

    for call_type, group in df.groupby('call_type'):
        call_type_analysis.append({
            'call_type': call_type,
            'total_calls': len(group),
            'issues': group['issue_discussed'].tolist()
        })
    
    time_columns = ['call_time', 'hold_time', 'route_time', 'resolution_time']
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

task_status = {}  # Store task status as {filename: status}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_transcript_func(file_path: str, filename: str, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Background task to generate transcripts from audio files.
    
    Args:
        file_path (str): Path to the audio file
        filename (str): Name of the file
        background_tasks (BackgroundTasks): FastAPI background tasks object
        
    Raises:
        Exception: If there are issues generating the transcript
    """
    task_status[filename] = "processing"
    try:
        result = get_transcripts(file_path)
        text_file_path = os.path.join(local_transcripts_dir, f'{filename[:-4]}.txt')
        background_tasks.add_task(extract_features, text_file_path)
        task_status[filename] = "completed"
    except Exception as e:
        task_status[filename] = "failed"
        print(f"Error generating transcript for {filename}: {e}")
        raise

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Endpoint to handle file uploads and initiate transcript generation.
    
    Args:
        files (List[UploadFile]): List of uploaded files
        background_tasks (BackgroundTasks): FastAPI background tasks object
        
    Returns:
        List[Dict]: Status for each uploaded file
        
    Raises:
        HTTPException: If there are upload or processing errors
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        raw_dir = os.path.join(os.getcwd(), 'data', 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        
        results = []
        latest_uploaded_files.clear()
        
        for file in files:
            file_path = os.path.join(raw_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            task_status[file.filename] = "queued"
            latest_uploaded_files.append(file.filename)
            print(f'{file.filename} added to latest_upload_files')

            background_tasks.add_task(get_transcript_func, file_path, file.filename)
                
            results.append({
                "filename": file.filename,
                "status": "uploaded"
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/transcripts-status/{filename}")
async def get_transcript_status(filename: str):
    """
    Get the current status of transcript generation for a file.
    
    Args:
        filename (str): Name of the file to check
        
    Returns:
        Dict: Current status of the transcript generation
    """
    status = task_status.get(filename, "not found")
    return {"filename": filename, "status": status}

@router.get("/generate_transcripts")
async def generate_transcripts(files: str, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Generate transcripts for specified files.
    
    Args:
        files (str): Comma-separated list of filenames
        background_tasks (BackgroundTasks): FastAPI background tasks object
        
    Returns:
        List[Dict]: Status and transcript info for each file
        
    Raises:
        HTTPException: If there are processing errors
    """
    try:
        file_list = files.split(',')
        results = []
        
        for filename in file_list:
            file_path = os.path.join(os.getcwd(), 'data', 'raw', filename)
            transcript_filename = f"{os.path.splitext(filename)[0]}.txt"
            transcript_path = os.path.join(os.getcwd(), 'data', 'extracted', transcript_filename)
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
            
            if os.path.exists(transcript_path):
                with open(transcript_path, 'r', encoding='utf-8') as txt_file:
                    extracted_text = txt_file.read()
                status = "completed"
            else:
                background_tasks.add_task(get_transcripts, file_path, filename)
                extracted_text = ""
                status = "processing"

            if status == "completed":
                background_tasks.add_task(extract_features, transcript_path)
            
            results.append({
                "filename": filename,
                "transcript": extracted_text if status == "completed" else None,
                "transcript_file": transcript_filename,
                "status": status
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest-uploads")
async def get_latest_uploads():
    """Get list of recently uploaded files."""
    return {"files": latest_uploaded_files}

@router.get("/raw-data")
async def get_raw_data():
    """Get raw dashboard data."""
    df = get_dashboard_dataframe()
    return df.to_dict(orient='records')

@router.get("/table-data")
async def get_table_data():
    """Get data formatted for table display."""
    df = get_table_dataframe()
    return df.to_dict(orient='records')

@router.get("/call-type-analysis")
async def get_call_analysis():
    """
    Endpoint to get call type analysis data for the dashboard table.
    
    Returns:
        Dict: Call type statistics and associated issues
        
    Raises:
        HTTPException: If there are errors processing the request
    """
    try:
        analysis_data = {'call_types' : call_type_analysis}
        return analysis_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TranscriptRequest(BaseModel):
    files: List[str]

class ChatInput(BaseModel):
    message: str

def load_vector():
    embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": 'cpu'},
    encode_kwargs={"normalize_embeddings": True}
)
#loading embeddings from local
    vectorstore = FAISS.load_local('embeddings/faiss_index_each_doc',embedding_model,allow_dangerous_deserialization= True)
    return vectorstore

vectorstore = load_vector()

def chatbot_response(input_prompt):
    chat_history = []
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 18})
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human",'{input}'),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])


    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        prompt = retriever_prompt
    )

    custom_rag_prompt = ChatPromptTemplate.from_messages([
        ("system","""You are an assistant for question-answering tasks. Use the following pieces of retrieved context: {context} to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise
        
        {format_instructions}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ('human',"{input}")
    ]
    )

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="source: {source}\n\n{page_content}"
    )

    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=custom_rag_prompt,
        document_prompt = document_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    format_instructions = """Strictly Provide the Output Response in the below given format only otherwise you will be penalized.
    Answer: (This is the Place holder for your Response) \n

    Source: (Here I want you to output only the base file name from the file path) \n
    
    Citation: (Here I want you to output the "Refferred Paragraph & Referred section" from the context document)
    """
    out = retrieval_chain.invoke({"input":input_prompt,"chat_history":chat_history,"format_instructions":format_instructions})
    chat_history.append(HumanMessage(content=input_prompt))
    chat_history.append(AIMessage(content=str(out['answer'])))
    return out['answer']


@router.post("/chat")
async def chat(chat_input: ChatInput):
    """
    Endpoint to handle chat interactions.
    
    Args:
        chat_input (ChatInput): User's chat message
        
    Returns:
        Dict: Chatbot response
        
    Raises:
        HTTPException: If there are errors processing the chat request
    """
    try:
        results = chatbot_response(chat_input.message)
        # response_content = results['result'] if results else "No results found."
        return {"response": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
