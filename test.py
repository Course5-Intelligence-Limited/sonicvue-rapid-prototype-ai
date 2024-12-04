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



local_data_dir = "data/raw"
local_transcripts_dir = "data//extracted"
local_metadata_dir = "metadata"
for dir_path in [local_data_dir, local_metadata_dir, local_transcripts_dir] : 
    os.makedirs(dir_path, exist_ok=True)

latest_uploaded_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']

def get_table_dataframe() -> pd.DataFrame:
    """Convert metadata JSON files to a DataFrame for table view with specific KPIs"""
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
    # print(latest_uploaded_files)
    for filename in latest_uploaded_files:
        json_path = os.path.join(local_metadata_dir, f"{filename[:-4]}.json")
        print(os.path.exists(json_path))
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Only include specified KPI columns
                filtered_data = {k: data.get(k, '') for k in kpi_columns}
                filtered_data['filename'] = filename
                data_list.append(filtered_data)
    
    if not data_list:
        return pd.DataFrame()
        
    df = pd.DataFrame(data_list)
    
    # Convert time columns to numeric
    time_columns = ['call_time', 'hold_time', 'route_time', 'resolution_time']
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(df.head())
    # return df

get_table_dataframe()