import json
import os
from typing import Dict, Any, List
import asyncio
from .create_summaries import summarize_document
from .topic_extraction_using_llm import generate_topics
from .sentiment_analysis_graph import get_sentiment_data
# from create_metadata import ;

async def process_metadata_for_file(filename: str, transcript: str) -> None:
    """Process a transcript and save metadata asynchronously"""
    
    # Create metadata directory if it doesn't exist
    metadata_dir = os.path.join(os.getcwd(), 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Base filename without extension
    base_filename = os.path.splitext(filename)[0]
    
    # Process different aspects concurrently
    tasks = [
        asyncio.create_task(async_generate_summary(transcript)),
        asyncio.create_task(async_extract_topics(transcript)),
        asyncio.create_task(async_analyze_sentiment(transcript)),
        # asyncio.create_task(async_extract_kpis(transcript))
    ]
    
    # Wait for all tasks to complete
    # summary, topics, sentiment, kpis = await asyncio.gather(*tasks)
    summary, topics, sentiment = await asyncio.gather(*tasks)
    
    # Combine all metadata
    metadata = {
        "filename": filename,
        "summary": summary,
        "topics": topics,
        "sentiment": sentiment,
        # "kpis": kpis
    }
    
    # Save metadata to JSON file
    metadata_path = os.path.join(metadata_dir, f"{base_filename}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

# Async wrapper functions for your existing utilities
async def async_generate_summary(transcript: str) -> str:
    return summarize_document(transcript)

async def async_extract_topics(transcript: str) -> List[str]:
    return generate_topics(transcript)

async def async_analyze_sentiment(transcript: str) -> Dict[str, Any]:
    return get_sentiment_data(transcript)
