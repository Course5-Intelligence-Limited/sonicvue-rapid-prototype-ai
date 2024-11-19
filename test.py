import os, re
from groq import Groq
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from pydub import AudioSegment
import assemblyai as aai

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)

connection_string = os.getenv('connection_string')
container_name = 'sonicvue-rapid-prototype'
folder_path = 'extracted/'

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

output_folder = os.path.join(os.getcwd(), 'data', 'extracted')
os.makedirs(output_folder, exist_ok=True)

# Set AssemblyAI API Key
aai.settings.api_key = "c4f23fe64e1e44c8bcd5a69b9ede0a90"

config = aai.TranscriptionConfig(speaker_labels=True)
transcriber = aai.Transcriber()

# Function to detect file encoding using chardet
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

# Function to convert time in milliseconds to seconds
def convert_time_to_seconds(input_file):
    time_pattern = re.compile(r"(Start Time|End Time): (\d+)")

    with open(input_file, 'r+', encoding='utf-8') as file:
        content = file.readlines()

        for i, line in enumerate(content):
            match = time_pattern.search(line)
            if match:
                time_value = int(match.group(2))  # Time in milliseconds
                time_in_seconds = round(time_value / 1000, 2)
                content[i] = line.replace(str(time_value), f"{time_in_seconds:.2f}")

        file.seek(0)
        file.writelines(content)
        file.truncate()  # Ensure to remove any excess content

    print(f"Time conversion complete for '{input_file}'.")

# Function to edit speaker labels and modify transcript
def edit_in_place(file_path, labels):
    with open(file_path, 'r+', encoding='utf-8') as file:
        text = file.read()

        # Replace specific labels like Speaker A -> Agent, Speaker B -> Customer
        text = text.replace('Speaker A', labels.get('Speaker A', 'Agent'))
        text = text.replace('Speaker B', labels.get('Speaker B', 'Customer'))

        # Handle remaining speakers (e.g., 'Speaker C', 'Speaker D' etc.)
        pattern = r'\bSpeaker (\w)\b'

        def replace_speaker(match):
            speaker_letter = match.group(1)
            count = ord(speaker_letter) - ord('A')  # Map to agent number
            return f'Agent {count + 1}'  # Start numbering agents from 1

        text = re.sub(pattern, replace_speaker, text)

        file.seek(0)
        file.write(text)
        file.truncate()

    print(f"Speaker labels updated in '{file_path}'.")

# Function to process the transcript: identify speaker labels and convert times
def process_transcript_in_place(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    agent_count = 1
    customer_identified = False
    labels = {}

    for line in lines:
        if any(phrase in line.lower() for phrase in ['how may i help you', 'how can i help you', 'may i have your name', 'can i have your name']):
            if "Speaker A" in line:
                labels['Speaker A'] = "Agent"
                labels['Speaker B'] = "Customer"
            else:
                labels['Speaker B'] = "Agent"
                labels['Speaker A'] = "Customer"
            break

    edit_in_place(file_path, labels)
    convert_time_to_seconds(file_path)

# Function to get transcripts from AssemblyAI and process them
def get_transcripts(file_path):
    filename = os.path.basename(file_path)[:-4]  # Remove file extension
    text_file_path = os.path.join('./data/extracted', filename + '.txt')

    if not os.path.exists(text_file_path):
        # Transcribe the audio file if the transcript doesn't exist
        transcript = transcriber.transcribe(file_path, config)
        text = ''
        for utterance in transcript.utterances:
            text += f"Speaker {utterance.speaker}: {utterance.text}\n"
            text += f"Start Time: {utterance.start}\nEnd Time: {utterance.end}\n"
        
        with open(text_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)

        process_transcript_in_place(text_file_path)
    else:
        with open(text_file_path, 'r', encoding='utf-8') as txt_file:
            text = txt_file.read()

    return text

# Main execution example
input_folder = os.path.join(os.getcwd(), 'data', 'raw')
input_file = os.path.join(input_folder, 'audio2.mp3')

# Generate and process the transcript
get_transcripts(input_file)
