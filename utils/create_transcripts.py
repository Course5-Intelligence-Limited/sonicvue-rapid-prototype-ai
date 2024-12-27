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
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

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
        text = text.replace('Speaker A', labels.get('Speaker A', 'Agent 1'))
        text = text.replace('Speaker B', labels.get('Speaker B', 'Customer'))

        # Handle remaining speakers (e.g., 'Speaker C', 'Speaker D' etc.)
        pattern = r'\bSpeaker (\w)\b'

        def replace_speaker(match):
            speaker_letter = match.group(1)
            count = ord(speaker_letter) - ord('A')  # Map to agent number
            return f'Agent {count}'  # Start numbering agents from 1

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
                labels['Speaker A'] = "Agent 1"
                labels['Speaker B'] = "Customer"
            else:
                labels['Speaker B'] = "Agent 1"
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
# input_folder = os.path.join(os.getcwd(), 'data', 'raw')
# input_file = os.path.join(input_folder, 'audio2.mp3')

# Generate and process the transcript
# get_transcripts(input_file)


# def chunk_audio_file(audio_file_path, max_size_mb=25):
#     audio = AudioSegment.from_file(audio_file_path)
    
#     # Check file size in MB
#     if os.path.getsize(audio_file_path) > max_size_mb * 1024 * 1024:
#         num_chunks = os.path.getsize(audio_file_path) // (max_size_mb * 1024 * 1024) + 1
#         chunk_length = len(audio) // num_chunks if num_chunks > 0 else len(audio)
        
#         chunks = []
#         for i in range(0, len(audio), chunk_length):
#             chunks.append(audio[i:i + chunk_length])
#         return chunks
#     else:
#         return [audio]  # Return as a single chunk if within size limit

# Main function
# def process_audio(audio_file_path):
#     chunks = chunk_audio_file(audio_file_path)
    
#     transcripts = []
#     for i, chunk in enumerate(chunks):
#         # Export the chunk to a temporary file for transcription
#         temp_filename = f"temp_audio_part_{i + 1}.wav"
#         chunk.export(temp_filename, format="wav")
        
#         # Get transcript
#         transcript = get_transcripts(temp_filename)
#         transcripts.append(transcript)
    
#     return transcripts


# with open(input_file, 'r', encoding='utf-8') as txt_file:
#     text = txt_file.read()    

# print(text)   
# print(transcripts)

# for file in os.listdir(input_folder) : 
#     file_path = os.path.join(input_folder, file)
    # transcripts = get_transcripts(file_path)
    # txt_filename = f"{os.path.splitext(file)[0]}.txt"
    # txt_filepath = os.path.join(output_folder, txt_filename)
    
    # extracted_text = ' '.join(transcripts)
    

#     print(extracted_text)
    # upload the extracted text to blob storage
    # txt_filename = f"{os.path.splitext(file)[0]}.txt"
    # txt_filepath = os.path.join(output_folder, txt_filename)
    # with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
    #     txt_file.write(extracted_text)
    # with open(txt_filepath, 'r', encoding='utf-8') as txt_file:    
    #     text = txt_file.read()
    #     container_client.upload_blob(name=f"extracted/{txt_filename}", data = text)
    # print(f"Text from {file} saved to {txt_filepath}")
    # print(f"{txt_filename} uploaded to blob storage")

# file_path = os.path.join(input_folder, 'audio__5.wav')
# file = 'audio__5.wav'
# transcripts = process_audio(file_path)
# extracted_text = ' '.join(transcripts)
# # upload the extracted text to blob storage
# txt_filename = f"{os.path.splitext(file)[0]}.txt"
# txt_filepath = os.path.join(output_folder, txt_filename)
# with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
#     txt_file.write(extracted_text)
# with open(txt_filepath, 'r', encoding='utf-8') as txt_file:    
#     text = txt_file.read()
#     container_client.upload_blob(name=f"extracted/{txt_filename}", data = text)
# print(f"Text from {file} saved to {txt_filepath}")
# print(f"{txt_filename} uploaded to blob storage")

