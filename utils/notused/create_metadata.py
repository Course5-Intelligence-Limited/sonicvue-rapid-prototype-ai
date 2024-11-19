import os, json
from pydub import AudioSegment
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.llm import LLMChain
from warnings import filterwarnings
from dotenv import load_dotenv
load_dotenv()
filterwarnings('ignore')
import ast

groq_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(
    api_key=groq_key,
    seed = 44,
    model="llama-3.1-70b-versatile"
)
# llm_chain = LLMChain(llm=llm, prompt=prompt)

"""
average call time 
tone -positive or negative
call hygiene
"""

def load_text_file(file_path) : 
    with open(file_path, 'r') as file : 
        text = file.read()
    return text

def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_in_seconds = len(audio) / 1000  # Length is in milliseconds
    return duration_in_seconds

def call_hygiene(file_path) : 
    prompt_template = """
        Check three fields in the given text (Greetings, Phone Number and EmailAddress) and report 0/1 for all three.
        - Return only a list of numbers.
        - Do not add introductory paragraph else you will be penalized.

        For example : 
        Situation : if agent greeted the customer and asked for email but didn't ask for phone number
        Output : (1, 0, 1)  


        Text: '{text}'
    """
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    text = load_text_file(file_path)
    result = llm_chain.run(text)
    return result


def call_tone(file_path) : 
    prompt_template = """
        Return the tone of the following conversation between 0 and 100 where 100 is fully positive and 0 is entirely negative tone.
        - Return only one number.
        - Do not add introductory paragraph else you will be penalized.

        For example : 
        Output : 60  


        Text: '{text}'
    """
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    text = load_text_file(file_path)
    result = llm_chain.run(text)
    return result

def issue_type(file_path) : 
    prompt_template = """
        Check the type of issue in the following conversation either (Assistance, Maitanence or Enquiry) and report 1 for the one of them and 0 for the rest.
        - Return only a list of numbers.
        - Do not add introductory paragraph else you will be penalized.

        For example : 
        Situation : if the conversation is of enquiry type
        Output : (0, 0, 1)  


        Text: '{text}'
    """
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    text = load_text_file(file_path)
    result = llm_chain.run(text)
    return result

path = './data/extracted/audio__1.txt'
audio_path = './data/raw_data/audio__1.mp3'

kpis = {}
total_time = 0
tone = 0
hygiene = [0, 0, 0]
issues = [0, 0, 0]

for file in os.listdir('./data/raw_data') : 
    folder_path = os.path.abspath('./data')
    audio_path = os.path.join(folder_path,'raw_data',file)
    file_name = os.path.basename(audio_path)[:-4]
    file_path = os.path.join(folder_path,'extracted', file_name+'.txt')
    total_time += get_audio_duration(audio_path)
    tone += (int(call_tone(file_path))/100)
    # call_hygiene(file)
    x =  ast.literal_eval(call_hygiene(file_path))
    # print(x)
    hygiene[0] += x[0]
    hygiene[1] += x[1]
    hygiene[2] += x[2]
    x =  ast.literal_eval(issue_type(file_path))
    # print(x)
    issues[0] += x[0]
    issues[1] += x[1]
    issues[2] += x[2]
    # print(file_path)
    
# print(total_time, tone, hygiene, issues)
kpis = {
    'total_calls' : len(os.listdir('./data/raw_data')), 
    'avg_duration' : total_time/len(os.listdir('./data/raw_data')), 
    'avg_tone' : tone,
    'issue_type' : issues, 
    'hygiene' : hygiene
}

with open('./metadata/kpis.json', 'w') as file : 
    json.dump(kpis, file, indent=4)

# print('duration: ')
# get_audio_duration(audio_path)
# print('duration: ')
# call_hygiene(path)
# print('duration: ')
# call_tone(path)
# print('duration: ')
# issue_type(path)

# with open(f'metadata//summaries//{text_file}.json', 'w') as file: 
#         json.dump({'summary' : summary}, file)