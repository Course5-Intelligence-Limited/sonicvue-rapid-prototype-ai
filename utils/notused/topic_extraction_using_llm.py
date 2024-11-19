import os, re, json, ast
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.llm import LLMChain
from warnings import filterwarnings
from dotenv import load_dotenv
load_dotenv()
filterwarnings('ignore')

groq_key = os.getenv('GROQ_API_KEY')

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


prompt_template = """
    You are a topic modelling agent. Extract a numbered list of specific, relevant topics from the following customer support interactions.

    - Focus on unique identifiers, problems, impacts and resolutions.
    - Do not include general or administrative details.
    - Do not include introductory lines; failure to comply will result in penalties.
    - Do not explain the topics / reason at all.
    - Do not emphasize any topic with '**' at all.
    - Topics should have 3 words at max.
    - Total 7 topics max.

    Text: '{text}'
"""

prompt = PromptTemplate.from_template(prompt_template)

llm = ChatGroq(
    api_key=groq_key,
    seed = 44,
    model="llama3-8b-8192"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)


metadata_folder = os.path.join('//home//akshatjain//Desktop//rapid_prototyping//sonicVUE_2//metadata')
inputfolder = os.path.join('//home//akshatjain//Desktop//rapid_prototyping//sonicVUE_2//data//extracted')

def generate_topics(text) : 
    result = llm_chain.run(text)
    list_of_topcs = re.findall(r'\d+\.\s*(.+)', result)
    cleaned_topics = [re.sub(r'\*', '', topic) for topic in list_of_topcs]    
    return cleaned_topics

# for file in os.listdir(inputfolder) : 
#     file_path = os.path.join(inputfolder, file)
#     text = load_text_file(file_path)
#     result = llm_chain.run(text)
#     list_of_topcs = re.findall(r'\d+\.\s*(.+)', result)
#     cleaned_topics = [re.sub(r'\*', '', topic) for topic in list_of_topcs]
#     with open(f'metadata//topics//{file}.json', 'w') as fix: 
#         json.dump({'list_of_topics' : list_of_topcs}, fix)
#     print(cleaned_topics, '\n')
