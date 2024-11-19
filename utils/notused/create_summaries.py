import os
import json
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from warnings import filterwarnings

filterwarnings('ignore')

#load env data
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
#output parser
parser = StrOutputParser()

# Initialize the ChatGroq model
model = ChatGroq(model = 'llama3-8b-8192', seed = 20, api_key=groq_api_key)

#folder to store work
metadata_folder = os.path.join(os.getcwd(), 'metadata')
if not os.path.exists(metadata_folder) : 
    os.makedirs(metadata_folder)


def load_text_file(file_path):
    """Load the contents of a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

chunks_prompts = """
Please summarize the below text : 
Text : '{text}'
Summary : 
"""

map_prompt_template = PromptTemplate(
    input_variables=['text'], 
    template=chunks_prompts
)

# final_combine_prompt = """
# Provide a final summary of the entire document using these rules : 
# 1. The summary should be atleast 150 words long. 
# 2. Don't make up any details. 
# 3. Use bullet points when necessary. 
# 4. Don't write any introductory phrases like "here is the summary of the document".
# document : '{text}'
# """

# final_combine_prompt = """
# Summarize the following document, adhering to these guidelines:
# 1. The summary should be at least 150 words long.
# 2. Do not invent any details.
# 3. Use bullet points when necessary.
# 4. Avoid introductory phrases.

# Document: '{text}'
# """

# final_combine_prompt = """
# Summarize the following document without any introductory phrases or sentences:

# 1. The summary should be at least 150 words long.
# 2. Do not invent any details.
# 3. Use bullet points when necessary.

# Document: '{text}'
# """

final_combine_prompt = """
Summarize the following document directly, starting with the summary content and avoiding any introductory phrases:

1. Do not invent any details.
2. Summary should be atleast 200 words.
3. If you put any introductory lines you will be penalized.
Document: '{text}'
"""


final_combine_prompt_template = PromptTemplate(input_variables=['text'], template=final_combine_prompt)

#using Map Reduce
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
def summarize_document(document):
    """Generate a summary for the given document."""
    chunks = text_splitter.create_documents([document])
    chain = load_summarize_chain(
        model, 
        chain_type = 'map_reduce', 
        map_prompt = map_prompt_template, 
        combine_prompt = final_combine_prompt_template,
        verbose=False
    )
    summary = chain.run(chunks)
    return summary
    

# input_folder = os.path.join(os.getcwd(), 'data//extracted')
# text_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

# for text_file in text_files : 
#     text_path = os.path.join(input_folder, text_file)
#     document = load_text_file(text_path)
#     summary = summarize_document(document)
#     summary.replace("\n\n", "  \n\n")
#     print('done')
#     # summary.replace("\n", " \n")
#     with open(f'metadata//summaries//{text_file}.json', 'w') as file: 
#         json.dump({'summary' : summary}, file)

# text = ''
# for num in [1, 2] : 
#     inputfile = os.path.join(os.getcwd(), f'metadata//summaries//audio__{num}.txt.json')
#     print(os.path.exists(inputfile))
#     with open(inputfile, 'r') as file : 
#         text_ = json.load(file)
#     text += text_['summary']
#     text += '\n\n'
#     print(len(text_['summary']))

# print(len(text))
# summary = summarize_document(text)
# prompt_template = """
#     Rephrase the following text.

#     Text: '{text}'
# """

# prompt = PromptTemplate.from_template(prompt_template)

# llm = ChatGroq(
#     api_key=groq_api_key,
#     seed = 44,
#     model="llama3-8b-8192"
# )
# llm_chain = LLMChain(llm=llm, prompt=prompt)
# summary = llm_chain.run(text)
# with open(os.path.join(os.getcwd(), 'metadata//summaries//overall_summary.json'), 'w') as file : 
#     json.dump({'summary' : summary}, file)    

