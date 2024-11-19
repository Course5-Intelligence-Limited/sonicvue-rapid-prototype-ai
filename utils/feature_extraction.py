import os, ast, json
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
model = ChatGroq(model = 'llama-3.1-70b-versatile', seed = 20, api_key=groq_api_key, temperature=0.7)

#folder to store work
metadata_folder = os.path.join(os.getcwd(), 'metadata')
if not os.path.exists(metadata_folder) : 
    os.makedirs(metadata_folder)

def load_text_file(file_path):
    """Load the contents of a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_features(file_path) : 
   file_name = os.path.splitext(os.path.basename(file_path))[0]
   json_file_path = f"./metadata/{file_name}.json"

   if not os.path.exists(json_file_path) : 
      prompt_template = """
      You are an expert call center analyst. For the following transcript along with their timestamps (in seconds), extract the following features : 

      1. **Call Handling Efficiency**:
         - Call time (in secs) 
         - Hold time (in secs) else 0 - if agents asks to hold, subtract that end time and start time from when he next speaks
         - Route time (in secs) else 0 - if agent says he will transfer, subtract that end time with the next agent start time
         - Time taken to comprehend the problem (in secs) else 0 - appropriately subtract the start time with the end time.

      2. **Agent Performance**:
         - Was the agent greeting appropriate? (Yes/No)
         - Was the email address taken? (Yes/No)
         - Was the phone number taken? (Yes/No)

      3. **Customer Experience / Sentiments**:
         - Was the call quality good from the agent's side (identify using internet connection issue/if customer says something regarding it) ? (yes/no)
         - Is the case existing or new? (Existing/New)
         - Did the customer confirm resolution? (Yes/No)
         - Was the customer put on hold? (Yes/No)
         - if customer was put on hold, were they satisfied with hold time ? (yes/no)
         - Did the agent invite more agents? (Yes/No)
         - Was an escalation raised? (Yes/No)
         - Tone of the conversation (if neutral mention as positive) ? (Positive/Negative)


      4. **Problem Resolution**:
         - What issue was discussed? (in 5 words)
         - What was the complexity of the issue? (Easy/Intermediate/Difficult)
         - Issue type? (Assistance, Maintainence, Service)
         - Was the call regarding status query (regarding any field service or digital service) ? (yes/no)
         - Type of call: (Installation/Proactive/Reactive/Scheduled/Incident/Other)
         - Was replacement required? (Yes/No)
         - Part request call? (yes/no)
         - If part request call, were parts dispatched? (yes/no)
         - Was refund required? (Yes/No)
         - Was field service required? (Yes/No)
         - Was digital service offered? (Yes/No)
         

      Call Transcript:
      {transcript}

      Return the response in JSON format, strictly like the given example. 
      No introductory lines (for instance, here is the response in json_format) or explanation should be written else you will be penalized.

      Example Answer : 
      {{
         "call_time": 180,
         "hold_time": 20,
         "route_time": 30,
         "resolution_time": 100,
         "greeting" : "Yes", 
         "phone_number" : "Yes", 
         "email_address" : "No", 
         "call_quality" : "Yes", 
         "new_case" : "New", 
         "resolution_confirmation": "Yes",
         "hold" : "Yes", 
         "hold_satisfaction" : "Yes", 
         "multiple_agents": "No",
         "escalation": "No",
         "call_tone" : "Positive", 
         "issue_discussed": "Product not working",
         "complexity" : "Intermediate",
         "issue_type" : "Maintainence", 
         "status_query" : "No", 
         "call_type": "Scheduled",
         "replacement_required" : "Yes", 
         "part_request" : "No", 
         "parts_dispatch" : "No", 
         "refund_required" : "Yes", 
         "field_service" : "No", 
         "digital_service" : "No"
      }}
      """

      template = PromptTemplate(input_variables=['text'], template=prompt_template)

      # text_file_path = f"./data/extracted/{file_name}.txt"
      transcript = load_text_file(file_path)
      chain = template | model | parser
      features = {}

      while True:
         response = chain.invoke(transcript)
         try:
               # Attempt to parse the LLM response as a dictionary
               features = ast.literal_eval(response)
               break
         except (ValueError, SyntaxError) as e:
               print('Error converting to dictionary')

      print(features)

      # Save the features as a JSON file
      with open(json_file_path, 'w') as json_file:
         json.dump(features, json_file, indent=4)
      
      print(f"Features saved to {json_file_path}")