import os, ast, json
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from warnings import filterwarnings
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import StrOutputParser
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

def format_instructions_creator():
    """
    Creates formatting instructions for the output parser based on the call center analysis schema.

    Returns:
    str: The formatting instructions.
    """
    call_time = ResponseSchema(
        name="call_time",
        description="This column contains the total time of the call, calculated based on the difference between the start and end times. Return only time in secs."
    )
    # hold = ResponseSchema(
    #     name="Hold",
    #     description="Indicates whether the customer was placed on hold during the call ('Yes' or 'No')."
    # )
    hold = ResponseSchema(
    name="hold",
    description="Indicates whether the customer was placed on hold (i.e., temporarily paused from speaking with an agent) during the call. Respond 'Yes' if they were on hold, 'No' if they were not."
    )

    # hold_satisfaction = ResponseSchema(
    #     name="Hold Satisfaction",
    #     description="If the customer was placed on hold, this column indicates their satisfaction with the hold time, default is yes unless they complain ('Yes' or 'No')."
    # )

    hold_satisfaction = ResponseSchema(
    name="hold_satisfaction",
    description="If the customer was placed on hold, indicates their satisfaction with the hold time. Default is 'Yes' unless they complain about the hold, in which case respond 'No'."
    )

    # hold_time = ResponseSchema(
    #     name="Hold Time",
    #     description="Contains the total duration of time for holding, can be to check details, issue history etc. and does not include time for routing the call to another agent, calculated from the start and end times. Return only time in secs."
    # )

    hold_time = ResponseSchema(
    name="hold_time",
    description="The total time the customer was on hold (i.e., time not speaking with an agent), excluding call routing time. Return only the time in seconds."
    )

    multiple_agents = ResponseSchema(
    name="multiple_agents",
    description="Indicates if call was routed two times or more, else no ('yes', 'no')."
    )

    route_time = ResponseSchema(
        name="route_time",
        description="The total time the call was routed to a different agent, calculated using start and end times. Return only time in secs."
    )
    resolution_time = ResponseSchema(
        name="resolution_time",
        description="The time taken by the final routed agent between when he starts solving the issue or trying out solutions and when the resolution is done, calculated based on the transcript. Return only time in secs."
    )
    resolution_time_start = ResponseSchema(
        name="resolution_time_start",
        description="The time when final routed agent starts solving the issue or trying out solutions, calculated based on the transcript. Return only time in secs."
    )
    greeting = ResponseSchema(
        name="greeting",
        description="Indicates whether the agent's greeting was appropriate ('Yes' or 'No')."
    )
    phone_number = ResponseSchema(
        name="phone_number",
        description="Indicates if the agent asked for and recorded the customer's phone number ('Yes' or 'No')."
    )
    email_address = ResponseSchema(
        name="email_address",
        description="Indicates if the agent asked for and recorded the customer's email address ('Yes' or 'No')."
    )
    call_quality = ResponseSchema(
        name="call_quality",
        description="Indicates if the customer didn't have issues with the call quality, such as inaudibility, or connection issues etc. ('Yes' or 'No')."
    )
    new_case = ResponseSchema(
        name="case_type",
        description="Indicates if the case was new or existing. The value should be 'New' or 'Existing'."
    )
    resolution_confirmation = ResponseSchema(
        name="resolution_confirmation",
        description="Indicates whether the customer confirmed the resolution ('Yes' or 'No')."
    )
    escalation = ResponseSchema(
        name="escalation",
        description="Indicates whether an escalation was raised during the call ('Yes' or 'No')."
    )
    call_tone = ResponseSchema(
        name="call_tone",
        description="The tone of the conversation, either 'Positive' or 'Negative'."
    )
    issue_discussed = ResponseSchema(
        name="issue_discussed",
        description="Brief description of the issue discussed in the call (5 words or fewer)."
    )
    complexity = ResponseSchema(
        name="complexity",
        description="The complexity of the issue, can be 'Easy', 'Intermediate', or 'Difficult'."
    )
    issue_type = ResponseSchema(
        name="issue_type",
        description="The type of issue discussed. Can be 'Assistance', 'Maintenance', or 'Service'."
    )
    status_query = ResponseSchema(
        name="status_query",
        description="Indicates if the call was related to a status query ('Yes' or 'No')."
    )
    call_type = ResponseSchema(
        name="call_type",
        description="""The type of call. It can be one of the following:
        'Installation' - Waiting for installing missing parts or services.
        'Proactive' - Call regarding scheduled maintenance or preventive checks.
        'Reactive' - System will send telemetric messages that need to be addressed.
        'Scheduled' - Calls based on a contract (quarterly/half-yearly/annually).
        'Incident' - General inquiry call for normal customer concerns.
        'Other' - Any other type of call not covered above."""
    )
    replacement_required = ResponseSchema(
        name="replacement_required",
        description="Indicates whether a replacement part or service was required during the call ('Yes' or 'No')."
    )
    # Additional fields for part requests, dispatch, refunds, and services
    part_request = ResponseSchema(
        name="part_request",
        description="Indicates whether this was a part request call ('Yes' or 'No')."
    )
    parts_dispatch = ResponseSchema(
        name="parts_dispatch",
        description="Indicates whether parts were dispatched during the call ('Yes' or 'No')."
    )
    # refund_required = ResponseSchema(
    #     name="Refund Required",
    #     description="Indicates whether a refund was required during the call ('Yes' or 'No')."
    # )
    field_service = ResponseSchema(
        name="field_service",
        description="Indicates whether field service was required during the call ('Yes' or 'No')."
    )
    digital_service = ResponseSchema(
        name="digital_service",
        description="Indicates whether digital portal was recommended during the call ('Yes' or 'No')."
    )

    response_schemas = [
        call_time, hold, hold_satisfaction, hold_time, multiple_agents, route_time, resolution_time, resolution_time_start, greeting,
        phone_number, email_address, call_quality, new_case, resolution_confirmation, escalation, call_tone,
        issue_discussed, complexity, issue_type, status_query, call_type, replacement_required,
        part_request, parts_dispatch, field_service, digital_service
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    return format_instructions, output_parser

def generate_response_call_center(llm, transcript_text):
    """
    Function to process the call center transcript and generate structured responses based on the provided transcript.
    This function returns the output as JSON format.
    """
    format_instructions, output_parser = format_instructions_creator()

    # Create a prompt using the template
    prompt = PromptTemplate.from_template("""
    Please analyze the following call center transcript and provide the required structured data in the format described below:
    '{format_instructions}'

    Transcript:
    '{context}'
    """, partial_variables={"format_instructions": format_instructions})

    chain = prompt | llm | output_parser
    out = chain.invoke({"context": transcript_text})
    print(out)
    return out
    # Convert the result to JSON format and return it
    # result_json = json.loads(out)  # Assuming the model returns a valid JSON string
    # return result_json

def extract_features(file_path) : 
   file_name = os.path.splitext(os.path.basename(file_path))[0]
   json_file_path = f"./metadata/{file_name}.json"

   if not os.path.exists(json_file_path) : 
      with open(file_path, 'r') as file : 
         text = file.read()
      json_response = generate_response_call_center(model, text)
      with open(json_file_path, 'w') as json_file:
         json.dump(json_response, json_file, indent=4)
      print(f"Features saved to {json_file_path}")

input_folder = './data/extracted'
for file in os.listdir(input_folder) : 
    file_path = os.path.join(input_folder, file)
    extract_features(file_path)
    

#       while True:
#          response = chain.invoke(transcript)
#          try:
#                # Attempt to parse the LLM response as a dictionary
#                features = ast.literal_eval(response)
#                break
#          except (ValueError, SyntaxError) as e:
#                print('Error converting to dictionary')

#       print(features)

      # Save the features as a JSON file
