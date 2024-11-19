import os, re, json
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.llm import LLMChain
from warnings import filterwarnings
from dotenv import load_dotenv
import ast
import plotly.graph_objects as go
load_dotenv()
filterwarnings('ignore')

api_key = os.getenv('GROQ_API_KEY')
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

prompt_template = """
    You are a sentiment analyzer. 
    For the provided conversation, return a dictionary with sentiment scores at various points throughout the conversation. 
    - The keys should represent the percentage of the conversation completed ranging from 0 to 100 in multiples of 10, 
    - The values should be tuples containing the sentiment score and an explanation.
    Example output : 
    
    {{5: (0.2, 'general greeting and positive tone'), 10: (-1, 'issue with the product'), ...}}
    where the keys (e.g., 5, 10) represent 5%, 10%, etc. 
    
    - Only return the dictionary as output, without any introductory phrases or explaination else you will be penalized.

    Text: '{text}'
"""

prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
llm = ChatGroq(
    api_key = api_key, 
    seed = 21,
    model="llama3-8b-8192"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# def plot_graph(sentiment_data) :
#     # Extract data for plotting
#     x_values = list(sentiment_data.keys())
#     y_values = [value[0] for value in sentiment_data.values()]
#     hover_text = [value[1] for value in sentiment_data.values()]
#     colors = ['green' if value > 0 else 'red' for value in y_values]

#     # Create the bar chart
#     fig = go.Figure(data=[go.Bar(
#         x=x_values,
#         y=y_values,
#         hovertext=hover_text,
#         hoverinfo='text', 
#         marker_color=colors,
#     )])

#     # Update layout
#     fig.update_layout(
#         title='Sentiment Analysis Results',
#         xaxis_title='Sentiment Score (%)',
#         yaxis_title='Sentiment Value',
#         showlegend=False
#     )

#     # Show the plot
#     fig.show()

# import streamlit as st
# import plotly.graph_objects as go

# index = 1
# def plot_graph_streamlit(sentiment_data):
#     # Extract data for plotting
#     st.header('Graph')
#     x_values = list(sentiment_data.keys())
#     y_values = [value[0] for value in sentiment_data.values()]
#     hover_text = [value[1] for value in sentiment_data.values()]
#     colors = ['green' if value > 0 else 'red' for value in y_values]
#     index =+ 1
#     # Create the bar chart
#     fig = go.Figure(data=[go.Bar(
#         x=x_values,
#         y=y_values,
#         hovertext=hover_text,
#         hoverinfo='text',
#         marker_color=colors,
#     )])

#     # Update layout
#     fig.update_layout(
#         title='Sentiment Analysis Results',
#         xaxis_title='Sentiment Score (%)',
#         yaxis_title='Sentiment Value',
#         showlegend=False
#     )

#     return fig

metadata_folder = os.path.join('//home//akshatjain//Desktop//rapid_prototyping//sonicVUE_2//metadata')

# inputfolder = os.path.join(os.getcwd(), 'data//extracted')
# sentiments = []
# for file in os.listdir(inputfolder) : 
#     file_path = os.path.join(inputfolder, file)
#     text = load_text_file(file_path)
#     result = llm_chain.run(text)
#     # print(result, '\n\n')
#     sentiment_data = ast.literal_eval(result)
#     # print(sentiment_data, '\n\n')
#     # get_sentiment_analysis(sentiment_data)
#     with open(f'metadata//sentiments//{file}.json', 'w') as fix: 
#         json.dump({'sentiment' : sentiment_data}, fix)
    # print('done')

# for file in os.listdir('/home/akshatjain/Desktop/rapid_prototyping/sonicVUE_2/metadata/sentiments/') : 
#     with open(f"/home/akshatjain/Desktop/rapid_prototyping/sonicVUE_2/metadata/sentiments/{file}", 'r') as f : 
#         doc = json.load(f)
#     sentiment_data = doc['sentiment']
#     fig = plot_graph_streamlit(sentiment_data)
#     st.plotly_chart(fig)
#     st.write(file)
    # print(file)

def get_sentiment_data(text) : 
    result = llm_chain.run(text)
    sentiment_data = ast.literal_eval(result)
    return sentiment_data

