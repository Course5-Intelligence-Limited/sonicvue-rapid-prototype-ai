import os
import json
import pandas as pd

# Path to the metadata folder
metadata_folder = 'data/extracted'
# print(os.path.exists(metadata_folder))
complete_text = ''
for file_name in os.listdir(metadata_folder) : 
    with open(os.path.join(metadata_folder, file_name), 'r') as file : 
        text = file.read()
    complete_text += file_name[:-4]
    complete_text += '\n\n'
    complete_text += text
    complete_text += '\n\n'
    
with open('full_transcript.txt', 'w') as file:
    file.write(complete_text)


print('done')
