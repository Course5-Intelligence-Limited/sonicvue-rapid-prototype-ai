import os
import json
import pandas as pd

# Path to the metadata folder
metadata_folder = 'metadata'
print(os.path.exists(metadata_folder))
# Initialize an empty list to hold all rows (data from each JSON file)
data = []

# Loop through each JSON file in the metadata folder
for filename in os.listdir(metadata_folder):
    
    if filename.endswith('.json'):
        # Full path to the file
        file_path = os.path.join(metadata_folder, filename)
        
        # Read the content of the JSON file
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        
        # Add the recording name (filename without extension) as the first column
        json_data['recording_name'] = filename.replace('.json', '')
        
        # Append the data to the list
        data.append(json_data)

# Create a DataFrame from the list of dictionaries (json_data)
df = pd.DataFrame(data)

# Reorder the columns to make sure 'recording_name' is the first column
df = df[['recording_name'] + [col for col in df.columns if col != 'recording_name']]

# Save the DataFrame to an Excel file
output_file = 'metadata_output.xlsx'
df.to_excel(output_file, index=False)

print(f"Data has been saved to {output_file}")
