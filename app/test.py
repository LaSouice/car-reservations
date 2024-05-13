import requests
import pandas as pd
import json
from kedro.io import DataCatalog
from kedro_datasets import pandas
# Load the test data
io = DataCatalog({"X_test": pandas.CSVDataset(filepath="data/05_model_input/X_test.csv")})
df = io.load(name='X_test')

# Convert DataFrame to JSON format
data_json = df.to_json(orient='records')
#print(data_json)
# Set the URL of your Flask API endpoint
url = 'http://127.0.0.1:5000/prediction'  # Replace with the actual URL of your API endpoint

# Set the content type header
headers = {'Content-Type': 'application/json'}
print(data_json)
#data = json.dumps({"technology": "0", "actual_price": "93.44", "recommended_price": "64", "num_images": "4", "street_parked": "0", "description": "2"})
# Send POST request with JSON data to the API endpoint
response = requests.post(url, data=data_json, headers=headers)

# Print the response
print(response.content)