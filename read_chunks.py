import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    embeddings = []
    batch_size = 100
    for start in range(0, len(text_list), batch_size):
        batch = text_list[start:start + batch_size]
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": batch
        }, timeout=120)
        try:
            r.raise_for_status()
        except requests.HTTPError as error:
            raise RuntimeError(f"Ollama embedding request failed: {r.text}") from error

        response = r.json()
        if "embeddings" not in response:
            raise RuntimeError(f"Ollama response did not contain embeddings: {response}")
        embeddings.extend(response["embeddings"])

    return embeddings


jsons = os.listdir("jsons")  # List all the jsons 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
# Save this dataframe
joblib.dump(df, 'embeddings.joblib')


