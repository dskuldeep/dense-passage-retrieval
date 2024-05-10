from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import configparser
from pymilvus import connections, Collection

def connect_to_milvus():
    """Connect to Milvus."""
    cfp = configparser.RawConfigParser()
    cfp.read('config_serverless.ini')
    milvus_uri = cfp.get('example', 'uri')
    token = cfp.get('example', 'token')

    print(f"Milvus URI: {milvus_uri}")
    print(f"Token: {token}")

    connections.connect("default",
                        uri=milvus_uri,
                        token=token)
    print(f"Connecting to Milvus: {milvus_uri}")

def load_dpr_model():
    """Load pretrained DPR model."""
    model_name = "facebook/dpr-ctx_encoder-single-nq-base"
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
    model = DPRContextEncoder.from_pretrained(model_name)
    return model, tokenizer

def encode_text(text, model, tokenizer):
    """Encode text into embeddings using DPR model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.pooler_output
    return embeddings.numpy()

import time
import json

def parse_results(retrieval_results):
    parsed_results = []
    for hits in retrieval_results:
        parsed_hits = []
        for hit in hits:
            # Extract relevant information from the hit
            hit_info = {
                "id": hit.ids,
                "distance": hit.distances,
                #"entity": hit.entity
            }
            parsed_hits.append(hit_info)
        parsed_results.append(parsed_hits)
    return parsed_results


def perform_dense_retrieval(query, collection, dpr_model, tokenizer):
    # Encode the query using the DPR model and tokenizer
    query_embedding = encode_text(query, dpr_model, tokenizer)
    print("*********" + str(len(query_embedding)))
    # Define search parameters (e.g., metric type and top-k retrieval)
    search_param = {"metric_type": "IP", "params": {"nprobe": 16}}
    limit = 10  # Number of results to retrieve
    
    # Ensure the collection is loaded
    collection.load()
    
    # Perform dense retrieval
    retrieval_results = []
    
    for i in range(len(query_embedding)):
        search_vec = [query_embedding[i]]
        t0 = time.time()
        results = collection.search(search_vec,
                                    anns_field="embedding",
                                    param=search_param,
                                    limit=limit)
        t1 = time.time()
        print(f"Search {i+1} latency: {round(t1 - t0, 4)} seconds")
        retrieval_results.append(results)
    
    return retrieval_results




def main():
    # Connect to Milvus
    connect_to_milvus()

    # Load pretrained DPR model
    dpr_model, tokenizer = load_dpr_model()
    print("DPR model loaded")
    # Load collection
    collection_name = 'embeddings_collection'
    collection = Collection(collection_name)
    print("Collection added")

    # Example query
    query = "What is Kuldeep's Date of Birth ?"
    print("setting query")

    # Perform dense retrieval
    retrieval_results = perform_dense_retrieval(query, collection, dpr_model, tokenizer)
    print(retrieval_results)

    parsed_results = parse_results(retrieval_results)
    print(parsed_results)


    connections.disconnect("default")

if __name__ == "__main__":
    main()
