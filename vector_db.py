import configparser
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
import random
import time

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


def create_collection(collection_name, dimension, embeddings, file_names, batch_size=1000):
    """Create a collection in Milvus with an auto-incremented ID as the primary key."""
    # Establish connection to Milvus
    #connections = connect_to_milvus()
    
    # Define collection schema
    dim = dimension
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True, description="auto-incremented ID")
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim, description="embedding vector")
    file_name_field = FieldSchema(name="file_name", dtype=DataType.VARCHAR, description="file name", max_length=256)
    schema = CollectionSchema(fields=[id_field, embedding_field, file_name_field], description="Collection for embeddings")
    
    # Check if collection already exists, drop and recreate if needed
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists. Dropping and recreating the collection.")
        drop_result = utility.drop_collection(collection_name)
        
    # Create collection
    collection = Collection(name=collection_name, schema=schema)
    print(f"Created collection: {collection_name}")
    
    # Insert embeddings and file names into the collection in batches
    total_rt = 0
    for i in range(0, len(embeddings), batch_size):
        embeddings_batch = embeddings[i:i+batch_size]
        file_names_batch = file_names[i:i+batch_size]
        entities = [
            {"embedding": embedding.tolist(), "file_name": file_name}
            for embedding, file_name in zip(embeddings_batch, file_names_batch)
        ]
        collection.insert(entities)
    
    print("Insertion completed successfully.")
    index_params = {"index_type": "IVF_FLAT", "params": {"nlist": 16384}}
    collection.create_index("embedding", index_params=index_params, timeout=60)  # Adjust timeout as needed
    print("Index created successfully.")

def upload_embeddings_to_milvus(embeddings_path, file_names_path, collection_name, dimension, host='localhost', port='19530'):
    """Upload embeddings to Milvus."""
    embeddings = np.load(embeddings_path)
    print("Embedding Loaded")
    file_names = np.load(file_names_path)
    print("filenames loaded")
    connect_to_milvus()
    # Create collection
    create_collection(collection_name, dimension, embeddings, file_names)

# Example usage:
embeddings_path = 'embeddings.npy'
file_names_path = 'file_names.npy'
collection_name = 'embeddings_collection'
dimension = 768  # Dimensionality of the embeddings
upload_embeddings_to_milvus(embeddings_path, file_names_path, collection_name, dimension)
