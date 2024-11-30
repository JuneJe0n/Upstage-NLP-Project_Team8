'''
Reset Database by
python populate_milvus.py --reset
'''
import argparse
import yaml
import pymilvus
from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema, DataType, Collection,model
)
from util import split_documents, load_pdf
import uuid
import torch

# Load configuration
config_path = "../configs.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
data_path = config["DATA_PATH"]

# Set device and initialize embedding model
DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
embedding_model = model.hybrid.BGEM3EmbeddingFunction(use_fp16=False, device=DEVICE)
EMBEDDING_DIM = embedding_model.dim['dense']
MILVUS_URI = "./milvus.db"
COLLECTION_NAME = "my_rag_collection"


# Connect to Milvus
connections.connect(uri=MILVUS_URI)
print("✅ Successfully connected to Milvus.")

# Function to reset the database
def reset_database():
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"✅ Dropped existing collection: {COLLECTION_NAME}")

# Function to create a collection
def create_collection():
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema = CollectionSchema(fields, "")
    collection = Collection(COLLECTION_NAME, schema, consistency_level="Eventually")

    
    # Add HNSW search index
    M = 16
    efConstruction = M * 2
    INDEX_PARAMS = {"M": M, "efConstruction": efConstruction}

    # Create indices for vector fields
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    dense_index = {"index_type": "HNSW", "metric_type": "COSINE", "params": INDEX_PARAMS}
    collection.create_index("sparse_vector", sparse_index)
    collection.create_index("dense_vector", dense_index)
    print(f"✅ Created collection: {COLLECTION_NAME}")
    return collection

# Function to add data to Milvus
def add_to_milvus(collection, chunks, embeddings):
    # Prepare separate lists for each field
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]  # Unique IDs for each chunk
    chunks_list = [chunk.page_content for chunk in chunks]
    sparse_vectors = embeddings["sparse"]
    dense_vectors = embeddings["dense"]

    # Check the lengths of all fields to ensure they match
    if not (len(ids) == len(chunks_list) == sparse_vectors.shape[0] == len(dense_vectors)):
        raise ValueError("Lengths of IDs, chunks, sparse vectors, and dense vectors must match.")

    try:
        # Insert data into the collection
        collection.insert([ids, chunks_list, sparse_vectors, dense_vectors])
        print(f"👉 Successfully added {len(chunks_list)} documents to Milvus.")
    except Exception as e:
        print(f"❌ Failed to insert data: {e}")


# Main workflow
def main(reset=False):
    try:
        if reset:
            reset_database()

        if not utility.has_collection(COLLECTION_NAME):
            collection = create_collection()
        else:
            collection = Collection(COLLECTION_NAME)

        collection.load()

        # # Process documents
        # print("📄 Loading PDF documents...")
        # documents = load_pdf(data_path)
        # chunks = split_documents(documents)
        # list_of_strings = [chunk.page_content for chunk in chunks]

        # # Generate embeddings
        # print("🔍 Generating embeddings...")
        # embeddings = embedding_model(list_of_strings)

        # Add data to Milvus
        # print("🛠 Adding data to Milvus...")
        # add_to_milvus(collection, chunks, embeddings)

    finally:
        # Ensure disconnection at the end
        connections.disconnect("default")
        print("✅ Safely disconnected from Milvus.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    main(reset=args.reset)
