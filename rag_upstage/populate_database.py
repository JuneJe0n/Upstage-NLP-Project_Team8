"""
This code creates a database from pdf files, 
providing each chunk with a unique ID in order to avoid overlap.
"""

import argparse
import os
import shutil
import uuid
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from util import get_embedding_function, split_documents, load_pdf
from langchain_community.vectorstores.chroma import Chroma


# Get config
config_path = "/home/jiyoon/UpstageNLP_Team8/configs.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader = yaml.FullLoader)

chroma_path = config["CHROMA_PATH"]
data_path = config["DATA_PATH"]

def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_pdf(data_path)
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=chroma_path, embedding_function=get_embedding_function()
    )

    # Calculate IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "ewha.pdf:6:2"
    # Page Source : Page Number : Chunk Index : Chunk Size : Unique ID

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        chunk_size = chunk.metadata.get("chunk_size")
    
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        unique_id = uuid.uuid4().hex[:8] # in case there's a chunk that has the same size in the same page 
        chunk_id = f"{current_page_id}:{current_chunk_index}:{chunk_size}:{unique_id}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


if __name__ == "__main__":
    main()
