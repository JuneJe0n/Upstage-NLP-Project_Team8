import os
import yaml
import torch
import argparse
from pymilvus import (
    MilvusClient, utility, connections,
    FieldSchema, CollectionSchema, DataType, IndexType,
    Collection, AnnSearchRequest, RRFRanker, WeightedRanker,model
)
from langchain.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from langchain.schema import Document
from dotenv import load_dotenv
from util import (read_test_data, split_documents,
                  extract_question_queries, extract_question_keywords, fetch_wiki_page,
                  detect_missing_context, accuracy)
import uuid



# Get env
load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']
user_agent = os.environ['USER_AGENT']

# Get config
config_path = "/mnt/d/UpstageNLP_milvus/configs.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader = yaml.FullLoader)

test_path = config["TEST_PATH"]
prompt_template = config["PROMPT_TEMPLATE"]
prompt_template_wiki = config["PROMPT_TEMPLATE_WIKI"]
MILVUS_URI = "./milvus.db"
COLLECTION_NAME = "my_rag_collection"



def main():
    # Connect to Milvus
    connections.connect("default", uri=MILVUS_URI)
    print("✅ Connected to Milvus.")

    # Load the existing collection
    collection = Collection(COLLECTION_NAME)
    collection.load()
    print(f"✅ Loaded collection: {COLLECTION_NAME}")
    print(f"Number of entities in collection: {collection.num_entities}")


    prompts, answers = read_test_data(test_path)
    responses = []

    for i, original_prompt in enumerate(prompts):
        # Extract question number (e.g., "QUESTION1")
        question_number = original_prompt.split(")")[0].strip()
        # extract question of prompt
        response = query_rag(original_prompt, collection,question_number)
        responses.append(response)
    
    acc = accuracy(answers, responses)
    print(f"Final Accuracy: {acc}%")
        
def query_rag(original_prompt: str, collection: Collection, question_number: str):
    print(f"🧠 Generating embeddings for {question_number}...")
    DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    embedding_model = model.hybrid.BGEM3EmbeddingFunction(use_fp16=False, device=DEVICE)
    embeddings = embedding_model([original_prompt])

    # Extract dense and sparse embeddings
    query_dense_vector = embeddings["dense"]
    query_sparse_vector = embeddings["sparse"]

    # Define search parameters
    print("🔍 Preparing search requests...")
    sparse_search_params = {"metric_type": "IP"}
    dense_search_params = {"metric_type": "COSINE", "params": {"ef": 500}}

    # Prepare search requests
    sparse_req = AnnSearchRequest(
        query_sparse_vector,  # Sparse embedding
        "sparse_vector",
        sparse_search_params,
        limit=20,
    )

    dense_req = AnnSearchRequest(
        query_dense_vector,  # Dense embedding
        "dense_vector",
        dense_search_params,
        limit=20,
    )

    # Output fields to return
    OUTPUT_FIELDS = ["id", "chunk"]

    print("🔗 Performing hybrid search...")
    try:
    # Perform hybrid search with WeightedRanker
        results = collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=WeightedRanker(0.1, 0.9),  # 가중치를 위치 인수로 전달
            limit=20,  # limit 값 조정
            output_fields=OUTPUT_FIELDS,
        )
    except Exception as e:
        print(f"❌ Error during hybrid search: {e}")
        return None

    # Process results
    if results and len(results[0]) > 0:
        context_text = "\n\n---\n\n".join(
            [hit.entity.get("chunk") for hit in results[0]]
        )
    else:
        context_text = "⚠️No relevant context found in the database."

    # Generate initial prompt
    prompt = ChatPromptTemplate.from_template(prompt_template).format(
        context=context_text, question=original_prompt
    )
    chat_model = ChatUpstage(api_key=upstage_api_key)
    response = chat_model.invoke(prompt)

    # Handle missing context by fetching Wikipedia data
    if detect_missing_context(response.content):
        print(
            f"🔍 Missing context for '{original_prompt}'. Fetching data from Wikipedia..."
        )

        question = extract_question_queries(original_prompt)
        keyword = extract_question_keywords(question)
        print(f"✅ Extracted keyword '{keyword}' from {question}")

        try:
            # Fetch and process Wiki pages
            pages = fetch_wiki_page(keyword)
            for page in pages:
                chunks = split_documents([page])  # Split Wiki pages into smaller chunks
                add_documents_to_milvus(collection, chunks)  # Add chunks to Milvus

            print("✅ Successfully fetched and added Wiki data.")
        except Exception as wiki_error:
            print(f"❌ Error while fetching or adding Wiki data: {wiki_error}")
            return "Unable to fetch additional context from Wikipedia."

        # Re-run the search with the updated vector store
        embeddings = embedding_model([question])
        query_dense_vector = embeddings["dense"]
        query_sparse_vector = embeddings["sparse"]

        sparse_req = AnnSearchRequest(
            query_sparse_vector, "sparse_vector", sparse_search_params, limit=20
        )
        dense_req = AnnSearchRequest(
            query_dense_vector, "dense_vector", dense_search_params, limit=20
        )

        try:
        # Perform hybrid search with WeightedRanker
            results = collection.hybrid_search(
                [sparse_req, dense_req],
                rerank=WeightedRanker(0.1, 0.9),  # 가중치를 위치 인수로 전달
                limit=20,  # limit 값 조정
                output_fields=OUTPUT_FIELDS,
            )
        except Exception as e:
            print(f"❌ Error during hybrid search: {e}")
            return None

        context_text = (
            "\n\n---\n\n".join([hit.entity.get("chunk") for hit in results[0]])
            if results
            else "No context found."
        )
        prompt = ChatPromptTemplate.from_template(prompt_template_wiki).format(
            context=context_text, question=question
        )
        response = chat_model.invoke(prompt)

    return response.content


def add_documents_to_milvus(collection, chunks):
    # Prepare texts and embeddings
    texts = [chunk.page_content for chunk in chunks]

    DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    embedding_model =  model.hybrid.BGEM3EmbeddingFunction(use_fp16=False, device=DEVICE)
    embeddings = embedding_model(texts)

    # Prepare data for insertion
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    sparse_vectors = embeddings["sparse"]
    dense_vectors = embeddings["dense"]

    # Insert into Milvus
    try:
        collection.insert([ids, texts, sparse_vectors, dense_vectors])
        print(f"✅ Added {len(texts)} documents to Milvus.")
    except Exception as e:
        print(f"❌ Failed to add documents: {e}")   

if __name__ == "__main__":
    main()