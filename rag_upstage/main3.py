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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Define constants for hyperparameters (향후 튜닝 진행해보면 될듯)
WEIGHTED_RANKER_SPARSE_WEIGHT = 0.1
WEIGHTED_RANKER_DENSE_WEIGHT = 0.9
SEARCH_LIMIT = 50
EF = 500
SIMILARITY_THRESHOLD = 0.75

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
    responses = [query_rag(prompt, collection, f"QUESTION{i+1}") for i, prompt in enumerate(prompts)]
    acc = accuracy(answers, responses)
    logging.info(f"Final Accuracy: {acc}%")
    
    acc = accuracy(answers, responses)
    print(f"Final Accuracy: {acc}%")
        
def query_rag(prompt: str, collection, question_number: str):
    """
    Main query function for performing RAG-based search and returning a response.
    """ 
    logging.info(f"🧠 Generating embeddings for {question_number}...")
    dense_vector, sparse_vector = generate_embeddings([prompt])

    logging.info("🔍 Performing hybrid search...")
    try:
        # Perform hybrid search
        results = hybrid_search(collection, dense_vector, sparse_vector)
    except Exception as e:
        print(f"❌ Error during hybrid search: {e}")
        return None

    if results:
        processed_results = post_process_results(results)
        context_text = "\n\n---\n\n".join([res["chunk"] for res in processed_results])
    else:
        context_text = "⚠️ No relevant context found in the database."

    # Format the prompt with the context
    prompt_formatted = ChatPromptTemplate.from_template(prompt_template).format(
        context=context_text, question=prompt
    )
    chat_model = ChatUpstage(api_key=upstage_api_key)
    response = chat_model.invoke(prompt_formatted)


    # Handle missing context by fetching Wikipedia data
    #model로 답변 생성한 거에서 문맥에없다고 나오면 위키검색 시작하는거라, 이화관련 내용아니어도 모델이 답변 잘 만들어내면 wiki_Search안하는 거임
    if detect_missing_context(response.content):
        print(
            f"🔍 Missing context for '{prompt}'. Fetching data from Wikipedia..."
        )

        question = extract_question_queries(prompt)
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

         # Re-run hybrid search after updating collection
        dense_vector, sparse_vector = generate_embeddings([question])
        try:
            results = hybrid_search(collection, dense_vector, sparse_vector)
        except Exception as e:
            print(f"❌ Error during hybrid search: {e}")
            return None

        context_text = "\n\n---\n\n".join(
            [res["chunk"] for res in post_process_results(results)]
        ) if results else "No context found."

        prompt_formatted = ChatPromptTemplate.from_template(prompt_template_wiki).format(
            context=context_text, question=question
        )
        response = chat_model.invoke(prompt_formatted)

    return response.content


def generate_embeddings(texts, device='cuda:3' if torch.cuda.is_available() else 'cpu'):
    """
    Generate dense and sparse embeddings for given texts.
    """
    try:
        embedding_model = model.hybrid.BGEM3EmbeddingFunction(use_fp16=True, device=device)
        embeddings = embedding_model(texts)
        return embeddings["dense"], embeddings["sparse"]
    except KeyError as e:
        logging.error(f"Error generating embeddings: {e}")
        return None, None


def hybrid_search(collection, dense_vector, sparse_vector):
    """
    Perform a hybrid search by combining dense and sparse search results.
    """
    try:
        sparse_req = AnnSearchRequest(
            sparse_vector, "sparse_vector", {"metric_type": "IP"}, limit=SEARCH_LIMIT
        )
        dense_req = AnnSearchRequest(
            dense_vector, "dense_vector", {"metric_type": "COSINE", "params": {"ef": EF}}, limit=SEARCH_LIMIT
        )
        results = collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=WeightedRanker(WEIGHTED_RANKER_SPARSE_WEIGHT, WEIGHTED_RANKER_DENSE_WEIGHT),
            limit=SEARCH_LIMIT,
            output_fields=["id", "chunk"],
        )
        logging.info(f"Hybrid search returned {len(results[0]) if results else 0} results.")
        return results[0] if results else []
    
    except Exception as e:
        logging.error(f"Error during hybrid search: {e}")
        return []


def post_process_results(results, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    Additional post-processing of results to filter by similarity and improve ranking.
    """
    processed_results = []
    seen_chunks = set()
    for result in results:
        try:
            chunk = result.entity.get("chunk")
            similarity_score = result.distance
            if result.entity.id not in seen_chunks and similarity_score >= similarity_threshold:
                seen_chunks.add(result.entity.id)
                processed_results.append({
                    "chunk": chunk,
                    "similarity_score": similarity_score
                })
        except Exception as e:
            logging.warning(f"Error processing result: {e}")
    processed_results.sort(key=lambda x: x["similarity_score"], reverse=True)
    logging.info(f"Post-processed results count: {len(processed_results)}")
    return processed_results


def add_documents_to_milvus(collection, chunks):
    """
    Add processed documents to the Milvus collection.
    """
    try:
        texts = [chunk.page_content for chunk in chunks]
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        embedding_model = model.hybrid.BGEM3EmbeddingFunction(use_fp16=False, device=device)
        embeddings = embedding_model(texts)
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        sparse_vectors = embeddings["sparse"]
        dense_vectors = embeddings["dense"]
        collection.insert([ids, texts, sparse_vectors, dense_vectors])
        logging.info(f"✅ Added {len(texts)} documents to Milvus.")
    except Exception as e:
        logging.error(f"❌ Failed to add documents to Milvus: {e}")
 


if __name__ == "__main__":
    main()