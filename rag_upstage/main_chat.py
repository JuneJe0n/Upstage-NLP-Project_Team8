import os
import yaml
import torch
import argparse
import numpy as np
from pymilvus import (
    MilvusClient, utility, connections,
    FieldSchema, CollectionSchema, DataType, IndexType,
    Collection, AnnSearchRequest, RRFRanker, WeightedRanker,model
)
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_upstage import ChatUpstage
from langchain.schema import Document
from dotenv import load_dotenv
from util import (read_test_data, split_documents, check_question,detect_missing_context,check_chat,
                  extract_question_queries, extract_question_keywords, fetch_wiki_page, sem_split_documents, accuracy)
from generate_prompt import classify_mmlu_domain, generate_prompt, generate_chat_prompt
import uuid

# Setup
milvus_uri = "./milvus.db"


# Define hyperparameters 
WEIGHTED_RANKER_SPARSE_WEIGHT = 0.1
WEIGHTED_RANKER_DENSE_WEIGHT = 0.9
SEARCH_LIMIT = 10
EF = 500
THRESHOLD = "standard_deviation"   # ("percentile", "standard_deviation", "interquartile", "gradient")

# Get env
load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']
user_agent = os.environ['USER_AGENT']

# Get config
config_path = "../configs.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader = yaml.FullLoader)
test_path = config["TEST_PATH"]
prompt_template_ewha = config["PROMPT_TEMPLATE_EWHA"]

def main():
    # Connect to Milvus
    connections.connect("default", uri=milvus_uri)
    print("✅ Connected to Milvus.")

    # Load collection
    ewha_collection = Collection("ewha_collection")
    ewha_collection.load()
    print(f"✅ Loaded 'ewha_collection'")

    wiki_collection = Collection("my_rag_collection")
    wiki_collection.load()
    print(f"✅ Loaded 'wiki_collection'")

    # Read test datset
    prompts, answers = read_test_data(test_path)
    responses = []

    # Create responses
    for i, prompt in enumerate(prompts):
        response = query_rag(prompt=prompt,ewha_collection=ewha_collection,wiki_collection=wiki_collection)
        responses.append(response)
        acc = accuracy(answers[:i+1], responses)
        print(f"Final Accuracy: {acc}%")

    # Print accuracy
    acc = accuracy(answers, responses)
    print(f"Final Accuracy: {acc}%")
        
def query_rag(prompt: str, ewha_collection, wiki_collection):
    # Load chat model
    chat_model = ChatUpstage(api_key=upstage_api_key)

    # Extract question from the prompt
    question = extract_question_queries(prompt)
    dense_vector, sparse_vector = generate_embeddings([question])
    
    ################# If context exists in ewha policy pdf #################
    if check_chat(prompt):
        # Perform hybrid search in ewha db
        try:
            results = hybrid_search(ewha_collection, dense_vector, sparse_vector)
        except Exception as e:
            print(f"❌ Error during hybrid search: {e}")
            return None
        
        processed_results = post_process_results(results) # Polishing results
        context_text = "\n\n---\n\n".join([res["chunk"] for res in processed_results])        
        prompt_formatted = ChatPromptTemplate.from_template(prompt_template_ewha).format(
            context=context_text, question=prompt
        )
        response = chat_model.invoke(prompt_formatted)

    ################# If context doesn't exist in ewha policy pdf, search wiki#################
    else:
        print(f"🔍 Missing context for '{prompt}'. Fetching data from Wikipedia...")
        info = extract_question_keywords(question)
        print(f"✅ Extracted info '{info}'")        

        keywords= info[0]["keywords"]
        for keyword in keywords:
            # Fetch context from Wikipedia
            pages = fetch_wiki_page(keyword)
            if pages:
                for page in pages:
                    chunks = sem_split_documents([page], THRESHOLD)
                    add_documents_to_milvus(wiki_collection, chunks)  
                    print("✅ Successfully fetched and added Wiki data.")
            else :
                print("😳 No context found on Wiki.")

        # Re-run hybrid search after updating collection
        try:
            results_wiki = hybrid_search(wiki_collection, dense_vector, sparse_vector)
            results_wiki = post_process_results(results_wiki)
            context_text_wiki = "\n\n---\n\n".join(list(map(lambda x: x["chunk"], results_wiki)))
            
        except Exception as e:
            print(f"❌ Error during hybrid search: {e}")    
            return None
        
        # Create prompt template
        problem_type = info[0]["problem_type"]
        print(problem_type)
        domain = classify_mmlu_domain(question)
        prompt_template_wiki = generate_chat_prompt(
            problem_type=problem_type, domain=domain)
        # prompt_new = PromptTemplate.from_template(prompt_template_wiki)
        # chain = prompt_new| chat_model 
        chain = prompt_template_wiki | chat_model 
        response = chain.invoke({"question": prompt, "context" : context_text_wiki})

    return response.content


def generate_embeddings(texts, device='cuda:3' if torch.cuda.is_available() else 'cpu'):
    """
    Generate dense and sparse embeddings for given texts.
    """
    try:
        embedding_model = model.hybrid.BGEM3EmbeddingFunction(use_fp16=True, device=device)
        embeddings = embedding_model(texts)
        dense_embeddings = np.array(embeddings["dense"], dtype=np.float32)
        sparse_embeddings = embeddings["sparse"]
        # a = {}
        # .toarray().astype(np.float32)
        # for idx, value in zip(sparse_embeddings.indices, sparse_embeddings.data):
        #     a[idx] = value
        return dense_embeddings, sparse_embeddings
    
    except KeyError as e:
        print(f"❌ Error generating embeddings: {e}")
        return None, None


def hybrid_search(collection, dense_vector, sparse_vector):
    """
    Perform a hybrid search by combining dense and sparse search results.
    """
    try:
        sparse_req = AnnSearchRequest(
            data=sparse_vector, anns_field="sparse_vector", param={"metric_type": "IP"}, limit=SEARCH_LIMIT
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
        print(f"Hybrid search returned {len(results[0]) if results else 0} results.")
        return results[0] if results else []
    
    except Exception as e:
        print(f"❌ Error during hybrid search: {e}")
        return []


def post_process_results(results):
    """
    Additional post-processing of results. 
    Removes duplicates, filters results based on similarity_threshold, and sort in descending order of similarity.
    """
    processed_results = []
    seen_chunks = set() # remove redundant data
    for result in results:
        try:
            chunk = result.entity.get("chunk")
            similarity_score = result.distance
            if chunk not in seen_chunks:
                seen_chunks.add(chunk)
                processed_results.append({
                    "chunk": chunk,
                    "similarity_score": similarity_score
                })
        except Exception as e:
            print(f"❌ Error processing result: {e}")
    processed_results.sort(key=lambda x: x["similarity_score"], reverse=True)
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
        print(f"✅ Added {len(texts)} documents to Milvus.")
    except Exception as e:
        print(f"❌ Failed to add documents to Milvus: {e}")
 


if __name__ == "__main__":
    main()