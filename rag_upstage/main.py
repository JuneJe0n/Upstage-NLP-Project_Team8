import os
import re
import yaml
import torch
import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
import pandas as pd
from util import (read_test_data, split_documents,
                  get_embedding_function, extract_question_queries, extract_question_keywords, fetch_wiki_page,
                  detect_missing_context, accuracy, extract_answer, extract_again)

from langchain.retrievers import BM25Retriever, EnsembleRetriever ###
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# Get env
load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']
user_agent = os.environ['USER_AGENT']

# Get config
config_path = "C:/Users/jungmin/Desktop/UpstageNLP_Team8/rag_upstage/configs.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader = yaml.FullLoader)

chroma_path = config["CHROMA_PATH"]
test_path = config["TEST_PATH"]
prompt_template = config["PROMPT_TEMPLATE"]
prompt_template_wiki = config["PROMPT_TEMPLATE_WIKI"]

def main():
    prompts, answers = read_test_data(test_path)

    responses = []

    for original_prompt in prompts:
        # extract question of prompt
        response = query_rag(original_prompt)
        responses.append(response)
    
    acc = accuracy(answers, responses)
    print(f"Final Accuracy: {acc}%")

def load_chroma_db():
    embedding_function = get_embedding_function()
    chroma_db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    return chroma_db ###db초기화

def create_chroma_retriever(chroma_db):
    chroma_retriever = chroma_db.as_retriever(search_kwargs={"k": 20})  # 검색할 상위 k개 설정
    return chroma_retriever

def create_bm25_retriever(chunks):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10  # 상위 10개 반환
    return bm25_retriever

def create_ensemble_retriever(bm25_retriever, chroma_retriever):
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6]  # BM25: 0.4, Chroma: 0.6
    )
    return ensemble_retriever

def create_reranker(ensemble_retriever):
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=10)  # 상위 10개 재랭크
    reranker = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    return reranker

def query_rag(original_prompt:str,reranker: ContextualCompressionRetriever,chroma_db):
    
    print("🔍 Retrieving and Reranking context from retriever...")
    results = reranker.get_relevant_documents(original_prompt)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    
    # Generating the initial prompt
    prompt = ChatPromptTemplate.from_template(prompt_template).format(context=context_text, question=original_prompt)
    model = ChatUpstage(api_key=upstage_api_key)
    response = model.invoke(prompt)

    # Fetch data from Wikipedia if the context is not in database
    if detect_missing_context(response.content):
        print(f"🔍 Missing context for \n'{original_prompt}' \ndetected. Fetching data from Wikipedia...")

        # extract question from original prompt
        question = extract_question_queries(original_prompt)

        # Extract keyword from question
        keyword = extract_question_keywords(question)
        print(f"✅Extracted keyword '{keyword}' from {question}")

        # Add wiki page to vectorstore
        pages = fetch_wiki_page(keyword)
        for page in pages:
            chunks = split_documents(pages)    
            chroma_db.add_documents(chunks)
            print("👉added to database")

        # Context retrieval from the updated RAG database for the query
        results = reranker.get_relevant_documents(original_prompt)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        prompt = ChatPromptTemplate.from_template(prompt_template_wiki).format(context=context_text, question=question)
        response = model.invoke(prompt)  

    return response.content     

if __name__ == "__main__":
    chroma_db = load_chroma_db()
    bm25_retriever = create_bm25_retriever(chunks=load_chunks_from_database())
    ensemble_retriever = create_ensemble_retriever(bm25_retriever, chroma_db.as_retriever())
    reranker = create_reranker(ensemble_retriever)

    prompts, answers = read_test_data(test_path)
    responses = [query_rag(prompt, reranker, chroma_db) for prompt in prompts]
    acc = accuracy(answers, responses)
    print(f"Final Accuracy: {acc}%")