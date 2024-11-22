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


# Get env
load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']
user_agent = os.environ['USER_AGENT']

# Get config
config_path = "/home/jiyoon/UpstageNLP_Team8/configs.yaml"
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
        

def query_rag(original_prompt:str):
    # Make embedding
    embedding_function = get_embedding_function()
    vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Context retrieval from the RAG database for the query
    results = vectorstore.similarity_search_with_score(original_prompt, k=10)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

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
            vectorstore.add_documents(chunks)
            print("👉added to database")

        # Context retrieval from the updated RAG database for the query
        results = vectorstore.similarity_search_with_score(question, k=10)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt = ChatPromptTemplate.from_template(prompt_template_wiki).format(context=context_text, question=question)
        response = model.invoke(prompt)  

    return response.content     

if __name__ == "__main__":
    main()