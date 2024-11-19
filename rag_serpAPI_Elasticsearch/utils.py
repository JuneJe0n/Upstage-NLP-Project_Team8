import pandas as pd
import re
from collections import Counter
from elasticsearch import Elasticsearch
from serpapi import GoogleSearch
import requests
from urllib.parse import urlparse
from langchain.prompts import PromptTemplate
from langchain_upstage import ChatUpstage


def read_data(data_path):
    data = pd.read_csv(data_path)
    prompts = data['prompts']
    answers = data['answers']
    # returns two lists: prompts and answers
    return prompts, answers


def generate_responses(prompts, splits, chain):
    responses = []
    
    for prompt in prompts:
        combined_response = ""
        for split in splits:
            response = chain.invoke({"question": prompt, "context": split}).content
            combined_response += "-" + response + "\n"
        responses.append(combined_response)
    return responses


def extract_answer(response):
    """
    extracts the answer from the response using a regular expression.
    expected format: "[ANSWER]: (A) convolutional networks"

    if there are any answers formatted like the format, it returns None.
    """
    pattern = r"\[ANSWER\]:\s*\((A|B|C|D|E)\)"  # Regular expression to capture the answer letter and text
    match = re.search(pattern, response)

    if match:
        return match.group(1) # Extract the letter inside parentheses (e.g., A)
    else:
        return extract_again(response)


def select_answer(responses):
    """
    여러 개의 응답을 기반으로 가장 빈번하게 나온 답을 선택하는 함수
    :param responses: 응답 리스트 (응답은 각 question에 대해 여러 개일 수 있음)
    :return: 가장 빈번하게 나온 답 (A, B, C, D 중 하나)
    """
    answer_choices = []  # A, B, C, D 중 선택된 답을 저장할 리스트
    
    for response in responses:
        # 응답에서 답을 추출
        extracted_answer = extract_answer(response)
        if extracted_answer:
            answer_choices.append(extracted_answer)
    
    # 여러 개의 응답에서 가장 빈번하게 나온 답을 선택
    if answer_choices:
        most_common_answer = Counter(answer_choices).most_common(1)[0][0]
        return most_common_answer
    else:
        return "No valid answer"


def extract_again(response):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, response)
    if match:
        return match.group(0)
    else:
        return None
    

def calculate_accuracy(answers, responses):

    cnt = 0
    
    for answer, response in zip(answers, responses):
        print("-"*10)
        generated_answer = select_answer(response)
        print(response)
        # check
        if generated_answer:
            print(f"generated answer: {generated_answer}, answer: {answer}")
        else:
            print("extraction fail")
    
    
        if generated_answer == None:
            continue
        if generated_answer in answer:
            cnt += 1
    
    accuracy = (cnt / len(answers)) * 100 if len(answers) > 0 else 0
    print()
    print(f"acc: {accuracy}%")
    return accuracy


def extract_question_queries(prompts):

    UPSTAGE_API_KEY = "up_0L7RvF2YEPh96TxA3du1a1W225uIg"
    llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-1-mini-chat")

    # 프롬프트 템플릿 정의 (문제 유형과 핵심 질문을 자동 추출)
    prompt_template = PromptTemplate.from_template(
        """
        You are a question analyzer. Given the following multiple-choice question, please extract the problem type and core question.
        
        The problem type refers to the category or nature of the question (e.g., "Math Problem", "General Knowledge", "Legal Question", etc.).
        The core question is the main issue or query the question is asking.
        
        Provide the result in a single line, in the format. type: core question
        
        ---
        Question:
        {question_text}
        """
    )
    chain = prompt_template | llm
    query = []
    
    for prompt in prompts:
        input_dict = {"question_text": prompt}
        response = chain.invoke(input_dict).content  # chain을 재사용
        query.append(response.lstrip())  # 왼쪽 공백 제거
    
    return query


def configure_elasticsearch():
    # Elasticsearch 클라이언트 정의
    es_client = Elasticsearch(["http://localhost:9200"], request_timeout=120, max_retries=3)

    print("연결 상태 확인")
    # Elasticsearch 연결 상태 확인
    if es_client.ping():
        print("Elasticsearch is reachable")
        
        # 인덱스 삭제 (이미 존재할 경우)
        es_client.indices.delete(index="documents_index", ignore=[400, 404])  # 인덱스가 없을 경우 예외 처리
        
        # 새 인덱스 생성
        es_client.indices.create(index="documents_index", body={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            }
        })

        # 모든 인덱스에서 replica 설정을 0으로 변경
        es_client.indices.put_settings(index='*', body={
            "settings": {
                "number_of_replicas": 0
            }
        })
    else:
        print("Elasticsearch connection failed")

    print("\n클러스터 상태 확인")
    print(es_client.cluster.health())

    # es_client와 documents_index 반환
    return es_client, "documents_index"


def search_web(query, api_key):
    # SerpAPI로 웹 검색을 수행, Google 검색 엔진 사용
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": 10,
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    for i in range(len(results)):
        key = list(results.keys())[i]
        dictionary = list(results.values())[i]
        if key == "organic_results":
            print(dictionary)

            return(dictionary)



# 질문을 기반으로 Elasticsearch에서 문서 검색
def search_elastic(question, es_client, document_index):
    # Elasticsearch 쿼리 정의
    query = {
        "query": {
            "multi_match": {
                "query": question,  # 질문을 쿼리로 사용
                "fields": ["title^2", "snippet^1.5", "source", "content"],  # 검색할 필드들
                "fuzziness": "AUTO",  # 오타를 감안한 근사치 검색
                "operator": "or",  # 하나의 단어라도 포함되면 문서를 반환
                "minimum_should_match": "20%"  # 하나의 필드만 일치해도 결과에 포함
            }
        },
        "size": 10
    }
    
    # 검색 실행
    response = es_client.search(index=document_index, body=query)
    
    # 검색 결과에서 문서 추출
    relevant_documents = []
    for hit in response['hits']['hits']:
        relevant_documents.append({
            "title": hit["_source"]["title"],
            "snippet": hit["_source"]["snippet"],
            "link": hit["_source"]["link"],
            "source": hit["_source"]["source"]
        })
    
    return relevant_documents


# 검색된 문서들을 기반으로 context 생성
def create_context(relevant_documents):
    context = ""
    
    for doc in relevant_documents:
        context += f"Title: {doc['title']}\n"
        context += f"Snippet: {doc['snippet']}\n"
        context += f"Link: {doc['link']}\n"
        context += f"Source: {doc['source']}\n\n"
    
    return context
