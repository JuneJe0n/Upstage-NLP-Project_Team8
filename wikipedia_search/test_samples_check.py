from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
from utils import read_data, get_choices, extract_question_queries, extract_question_keywords

import requests
from lxml import html
import json
from utils_wiki import remove_html_tags, get_wikipedia_articles, parse_wikipedia_page


prompts, answers = read_data('./test_samples.csv')
prompts = [prompts[9]]   # 10번만 적용

# 질문 원본
print('질문 원본','='*100,'\n')
print(prompts[0],'\n')


## 기존의 질문을 "문제유형:핵심질문"으로 변형한 것 -> langchain 사용
print('문제유형:핵심질문','='*100,'\n')
queries = extract_question_queries(prompts)
for query in queries:
    print(query,'\n')


## 질문에서 키워드를 추출한 것 -> langchain 사용
print('키워드','='*100,'\n')
information = extract_question_keywords(prompts)
for info in information:
    print(info['keywords'],'\n')


# 키워드로 Wikipedia에서 검색
keywords = information[0]['keywords']
articles = get_wikipedia_articles(keywords)


## 검색된 문서 출력
print('검색된 문서 정보','='*100, '\n')
article = articles[0]
print(f"Title: {article['title']}")
print(f"Snippet: {article['snippet']}")
print(f"URL: {article['url']}")
print()


# json 파일 구조 출력
print('document_structure.json 파일 확인','='*100, '\n')
url = article['url']
document_structure = parse_wikipedia_page(url)

file_path = './document_structure.json'
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(document_structure, f, ensure_ascii=False, indent=4)

