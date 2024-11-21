from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
from utils import read_data, get_choices, extract_question_queries, extract_question_keywords

import requests
from lxml import html
import json
from utils_wiki import remove_html_tags, get_wikipedia_articles, parse_wikipedia_page


prompts, answers = read_data('./test_samples.csv')
prompts = prompts[5:] # 6번부터 10번만 적용

# 질문 원본
print('='*100, '질문 원본')
for prompt in prompts:
    print(prompt,'\n')


## 기존의 질문을 "문제유형:핵심질문"으로 변형한 것 -> langchain 사용
print('='*100, '문제유형:핵심질문')
queries = extract_question_queries(prompts)
for query in queries:
    print(query,'\n')


## 객관식 보기만을 추출한 것
print('='*100, '객관식 보기만')
choices = get_choices(prompts)
for choice in choices:
    print(choice,'\n')


## 질문에서 키워드를 추출한 것 -> langchain 사용
print('='*100, '키워드')
information = extract_question_keywords(prompts)
for info in information:
    print(info['keywords'],'\n')




# 예시 키워드로 Wikipedia에서 검색
keywords = ['jurisprudence', 'law', 'philosophy']
articles = get_wikipedia_articles(keywords)

## 검색된 문서 출력
print('='*100, '검색된 문서 정보')
for article in articles:
    print(f"Title: {article['title']}")
    print(f"Snippet: {article['snippet']}")
    print(f"URL: {article['url']}")
    print()





# json 파일로 저장
print('='*100, 'document_structure.json 파일 확인')
url = 'https://en.wikipedia.org/?curid=26364'
document_structure = parse_wikipedia_page(url)

file_path = './document_structure.json'
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(document_structure, f, ensure_ascii=False, indent=4)
