import requests
from bs4 import BeautifulSoup
from lxml import html
import json


# HTML 태그를 제거하고 순수 텍스트만 추출하는 함수
def remove_html_tags(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()


# Wikipedia API에서 결과를 검색하고 순수 텍스트로 비교하는 함수
def get_wikipedia_articles(keywords):
    # Wikipedia API 호출 예시
    endpoint = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": " ".join(keywords),
        "utf8": 1
    }
    
    response = requests.get(endpoint, params=params)
    data = response.json()
    
    articles = []
    
    # 각 검색된 항목에 대해 페이지 본문을 가져오기
    for result in data["query"]["search"]:
        title = result["title"]
        snippet = result["snippet"]
        # HTML 태그를 제거하여 순수 텍스트만 비교
        snippet_text = remove_html_tags(snippet)
        pageid = result["pageid"]
        
        # 각 페이지 ID로 url 가져오기
        url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=revisions&rvprop=content&rvslots=main&pageids={pageid}"
        
        articles.append({
            "title": title,
            "snippet": snippet_text,
            "url": url
        })
    
    return articles



def parse_wikipedia_page(url):

    response = requests.get(url)

    # HTML 파싱 (lxml을 사용하여 XPath 지원)
    tree = html.fromstring(response.text)
    # 본문 영역 선택
    contents = tree.xpath('//*[@id="mw-content-text"]/div[1]')[0]  # 첫 번째 div 선택

    # 등장 순서대로 p, ul 태그 또는 클래스가 'mw-heading mw-heading2', 'mw-heading mw-heading3', 'mw-heading mw-heading4'인 요소 수집
    elements = contents.xpath('./*')  # contents의 직계 자식 요소 가져오기

    # 결과 저장을 위한 리스트
    document_structure = []

    # 현재 추적 중인 계층
    current_h2 = None
    current_h3 = None
    introduction = ""
    content = []

    # 첫 번째 h1을 찾아 title1으로 저장
    first_h1 = tree.xpath('//h1')[0]
    title1 = first_h1.text_content().strip()

    # 문서 구조 시작
    document_structure.append({
        "title1": title1,  # h1 태그의 텍스트를 title1로 저장
        "introduction": introduction,  # 첫 번째 h2 이전의 내용
        "content": content  # 각 h2, h3, h4에 대한 내용
    })

    # 본문 내용 수집
    for element in elements:
        # 조건: p, ul 태그이거나, 클래스가 'mw-heading mw-heading2', 'mw-heading mw-heading3', 'mw-heading mw-heading4'
        is_target_tag = element.tag in ['p', 'ul']
        is_target_class = any(class_name in element.get('class', '') for class_name in ['mw-heading mw-heading2', 'mw-heading mw-heading3', 'mw-heading mw-heading4'])

        if is_target_tag:
            # p, ul 태그인 경우 텍스트 수집
            tag_name = element.tag
            text = element.text_content().strip()  # 텍스트 추출 및 공백 제거

            # 텍스트를 저장할 위치에 추가
            if current_h3:  # 현재 h3가 있으면 h3의 subcontent에 추가
                current_h3['subcontent'].append({"text": text})    # "tag": tag_name,
            elif current_h2:  # 현재 h2가 있으면 h2의 subcontent에 추가
                current_h2['subcontent'].append({"text": text})
            else:
                introduction += text + " "  # 공백으로 텍스트 결합

            # print(f"Tag: {tag_name}")
            # print(f"Text: {text}")
            # print('-' * 50)  # 구분선

        elif is_target_class:
            # h2, h3, h4인 경우
            if 'mw-heading2' in element.get('class', ''):  # h2
                tag_name = element.tag
                title2 = element.text_content().strip()  # 텍스트 추출
                current_h2 = {"title2": title2, "subcontent": []}
                content.append(current_h2)

            elif 'mw-heading3' in element.get('class', ''):  # h3
                tag_name = element.tag
                title3 = element.text_content().strip()  # 텍스트 추출
                current_h3 = {"title3": title3, "subcontent": []}
                current_h2['subcontent'].append(current_h3)

            elif 'mw-heading4' in element.get('class', ''):  # h4
                tag_name = element.tag
                title4 = element.text_content().strip()  # 텍스트 추출
                current_h3['subcontent'].append({"title4": title4, "subcontent": []})

            # print(f"Tag: {tag_name}")
            # print(f"Text: {title2 if 'title2' in locals() else title3 if 'title3' in locals() else title4}")
            # print('-' * 50) 

    # 업데이트된 introduction 저장
    document_structure[0]["introduction"] = introduction.strip()  # 마지막에 공백 제거
    return document_structure