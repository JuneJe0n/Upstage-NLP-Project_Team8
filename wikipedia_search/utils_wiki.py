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
        url = f'https://en.wikipedia.org/?curid={pageid}'
        
        articles.append({
            "title": title,
            "snippet": snippet_text,
            "url": url
        })
    
    return articles



def remove_empty_subcontent(content):
    """빈 subcontent를 제거하는 재귀 함수"""
    for section in content[:]:  # 리스트 복사 후 탐색
        if 'subcontent' in section:
            # subcontent가 비어 있으면 제거
            if not section['subcontent']:
                del section['subcontent']
            else:
                # subcontent가 있으면 재귀적으로 탐색
                remove_empty_subcontent(section['subcontent'])

    # content에서 subcontent가 비어 있거나 필요한 항목 제거
    return [section for section in content if 'subcontent' not in section or section['subcontent']]



def parse_wikipedia_page(url):
    import requests
    from lxml import html

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
            text = element.text_content().strip()  # 텍스트 추출 및 공백 제거

            # 텍스트를 저장할 위치에 추가
            if current_h3:  # 현재 h3가 있으면 h3의 subcontent에 추가
                current_h3['text'] += text + " "  # 공백으로 텍스트 결합
            elif current_h2:  # 현재 h2가 있으면 h2의 text에 추가
                current_h2['text'] += text + " "  # 공백으로 텍스트 결합
            else:
                introduction += text + " "  # introduction에 추가

        elif is_target_class:
            # h2, h3, h4인 경우
            title = element.text_content().strip()  # 텍스트 추출
            title = title.replace('[edit]', '').strip()  # '[edit]' 제거 및 공백 제거

            if 'mw-heading2' in element.get('class', ''):  # h2
                current_h2 = {"title2": title, "text": '', "subcontent": []}
                content.append(current_h2)
                current_h3 = None  # 새로운 h2가 나타나면 h3 초기화

            elif 'mw-heading3' in element.get('class', '') and current_h2:  # h3
                current_h3 = {"title3": title, "text": '', "subcontent": []}
                current_h2['subcontent'].append(current_h3)

            elif 'mw-heading4' in element.get('class', '') and current_h3:  # h4
                h4_structure = {"title4": title, "text": '', "subcontent": []}
                current_h3['subcontent'].append(h4_structure)

    # 업데이트된 introduction 저장
    document_structure[0]["introduction"] = introduction.strip()  # 마지막에 공백 제거

    # 빈 subcontent 제거
    document_structure[0]["content"] = remove_empty_subcontent(content)

    return document_structure
