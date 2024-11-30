from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema.output_parser import StrOutputParser
import pandas as pd
import wikipediaapi
import tiktoken
import re
import os
import torch
from tqdm import tqdm


load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']
user_agent = os.environ['USER_AGENT']

def load_pdf(data_path):
    pdf_loader = PyPDFDirectoryLoader(data_path)
    return pdf_loader.load()

def read_test_data(data_path):
    data = pd.read_csv(data_path)
    prompts = data['prompts']
    answers = data['answers']
    return prompts, answers

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)

    # add chunk size in metadata
    for chunk in chunks:
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        
    return chunks


def extract_question_queries(original_prompt):
    llm = ChatUpstage(api_key=upstage_api_key, model="solar-1-mini-chat")

    # 프롬프트 템플릿 정의 (문제 유형과 핵심 질문을 자동 추출)
    prompt_template = PromptTemplate.from_template(
        """
        You are a question analyzer. Given the following multiple-choice question, please extract the problem type and core question.
        
        The problem type refers to the category or nature of the question (e.g., "Math Problem", "General Knowledge", "Legal Question", etc.).
        The core question is the main issue or query the question is asking.
        
        Provide the result in a single line, in the format. problem type: core question
        ---
        Question:
        {question_text}
        """
    )
    chain = prompt_template | llm
    query = []

    input_dict = {"question_text": original_prompt}
    response = chain.invoke(input_dict).content  # chain을 재사용 
    
    return response

def check_question(question):
    llm = ChatUpstage(api_key=upstage_api_key, model="solar-1-mini-chat")
    prompt_template = PromptTemplate.from_template(
    """
    You are a specialized assistant trained to determine if a question is related to Ewha Womans University's policies. 
    Answer True if the question relates to Ewha's policies and False otherwise. 
    Generate your answers carefully based on your knowledge about school policies and the examples below.
    Provide no explanations or additional information—just "True" or "False". This is very important. 
    Never return answers other than "True" or "False".
    ---
    Examples:
    Question: 이화여자대학교의 학년 시작 날짜는 언제입니까?
    Answer: True

    Question: 학사학위과정과 대학원 교육을 연계하는 과정을 무엇이라 합니까?
    Answer: True

    Question: What are the tallest trees on Earth?
    Answer: False

    Question: How much interest will George pay on a 6-month loan of $300 at 8% interest?
    Answer: False
    ---
    Question: {question}
    Answer: 
    """)
    chain = prompt_template | llm | StrOutputParser()
    input_dict = {"question": question}
    response = chain.invoke(input_dict)
    flag = re.findall(r'True|False', response)
    if not flag:
        response = False
    else:
        response = eval(flag[0])
    return response

def check_chat(question):
    llm = ChatUpstage(api_key=upstage_api_key, model="solar-1-mini-chat")
    template = ChatPromptTemplate.from_messages([
    ("system","""You are a specialized assistant trained to determine if a question is related to school regulations. 
    Response True if the question relates to school regulations and False otherwise. 
    Generate your responses carefully based on your knowledge about school regulations and the examples below.
    If the question is in Korean rather than English, it is highly probable that it is related to school regulations. 
    Provide no explanations or additional information—just "True" or "False". This is very important. 
    Never return responses other than "True" or "False".
    ---
    Example 1:
    QUESTION2) LMS 시스템의 초기 비밀번호는 무엇으로 설정되어 있습니까?
    (A) 사용자가 직접 설정
    (B) 본인 생년월일
    (C) 0000
    (D) 임의 지정된 숫자
    Response) True

    Example 2:
    QUESTION3) 비대면 수업을 위해 필요한 필수 장비는 무엇입니까?
    (A) 컴퓨터, 카메라, 마이크
    (B) 필기구와 종이
    (C) USB 드라이브
    (D) 추가 모니터
    Response) True

    Example 3:
    QUESTION7) Ms. Chen purchased a used car, worth $1,650, on the installment plan, paying $50 down and $1,840 in monthly installment payments over a period of two years. What annual interest rate did she pay?
    (A) 10%
    (B) 17.5%
    (C) 15.2%
    (D) 13.3%
    (E) 20%
    (F) 19.8%
    (G) 18%
    (H) 16%
    (I) 12%
    (J) 14.4%
    Response) False

    Example 4:
    QUESTION10) Which of the following Enlightenment philosophes designed a system of checks and balances for government to avoid abuses of power?
    (A) Thomas Hobbes
    (B) Jean Jacques Rousseau
    (C) Baron Montesquieu
    (D) Voltaire
    (E) Denis Diderot
    (F) Immanuel Kant
    (G) David Hume
    (H) Mary Wollstonecraft
    (I) Adam Smith
    (J) John Locke
    Response) False
    
    Example 5:
    QUESTION15) 출석 인정이 가능한 결석 사유는 무엇입니까?
    (A) 질병
    (B) 직계존비속의 사망
    (C) 국제대회 참가
    (D) 모두 해당
    Response) True
     
    """),
    ("human",
     """Question: {question}
    Response) """)
    ])
    chain = template | llm | StrOutputParser()
    input_dict = {"question": question}
    response = chain.invoke(input_dict)
    print(response)
    
    flag = re.findall(r'True|False', response)
    if not flag:
        response = False
    else:
        response = eval(flag[0])
    return response


def detect_missing_context(response_content: str) -> bool:
    """
    Check if the response contains the exact phrase:
    'The information is not present in the context.'
    
    Parameters:
        response_content (str): The response text to check.

    Returns:
        bool: True if the exact phrase is present, otherwise False.
    """
    return "The information is not present in the context." in response_content


def extract_question_keywords(question):
    llm = ChatUpstage(api_key=upstage_api_key, model="solar-1-mini-chat")

    # 프롬프트 템플릿 정의
    prompt_template = PromptTemplate.from_template(
        """
        You are a question analyzer. For the following multiple-choice question, perform the following tasks:
        
        1. Identify the problem type among ("Law", "Psychology", "Business", "Philosophy", "History")
        2. Extract the core question being asked.
        3. Extract the most relevant keywords for search (3-5 keywords) to answer the question effectively.
        4. The problem type, core_question, keywords must be all in English, not Korean. This is important.

        Provide the output in JSON format:
        {{
            "problem_type": "[problem type]",
            "core_question": "[core question]",
            "keywords": ["keyword1", "keyword2", "keyword3", ...]
        }}

        ---
        Question:
        {question_text}
        """
    )
    
    # 체인 생성
    chain = prompt_template | llm
    results = []
    input_dict = {"question_text": question}
    response = chain.invoke(input_dict).content.strip()

    try:
        # 응답을 딕셔너리로 변환
            result_dict = eval(response)  # JSON 형식으로 변환
            results.append(result_dict)
    except Exception as e:
        print(f"Error parsing response for prompt: {question}\nError: {e}")

    return results


def fetch_wiki_page(keyword, lang="en"):
    """
    Fetch a snippet from Wikipedia based on the query and summarize it using ChatUpstage.
    
    Parameters:
        keyword (str): The keyword to search in Wikipedia.
        lang (str): The language code for Wikipedia (default: 'en').
    
    Returns:
        str: A summarized text of the Wikipedia content if the page exists, otherwise None.
    """

    wiki_wiki = wikipediaapi.Wikipedia(user_agent, lang)
    
    
    page_contents = []
    
    page = wiki_wiki.page(keyword)

    if page.exists():
        page_content = page.text
        document = Document(
            page_content=page_content,
            metadata={"title": page.title, "url": page.fullurl}
        )
        page_contents.append(document)
        print(f"✅ Wikipedia page fetched for '{keyword}'")
        return page_contents

    else:
        print(f"❌ Wikipedia page not found for '{keyword}'")
        return None

    

def sem_split_documents(documents: list[Document], threshold: str) -> list[Document]:
    """
    SemanticChunker로 문서를 분할하고, 크기가 큰 chunk는 다시 분할.
    """
    max_chunk_size = 1000
    buffer_size = 500

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    # SemanticChunker 설정
    sem_text_splitter = SemanticChunker(
        embeddings=UpstageEmbeddings(
            model="solar-embedding-1-large-query", 
            api_key=upstage_api_key
        ),
        buffer_size=buffer_size,
        breakpoint_threshold_type=threshold
    )
    
    chunks = text_splitter.split_documents(documents)


    sem_chunks = sem_text_splitter.split_documents(chunks)

    final_sem_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    final_chunks = final_sem_splitter.split_documents(sem_chunks)
        

    return final_chunks


            


def accuracy(answers, responses):
    """
    Calculates the accuracy of the generated answers.
    
    Parameters:
        answers (list): The list of correct answers.
        responses (list): The list of generated responses.
    Returns:
        float: The accuracy percentage.
    """
    cnt = 0

    for answer, response in zip(answers, responses):
        print("-" * 10)
        generated_answer = extract_answer(response)

        # check
        if generated_answer:
            print(f"generated answer: {generated_answer}, answer: {answer}")
        else:
            print("extraction fail")

        if generated_answer is None:
            continue
        if generated_answer in answer:
            cnt += 1

    acc = (cnt / len(answers)) * 100

    return acc


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

def extract_again(response):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, response)
    if match:
        return match.group(0)
    else:
        return None






