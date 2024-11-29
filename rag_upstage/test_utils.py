from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
import pandas as pd
import wikipediaapi
import re
import os
import itertools
import tiktoken


load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']
user_agent = os.environ['USER_AGENT']


def load_pdf(data_path):
    pdf_loader = PyPDFDirectoryLoader(data_path)
    documents = pdf_loader.load()
    return documents


#  키워드의 조합을 리스트로 생성
def generate_subsets(keywords):
    subsets = []
    for i in range(len(keywords), 0, -1):  # 크기가 큰 것부터 검색하도록 순서 조정
        subsets.extend(itertools.combinations(keywords, i))
    return [' '.join(subset) for subset in subsets]


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
    keywords = keyword[0]['keywords']
    
    page_contents = []
    ###
    keywords = generate_subsets(keywords)
    ###
    for key in keywords:
        page = wiki_wiki.page(key)

        if page.exists():
            page_content = page.text
            document = Document(
                page_content=page_content,
                metadata={"title": page.title, "url": page.fullurl}
            )
            page_contents.append(document)
            print(f"✅ Wikipedia page fetched for '{key}'")

        else:
            print(f"❌ Wikipedia page not found for '{key}'")

    return page_contents


def extract_answer(response):
    """
    extracts the answer from the response using a regular expression.
    expected format: "(A) convolutional networks"

    if there are any answers formatted like the format, it returns None.
    """
    pattern = r"\([A-J]\)"  # Regular expression to find the first occurrence of (A), (B), (C), etc.
    match = re.search(pattern, response)

    if match:
        return match.group(0) # Return the full match (e.g., "(A)")
    else:
        return None


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
        print(response)

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


from langchain_experimental.text_splitter import SemanticChunker
from langchain_upstage.embeddings import UpstageEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(document: Document) -> list[Document]:
    """
    RecursiveCharacterTextSplitter를 사용하여 단일 문서를 분할.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    # 단일 문서의 텍스트를 분할
    chunks = text_splitter.split_text(document.page_content)

    # 각 chunk를 Document 형식으로 변환하여 크기를 metadata에 추가
    chunked_documents = [
        Document(page_content=chunk, metadata={**document.metadata, "chunk_size": len(chunk)})
        for chunk in chunks
    ]
        
    return chunked_documents


def sem_split_documents(documents: list[Document], threshold: str) -> list[Document]:
    """
    SemanticChunker로 문서를 분할하고, 크기가 큰 chunk는 다시 분할.
    """
    max_chunk_size = 1000
    max_token_size = 4000
    buffer_size = 500

    # SemanticChunker 설정
    sem_text_splitter = SemanticChunker(
        embeddings=UpstageEmbeddings(
            model="solar-embedding-1-large-query", 
            api_key=upstage_api_key
        ),
        buffer_size=buffer_size,
        breakpoint_threshold_type=threshold
    )

    # 처음 문서 분할
    chunks = sem_text_splitter.split_documents(documents)

    # 결과를 담을 리스트
    final_chunks = []

    # 각 chunk에 대해 크기를 확인하고, 1000 이상이면 다시 분할
    for chunk in chunks:
        # chunk, token 길이 측정
        tokenizer = tiktoken.get_encoding("cl100k_base")
        token_size = len(tokenizer.encode(chunk.page_content))
        chunk_size = len(chunk.page_content)
        print(f"현재 chunk 크기: {chunk_size}, 현재 token 크기: {token_size}")

        if chunk_size >= max_chunk_size:
            print(f"크기가 큰 chunk 발견: {len(chunk.page_content)}. RecursiveCharacterTextSplitter로 재분할합니다.")
            # RecursiveCharacterTextSplitter를 사용해 재분할
            sub_chunks = split_documents(chunk)
            print(f"분할된 chunk 크기: {[len(sub_chunk.page_content) for sub_chunk in sub_chunks]}")
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks





