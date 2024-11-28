import os  # 파일 경로 조합을 위해 os 모듈 사용
import pdfplumber
import re
from util import read_test_data

DATA_PATH = "D:/UpstageNLP_test/rag_upstage/data/ewha.pdf"

def main():
    # Create CLI.
    pdf = extract_text_from_pdf(DATA_PATH)
    sections = split_sections_from_text(pdf)

def extract_text_from_pdf(file_path):
    """
    PDF 파일에서 텍스트를 추출합니다.
    :param file_path: PDF 파일 경로
    :return: 페이지별 텍스트 리스트
    """
    with pdfplumber.open(file_path) as pdf:
        pages_text = [page.extract_text() for page in pdf.pages]
    return pages_text

def split_sections_from_text(text_pages):
    """
    텍스트를 섹션별로 분리합니다.
    :param text_pages: 페이지별 텍스트 리스트
    :return: 조항별 데이터 리스트
    """
    sections = []
    current_section = {"조항": None, "내용": ""}
    for page in text_pages:
        lines = page.split("\n")
        for line in lines:
            match = re.match(r"제(\d+)조", line)  # 조항 번호 찾기
            if match:
                if current_section["조항"]:  # 현재 섹션 저장
                    sections.append(current_section)
                current_section = {"조항": line, "내용": ""}  # 새 섹션 시작
            else:
                current_section["내용"] += line + " "  # 섹션 내용 추가
    if current_section["조항"]:  # 마지막 섹션 저장
        sections.append(current_section)
    return sections

if __name__ == "__main__":
    main()