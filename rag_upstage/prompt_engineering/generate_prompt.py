def generate_prompt(question, context, examples):
    """
    - 검색된 컨텍스트와 Few-shot Learning 예제를 활용하여 프롬프트를 생성.
    - 검색된 컨텍스트가 없을 경우 기본 예제를 기반으로 유사한 질문에 답변할 수 있도록 지원.
    - 단계적 사고 유도를 위한 문구 포함.

    Parameters:
    - question (str): 질문 텍스트.
    - context (str): 검색된 컨텍스트 텍스트.
    - examples (list of dict): Few-shot Learning 예제.
      각 예제는 {"question": 질문, "answer": 답변} 형식의 딕셔너리.

    Returns:
    - str: 생성된 프롬프트 텍스트.
    """

    # 예제 포맷팅
    formatted_examples = "\n\n".join(
        f"Example {i+1}:\nQ: {ex['question']}\nA: {ex['answer']}"
        for i, ex in enumerate(examples)
    )

    # 컨텍스트 길이 제한 (예: 최대 500자-> 이것도 수정해보면 좋을듯)
    max_context_length = 500
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [Context Truncated]"

    if context == "No relevant context found.":
        # 컨텍스트가 없을 경우
        return f"""
        {formatted_examples}
        Now, answer the following question without any context:
        Let's think step by step to find the most accurate answer.
        ---
        Q: {question}
        ---
        """
    else:
        # 검색된 컨텍스트가 있는 경우
        return f"""
        {formatted_examples}
        Now, answer the following question based on the provided context:
        Let's think step by step to find the most accurate answer.
        ---
        Q: {question}
        ---
        Context: {context}
        """

# 수정된 코드 실행 예시
few_shot_examples = [
    {"question": "What is the primary objective of Ewha Womans University?", "answer": "The primary objective is to educate and research profound theories."},
    {"question": "How many credits are required for graduation?", "answer": "A total of 129 credits are required for graduation."},
    {"question": "What is the maximum number of credits a student can take in a semester?", "answer": "A student can take up to 21 credits with approval."},
]

# 예제 입력
sample_question = "What is the maximum number of credits a student can register for in a semester?"
sample_context = "The maximum number of credits for undergraduate students per semester is 19. Students exceeding this must seek academic approval."

# 함수 실행
generated_prompt = generate_prompt(sample_question, sample_context, few_shot_examples)
print("Generated Prompt:\n", generated_prompt)