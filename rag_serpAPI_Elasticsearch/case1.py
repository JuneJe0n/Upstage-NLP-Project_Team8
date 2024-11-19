#### SerpAPI + ElasticSearch + solar-1-mini-chat
from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
from langchain.vectorstores import ElasticsearchStore
from langchain_upstage import UpstageEmbeddings
from langchain.schema import Document
from utils import read_data, configure_elasticsearch, search_web, calculate_accuracy, search_elastic, create_context


UPSTAGE_API_KEY = "up_0L7RvF2YEPh96TxA3du1a1W225uIg"
SERPAPI_KEY = "7babe55c31b46aaebb2b10f7db960bfceb3e63f1b23b9e6a038faf0e9ee07e9e"
prompts, answers = read_data('./test_samples.csv')

#Elasticsearch 서버가 정상적으로 실행 중인지 확인
es_client, document_index = configure_elasticsearch()

print("\n클러스터 상태 확인")
print(es_client.cluster.health())


# Upstage 설정
llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-1-mini-chat")

# 프롬프트 템플릿 정의
prompt_template = PromptTemplate.from_template(
    """
    Please provide most correct answer from the following context.
    If the answer is not present in the context, please write "The information is not present in the context."
    ---
    Question: {question}
    ---
    Context: {context}
    """
)
chain = prompt_template | llm

# Embedding 설정
# embeddings = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large")

responses1 = []

# 질문 기반 웹 검색 및 응답 생성
for prompt in prompts:
    # 웹 검색 결과
    search_results = search_web(prompt, SERPAPI_KEY)

    # 검색 결과가 None 또는 비어 있으면 처리
    if search_results is None or len(search_results) == 0:
        responses1.append("No search results found.")
        continue
    
    # 관련 문서들을 ElasticSearch에 색인화
    for idx, result in enumerate(search_results):
        if 'snippet' in result:
            doc = {
                "title": result['title'],
                "snippet": result['snippet'],
                "link": result['link'],
                "source": result['source']
            }
            es_client.index(index=document_index, id=idx, body=doc)

    
    # Elasticsearch에서 관련 문서 검색
    relevant_documents = search_elastic(prompt, es_client, document_index)

    # 검색된 관련 문서 출력
    for doc in relevant_documents:
        print(doc)

    # context 생성
    context = create_context(relevant_documents)
    

    # 검색된 문서에서 질문에 대한 응답 생성
    input_dict = {"question": prompt, "context": context}
    response = chain.invoke(input_dict).content
    
    responses1.append(response)

calculate_accuracy(answers, responses1)
