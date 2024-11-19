from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
from utils import read_data, extract_question_queries

prompts, answers = read_data('./test_samples.csv')

'''
for p in prompts:
    print(p,'\n')
print('='*50)
'''


queries = extract_question_queries(prompts)

for query in queries:
    print(query)
