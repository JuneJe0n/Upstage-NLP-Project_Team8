# Upstage NLP Project (Team8)

## Project Overview
### Project Objective

This project aims to build a Retrieval-Augmented Generation (RAG) system using the solar-1-mini-chat LLM by Upstage. The system enhances question-answering performance by integrating prompt engineering, data preprocessing, and external information retrieval. External knowledge is retrieved via the Wiki Search API to provide accurate and reliable answers. 

### Base Conditions

1. **Model**
- Backbone LLM: solar-1-mini-chat (provided by Upstage).
- Maximum token length: 32,768.

2. **Datasets**
- Ewha Academic Policies: Data from Ewha University Academic Regulations.
- MMLU-Pro: QA datasets spanning Law, Psychology, Business, Philosophy, and History. 

3. **Key Rules**
- No Fine-Tuning: Model retraining is not allowed; only prompt engineering and external retrieval are used.
- External Retrieval: Information is retrieved via Wiki Search API.
- Token Constraints: Responses must be concise and fit within the token limit.

## Project Settings
### Requirements

A suitable conda enviroment named `nlp` can be created and activated with:
```python
conda create --name nlp python=3.12.12
conda activate nlp
```

To get started, install the required python packages into you `nlp` enviroment
```python
conda install onnxruntime -c conda-forge
pip install -r requirements.txt
```

### Environment Configuration Setup

Before running the project, ensure you configure the required environment variables. Follow these steps to set up the `.env` file:

1. Create a `.env` file in the root directory of the project.

2. Add the following environment variables to the `.env` file:
```plaintext
UPSTAGE_API_KEY=your API key for Upstage
USER_AGENT=your custom user agent string for Wikipedia-API requests (e.g., MyProject/1.0 (your_email@example.com))
```


### Create Database

1. To reset the database and start fresh, run this command
```python
python populate_milvus.py --reset
```
2. Create Milvus database
 ```python
python populate_milvus.py
```
3. Create ewha_milvus database, contains only ewha regarded data
```python
python populate_ewha_milvus.py
```

### Query the Database

Query the Chroma DB.

```python
python query_data.py "영어 및 정보 등에 관하여 일정한 기준의 능력이나 자격을 취득한 경우 인정 받는 학점은 몇점인가?"
```
or use a test.csv file to query the Chroma DB.
```python
python query_multiplechoice.py
```

## Project Detail
### Model and API setup
### Project Pipeline

## Contributions
1. **Jiyoon Jeon (전지윤)**
- Designed pipeline and created baseline code
- Gathered and polished every code written by team members 
   - Had to study every method used in the project, and invest a lot of time and effort in merging codes with different styles and structures.
   - Especially had a very hard time debugging, and selecting best methods and structures.
- Contributed a lot in enhancing performance
   - Found out that using ChatPromptTemplate and 5 shot prompting is very important, and polished every single prompt in order to improve performance.
   - Proposed the idea of separating the database and implemented it, which was a big help to the performance scores.

2. **(박경숙)**
- Created keywords extract prompt + used keyword subsets for wikipedia search
- Implemented semantic splitting using 4 ways of threshold types
- Implemented multichoice answer extraction

3. **Dain Han (한다인)**
- Implemented Wikipedia page fetching using the wikipediaapi library.
- Built a database using Milvus: created and managed data collections, including HNSW search index.
- Developed a hybrid search algorithm combining sparse and dense vector-based search functionalities.
- Implemented post-processing logic to calculate and filter similarity scores based on search results.
- Generated 50 questions for performance evaluation based on the Ewha Womans University regulations

4. **(허예원)**
- PDF Preprocessing: Implemented techniques for efficient data extraction and structuring from PDF files.
- Prompt Engineering
   - Designed domain-specific templates for Ewha academic regulations and MMLU-Pro datasets.
   - Integrated few-shot learning examples and fallback logic for accurate responses with limited context.
- Dataset Creation: Curated and organized datasets for Ewha academic policies and MMLU-Pro domains.

5. **Jungmin Byeon (변정민)**
- Proposed idea of contexual retrieval method, implementing sparse and dense vector retrieval splitting methods (semantic, recursive)
- ppt, presentation
