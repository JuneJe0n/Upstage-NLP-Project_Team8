# Upstage-NLP-Project_Team8

## Install Dependencies
1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing ,`onnxruntime` through `pip install onnxruntime`. 
```python
conda install onnxruntime -c conda-forge
```

2. Now run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

## Environment Configuration Set Up

Before running the project, ensure you configure the required environment variables. Follow these steps to set up the `.env` file:

1. Create a `.env` file in the root directory of the project.

2. Add the following environment variables to the `.env` file:
   ```plaintext
   UPSTAGE_API_KEY=your API key for Upstage
   USER_AGENT=your custom user agent string for Wikipedia-API requests (e.g., MyProject/1.0 (your_email@example.com))


## Create database

Create the Milvus DB.

```python
python populate_milvus.py --reset
```

## Query the database

Query the Chroma DB.

```python
python query_data.py "영어 및 정보 등에 관하여 일정한 기준의 능력이나 자격을 취득한 경우 인정 받는 학점은 몇점인가?"
```
or use a test.csv file to query the Chroma DB.
```python
python query_multiplechoice.py
```
