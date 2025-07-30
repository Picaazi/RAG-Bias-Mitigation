Pls do not commit API keys or data on github

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/api/)

> ** Security First**: This tool uses environment variables for API keys. Never commit API keys to version control!


### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
Add a .env file and insert the line
```
OPENAI_KEY=<your-api-key>
```


### 3. Run Queries
```bash
# Single query
python main.py

#then input your query and the script will decompose the question and check for bias
```
