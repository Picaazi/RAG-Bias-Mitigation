# 🤖 OpenAI Query Tool

**Simple command-line tool for querying OpenAI API with Python.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/api/)

> **🔒 Security First**: This tool uses environment variables for API keys. Never commit API keys to version control!

## ✨ Features

- 🤖 **Direct OpenAI Integration** - Query GPT models directly from command line
- 💬 **Interactive Mode** - Chat-like interface for multiple queries
- 📊 **Usage Tracking** - Monitor token usage and costs
- ⚡ **Simple CLI** - Easy-to-use command-line interface
- 🔐 **Secure** - Uses environment variables for API key management

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
```bash
export OPENAI_KEY='your-openai-api-key-here'
```

> 📖 **Get API Key**: [OpenAI Platform](https://platform.openai.com/api-keys)

### 3. Run Queries
```bash
# Single query
python main.py query "What is artificial intelligence?"

# Interactive mode
python main.py interactive

# Run examples
python examples.py
```
