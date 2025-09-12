#!/bin/bash

echo " Starting environment preparing"
source mskh_env/bin/activate
#git clone https://github.com/danielkty/debiasing-rag
#git clone https://github.com/helen-jiahe-zhao/BibleQA.git
#git clone https://github.com/nyu-mll/BBQ
export OPENAI_KEY=$(cat openai_key.txt)

# python3 -m venv mskh_env

# pip3 install -r requirements.txt
echo "Installing libs"

python3 environment_check.py