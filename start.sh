#!/bin/bash
# Ensure the model is installed
python -m spacy download de_core_news_sm

# Start Streamlit app
streamlit run stat_bot.py --server.port=$PORT --server.address=0.0.0.0
