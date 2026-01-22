# Cinnaholic-Recommendation-System-

Interactive content-based recommendation app that suggests rolls, frostings, and toppings based on user flavor preferences.

## How it works
- Parses flavor tags from `menu.csv`
- Converts tags into TF-IDF vectors
- Computes cosine similarity between user input and items
- Displays top matches by category

## Tech Stack
- Python
- pandas
- scikit-learn (TF-IDF, cosine similarity)
- Streamlit

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
