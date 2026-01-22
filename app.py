import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Cinnaholic Flavor Recommender", page_icon="🍰")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure column exists
    if "flavor_tags" not in df.columns:
        raise ValueError("menu.csv must contain a 'flavor_tags' column.")

    # Handle missing tags safely
    df["flavor_tags"] = df["flavor_tags"].fillna("").astype(str)
    df["flavor_tags_list"] = df["flavor_tags"].apply(
        lambda x: [tag.strip().lower() for tag in x.split(",") if tag.strip()]
    )
    df["tags_str"] = df["flavor_tags_list"].apply(lambda tags: " ".join(tags))
    return df

@st.cache_resource
def build_vectorizer_and_matrix(text_series: pd.Series):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )
    matrix = vectorizer.fit_transform(text_series)
    return vectorizer, matrix

df = load_data("menu.csv")
vectorizer, tag_matrix = build_vectorizer_and_matrix(df["tags_str"])

st.title("🍰 Cinnaholic Flavor Recommender")
st.caption("Type flavor vibes (e.g., fruity creamy chocolatey). The app recommends menu items based on tag similarity.")

preferences = st.text_input("Enter your flavor preferences:", placeholder="fruity creamy chocolatey")

col1, col2 = st.columns(2)
with col1:
    top_n = st.slider("How many recommendations per category?", 1, 10, 5)
with col2:
    types = st.multiselect("Show categories:", ["roll", "frosting", "topping"], default=["roll", "frosting", "topping"])

if preferences:
    pref_vector = vectorizer.transform([preferences.lower().strip()])
    similarity_scores = cosine_similarity(pref_vector, tag_matrix).flatten()

    recs = df.copy()
    recs["similarity"] = similarity_scores

    # Optional: remove items with zero similarity so results feel meaningful
    recs = recs[recs["similarity"] > 0].sort_values("similarity", ascending=False)

    if recs.empty:
        st.warning("No matches found. Try different keywords (e.g., cinnamon, caramel, nutty, vanilla).")
    else:
        st.subheader("Recommendations")

        for t in types:
            subset = recs[recs["type"].str.lower() == t].head(top_n)
            st.markdown(f"### Top Recommended {t.title()}s")
            if subset.empty:
                st.write("No matches in this category.")
            else:
                # Show name + similarity + tags for transparency
                st.dataframe(
                    subset[["name", "similarity", "flavor_tags"]],
                    use_container_width=True
                )
