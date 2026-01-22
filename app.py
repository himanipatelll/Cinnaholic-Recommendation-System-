import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Cinnaholic Flavor Recommender", page_icon="🍰", layout="centered")

@st.cache_data
def load_menu(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"item_id", "name", "type", "flavor_tags"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"menu.csv is missing columns: {missing}")

    # Clean + normalize
    df["name"] = df["name"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower()

    df["flavor_tags"] = df["flavor_tags"].fillna("").astype(str)

    # Convert "salty, fruity" -> ["salty","fruity"] (trim spaces, drop empties)
    def parse_tags(s: str):
        tags = []
        for t in s.split(","):
            t = t.strip().lower()
            if t:  # removes empty tags caused by trailing commas like "nutty,"
                tags.append(t)
        return tags

    df["tags_list"] = df["flavor_tags"].apply(parse_tags)
    df["tags_str"] = df["tags_list"].apply(lambda tags: " ".join(tags))

    return df

@st.cache_resource
def build_vectorizer_and_matrix(text_series: pd.Series):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    matrix = vectorizer.fit_transform(text_series)
    return vectorizer, matrix

df = load_menu("menu.csv")
vectorizer, tag_matrix = build_vectorizer_and_matrix(df["tags_str"])

st.title("🍰 Cinnaholic Flavor Recommender")
st.caption("Type flavor vibes (e.g., fruity, creamy, chocolatey). Recommendations are based on menu tag similarity.")

preferences = st.text_input(
    "Enter your flavor preferences:",
    placeholder="fruity creamy chocolatey"
)

col1, col2 = st.columns(2)
with col1:
    top_n = st.slider("How many results per category?", 1, 10, 5)
with col2:
    allowed_types = ["roll", "frosting", "topping", "other"]
    types = st.multiselect("Categories:", allowed_types, default=["roll", "frosting", "topping"])

if preferences:
    pref_vector = vectorizer.transform([preferences.strip().lower()])
    similarity_scores = cosine_similarity(pref_vector, tag_matrix).flatten()

    recs = df.copy()
    recs["similarity"] = similarity_scores

    # Keep only chosen categories
    recs = recs[recs["type"].isin(types)]

    # Sort best matches
    recs = recs.sort_values("similarity", ascending=False)

    # Optionally hide zero-similarity items so it feels smarter
    recs_nonzero = recs[recs["similarity"] > 0]

    if recs_nonzero.empty:
        st.warning("No strong matches found. Try keywords like: cinnamon, caramel, nutty, tangy, rich, zesty.")
    else:
        st.subheader("Top Recommendations")

        for t in types:
            subset = recs_nonzero[recs_nonzero["type"] == t].head(top_n)
            st.markdown(f"### {t.title()}s")

            if subset.empty:
                st.write("No matches in this category.")
            else:
                st.dataframe(
                    subset[["item_id", "name", "flavor_tags", "similarity"]],
                    use_container_width=True
                )
