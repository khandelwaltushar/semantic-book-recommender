import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "Default_image.jpg",
    books["large_thumbnail"]
)

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding=embeddings)


def retrieve_semantic_representation(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    # Search for the top 50 most similar books
    recs = db_books.similarity_search(query, k=initial_top_k)

    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_rec = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        books_rec = books_rec[books_rec["simple_categories"] == category].head(final_top_k)
    else:
        books_rec = books_rec.head(final_top_k)

    if tone == "Happy":
        books_rec.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_rec.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_rec.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        books_rec.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        books_rec.sort_values(by="sadness", ascending=False, inplace=True)

    return books_rec


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_representation(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 3:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"

        results.append((row["large_thumbnail"], caption))

    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book: ",
                                placeholder = "A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a Category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select a emotional tone:", value = "All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")

    output = gr.Gallery(label = "Recommended Books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)


if __name__ == "__main__":
    dashboard.launch()