
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Locate the shared dataset relative to this Python file.
DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "books.csv"

REQUIRED_COLUMNS = ["book_id", "title", "author", "subject", "year"]


def search_books(query, top_k=5):
    """
    Search the library catalogue using combined-text TF-IDF
    and cosine similarity.

    Returns a list of dictionaries containing:
    book_id, title, author, subject, year, score.
    """

    # Handle empty queries and invalid result limits.
    if not isinstance(query, str) or not query.strip():
        return []

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        return []

    # Load the dataset prepared by Member 1.
    books = pd.read_csv(
        DATASET_PATH,
        dtype=str,
        keep_default_na=False
    )

    # Verify the agreed dataset columns.
    missing = set(REQUIRED_COLUMNS) - set(books.columns)

    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")

    if books.empty:
        return []

    # Handle missing values.
    books = books.fillna("")

    # Combine all searchable fields without assigning weights.
    combined_text = (
        books[["title", "author", "subject", "year"]]
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )

    # Avoid fitting TF-IDF on completely empty documents.
    if not combined_text.str.contains(r"\w", regex=True).any():
        return []

    # Convert book descriptions into TF-IDF vectors.
    vectorizer = TfidfVectorizer()
    book_vectors = vectorizer.fit_transform(combined_text)

    # Convert the query using the same vocabulary.
    query_vector = vectorizer.transform([query.strip()])

    # Calculate cosine similarity between query and books.
    scores = cosine_similarity(query_vector, book_vectors).ravel()

    # Rank by descending similarity; preserve dataset order for ties.
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: (-scores[i], i)
    )

    # Prepare the agreed output format.
    results = []

    for i in ranked_indices:
        if len(results) >= top_k or scores[i] <= 0:
            break

        book = books.iloc[i]

        results.append({
            "book_id": book["book_id"],
            "title": book["title"],
            "author": book["author"],
            "subject": book["subject"],
            "year": book["year"],
            "score": float(scores[i])
        })

    return results


# Allow independent testing without affecting Streamlit integration.
if __name__ == "__main__":

    print("Library Catalogue - Combined-Text Search")

    while True:
        query = input("\nEnter search query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        try:
            results = search_books(query)

            if not results:
                print("No matching books found.")

            for rank, book in enumerate(results, start=1):
                print(f"\n{rank}. {book['title']}")
                print(f"Book ID: {book['book_id']}")
                print(f"Author: {book['author']}")
                print(f"Subject: {book['subject']}")
                print(f"Year: {book['year']}")
                print(f"Relevance Score: {book['score']:.4f}")

        except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as error:
            print(f"Dataset error: {error}")
