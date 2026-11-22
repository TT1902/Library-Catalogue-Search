from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "books.csv"
REQUIRED_COLUMNS = ["book_id", "title", "author", "subject", "year"]
FIELD_WEIGHTS = {"title": 0.5, "author": 0.2, "subject": 0.3}


def _field_similarity(query, values):
    """Return one TF-IDF cosine similarity per row for a single field."""
    documents = values.fillna("").astype(str)

    if not documents.str.contains(r"\w", regex=True).any():
        return [0.0] * len(documents)

    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform(documents)
    except ValueError:  # For example, an entirely blank field.
        return [0.0] * len(documents)

    query_vector = vectorizer.transform([query])
    return cosine_similarity(query_vector, vectors).ravel().tolist()


def weighted_search(query, top_k=5):
    """
    Search title, author, and subject independently using TF-IDF and cosine
    similarity, then combine scores with weights 0.5, 0.2, and 0.3.

    Returns dictionaries with exactly: book_id, title, author, subject,
    year, score.
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        return []

    books = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)
    missing = set(REQUIRED_COLUMNS) - set(books.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    if books.empty:
        return []

    books = books.fillna("")
    cleaned_query = query.strip()
    field_scores = {
        field: _field_similarity(cleaned_query, books[field])
        for field in FIELD_WEIGHTS
    }
    scores = [
        sum(FIELD_WEIGHTS[field] * field_scores[field][index]
            for field in FIELD_WEIGHTS)
        for index in range(len(books))
    ]
    ranked_indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))

    results = []
    for index in ranked_indices:
        if len(results) >= top_k or scores[index] <= 0:
            break
        book = books.iloc[index]
        results.append({
            "book_id": book["book_id"],
            "title": book["title"],
            "author": book["author"],
            "subject": book["subject"],
            "year": book["year"],
            "score": float(scores[index]),
        })
    return results


if __name__ == "__main__":
    print("Library Catalogue - Weighted Field-Based Search")
    while True:
        query = input("\nEnter search query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        try:
            results = weighted_search(query)
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
