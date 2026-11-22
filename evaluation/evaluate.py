import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.combined_search import search_books
from retrieval.weighted_search import weighted_search

EVALUATION_DIR = Path(__file__).resolve().parent
QUERIES_PATH = EVALUATION_DIR / "queries.csv"
JUDGEMENTS_PATH = EVALUATION_DIR / "relevance_judgements.csv"
K = 5


def precision_at_k(results, relevant_ids, k=K):
    """Precision@k uses k as its denominator, including when fewer books return."""
    returned_ids = [str(book["book_id"]) for book in results[:k]]
    return sum(book_id in relevant_ids for book_id in returned_ids) / k


def average_precision_at_k(results, relevant_ids, k=K):
    """Average precision at k, normalized by min(relevant count, k)."""
    denominator = min(len(relevant_ids), k)
    if denominator == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, book in enumerate(results[:k], start=1):
        if str(book["book_id"]) in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / denominator


def load_relevance(path=JUDGEMENTS_PATH):
    judgements = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"query_id", "book_id"}
    missing = required - set(judgements.columns)
    if missing:
        raise ValueError(f"Missing relevance judgement columns: {sorted(missing)}")
    return {
        str(query_id): set(group["book_id"].astype(str))
        for query_id, group in judgements.groupby("query_id")
    }


def evaluate():
    queries = pd.read_csv(QUERIES_PATH, dtype=str, keep_default_na=False)
    missing = {"query_id", "query"} - set(queries.columns)
    if missing:
        raise ValueError(f"Missing query columns: {sorted(missing)}")
    relevance = load_relevance()

    rows = []
    outcomes = []
    for _, row in queries.iterrows():
        query_id = str(row["query_id"])
        query = row["query"]
        relevant_ids = relevance.get(query_id, set())
        combined = search_books(query, top_k=K)
        weighted = weighted_search(query, top_k=K)

        combined_p = precision_at_k(combined, relevant_ids)
        weighted_p = precision_at_k(weighted, relevant_ids)
        combined_ap = average_precision_at_k(combined, relevant_ids)
        weighted_ap = average_precision_at_k(weighted, relevant_ids)

        rows.append({
            "query_id": query_id,
            "query": query,
            "combined_precision@5": combined_p,
            "weighted_precision@5": weighted_p,
            "combined_AP@5": combined_ap,
            "weighted_AP@5": weighted_ap,
        })
        outcomes.append((query, relevant_ids, combined, weighted,
                         combined_p, weighted_p, combined_ap, weighted_ap))

    table = pd.DataFrame(rows)
    print("Per-query comparison")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if not table.empty:
        print("\nMean scores")
        print(table[["combined_precision@5", "weighted_precision@5",
                     "combined_AP@5", "weighted_AP@5"]]
              .mean().to_string(float_format=lambda value: f"{value:.3f}"))

    improved = next((item for item in outcomes if item[5] > item[4]), None)
    if improved:
        query, relevant, combined, weighted, cp, wp, _, _ = improved
        print("\nExample where weighted search improves")
        print(f"Query: {query!r}; Precision@5 rises from {cp:.3f} to {wp:.3f}.")
        print("The separate field scores let title, author, and subject matches "
              "contribute according to the agreed weights.")
        print(f"Relevant IDs: {sorted(relevant)}")
        print(f"Combined top IDs: {[book['book_id'] for book in combined[:K]]}")
        print(f"Weighted top IDs: {[book['book_id'] for book in weighted[:K]]}")
    else:
        print("\nNo query has higher weighted Precision@5 in this run, so there is "
              "no observed improvement example in the current judged set.")

    no_better = next((item for item in outcomes if item[5] <= item[4]), None)
    if no_better:
        query, relevant, combined, weighted, cp, wp, _, _ = no_better
        print("\nExample where weighted search does not improve")
        print(f"Query: {query!r}; Precision@5 is {wp:.3f} weighted and {cp:.3f} combined.")
        print("A likely reason is that the query terms are distributed across fields "
              "or the chosen weights favor a field with fewer relevant matches.")
        print(f"Relevant IDs: {sorted(relevant)}")
        print(f"Combined top IDs: {[book['book_id'] for book in combined[:K]]}")
        print(f"Weighted top IDs: {[book['book_id'] for book in weighted[:K]]}")
    else:
        print("\nEvery query improved with weighted search; there is no observed "
              "failure or tie to explain in this judged set.")

    return table


if __name__ == "__main__":
    evaluate()
