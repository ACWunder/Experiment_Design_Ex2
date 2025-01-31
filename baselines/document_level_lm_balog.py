import json
import csv
from collections import Counter

# Globale Strukturen für die neue Ground Truth
experts_by_tag = {}  # z.B. {"0": ["19", "34", ...], "4": [...], ...}
tag_to_id = {}       # Umkehr-Mapping: {"Divorce": "0", "Dividing debts in a divorce": "4", ...}

# ---------------------------------------------------
# 0. Hilfsfunktion: Lade Ground-Truth-Dateien
# ---------------------------------------------------
def load_ground_truth_files(tag_ids_path, selection_file):
    """
    Lädt:
      - tagIDs.json (z.B. {"0": "Divorce", "4": "Dividing debts in a divorce", ...})
      - selection_tags_lawyers_experts.json (NDJSON, pro Zeile: {"tagID":"...","lawyerID":"...","expert":bool})
    und befüllt die globalen Strukturen:
      - experts_by_tag: (tagID → Liste von Lawyer-IDs) für alle Zeilen, bei denen expert=true
      - tag_to_id: Mapping von Tag-Text → Tag-ID (z.B. "Divorce" → "0")
    """
    global experts_by_tag, tag_to_id

    # 1) tagIDs.json laden
    with open(tag_ids_path, "r") as f:
        id_to_tag = json.load(f)  # z.B. {"0": "Divorce", "4": "Dividing debts in a divorce", ...}

    # Umkehrmapping: Tag-Text → Tag-ID
    tag_to_id = {v: k for k, v in id_to_tag.items()}

    # 2) selection_tags_lawyers_experts.json laden
    with open(selection_file, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            tID = record["tagID"]
            lID = record["lawyerID"]
            is_expert = record["expert"]
            if is_expert:
                if tID not in experts_by_tag:
                    experts_by_tag[tID] = []
                experts_by_tag[tID].append(lID)


# 1. Load data and calculate total term statistics
def load_and_calculate_statistics(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    document_term_frequencies = Counter()
    total_terms = 0
    total_answers = 0

    for question_id, content in data.items():
        for answer in content["answers"]:
            text = answer["answer_text"]
            words = text.split()
            document_term_frequencies.update(words)
            total_terms += len(words)
            total_answers += 1

    return data, document_term_frequencies, total_terms, total_answers

# 2. Angepasste Ground-Truth-Funktion
def calculate_ground_truth(data, category):
    """
    Greift auf die globalen Strukturen (experts_by_tag, tag_to_id) zu und liefert eine Liste
    von Experten-IDs (lawyerIDs) für den gesuchten 'category'-Tag.
    Parameter 'data' bleibt hier erhalten, um den Code ansonsten unverändert zu lassen.
    """
    global experts_by_tag, tag_to_id

    # Hole die Tag-ID anhand des Textes 'category'
    tag_id = tag_to_id.get(category, None)
    if tag_id is None:
        return []

    # Gebe alle Lawyer-IDs zurück, die in selection_tags_lawyers_experts.json mit "expert":true markiert sind
    return experts_by_tag.get(tag_id, [])

# 3. Document-Level Probability Calculations
def calculate_lambda_doc_level(doc_length, beta):
    return beta / (beta + doc_length)

def calculate_p_tc(term, term_frequencies, total_terms):
    return term_frequencies.get(term, 0) / total_terms

def calculate_document_score(query, document, term_frequencies, total_terms, beta):
    words = document.split()
    doc_length = len(words)
    lambda_doc_level = calculate_lambda_doc_level(doc_length, beta)

    score = 1
    for term in query.split():
        p_tc = calculate_p_tc(term, term_frequencies, total_terms)
        p_td = words.count(term) / doc_length if doc_length > 0 else 0
        term_score = (1 - lambda_doc_level) * p_td + lambda_doc_level * p_tc + 1e-10
        score *= term_score

    return score

# 4. Rank Experts Based on Document Scores
def rank_experts_doc_level(query, data, term_frequencies, total_terms, beta):
    expert_scores = {}

    for question_id, content in data.items():
        for answer in content["answers"]:
            expert_id = answer["attorney_link"]
            document = answer["answer_text"]
            document_score = calculate_document_score(query, document, term_frequencies, total_terms, beta)

            if expert_id not in expert_scores:
                expert_scores[expert_id] = []
            expert_scores[expert_id].append(document_score)

    # Aggregate scores per expert
    expert_ranking = []
    for expert_id, scores in expert_scores.items():
        avg_score = sum(scores) / len(scores)
        expert_ranking.append((expert_id, avg_score))

    return sorted(expert_ranking, key=lambda x: x[1], reverse=True)

# 5. Precision and MAP Calculations
def calculate_precision_at_k(relevant_experts, ranking, k):
    top_k = [expert_id for expert_id, _ in ranking[:k]]
    relevant_in_top_k = sum(1 for expert in top_k if expert in relevant_experts)
    return relevant_in_top_k / k if k > 0 else 0

def calculate_map_mrr_and_precision(relevant_experts, ranking):
    average_precision = 0
    reciprocal_rank = 0
    num_relevant_found = 0

    for rank, (expert_id, _) in enumerate(ranking, 1):
        if expert_id in relevant_experts:
            num_relevant_found += 1
            precision_at_k = num_relevant_found / rank
            average_precision += precision_at_k

            if reciprocal_rank == 0:
                reciprocal_rank = 1 / rank

    map_score = average_precision / len(relevant_experts) if relevant_experts else 0

    # Calculate P@1, P@2, P@5
    p1 = calculate_precision_at_k(relevant_experts, ranking, 1)
    p2 = calculate_precision_at_k(relevant_experts, ranking, 2)
    p5 = calculate_precision_at_k(relevant_experts, ranking, 5)

    return map_score, reciprocal_rank, p1, p2, p5

# 6. Extract Tags from CSV
def extract_tags_from_csv(csv_file, min_occurrences=700):
    tags = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["count_of_occurrences"]) > min_occurrences:
                tags.append(row["stag_name"])
    return tags

# Main Function
def main():
    # Datei-Pfade für JSON und CSV
    data_file_path = "/Users/arthurwunder/PycharmProjects/EF_in_Legal_CQA-ECIR2022/data/data_with_ids.json"
    csv_file_path = "/Users/arthurwunder/PycharmProjects/EF_in_Legal_CQA-ECIR2022/data/all_tags_stat.csv"

    # Ground-Truth-Dateien
    tag_ids_file_path = "/Users/arthurwunder/PycharmProjects/EF_in_Legal_CQA-ECIR2022/data/tagIDs.json"
    selection_experts_file_path = "/Users/arthurwunder/PycharmProjects/EF_in_Legal_CQA-ECIR2022/data/selection_tags_lawyers_experts.json"

    # 0. Ground Truth laden
    load_ground_truth_files(tag_ids_file_path, selection_experts_file_path)

    # 1. Data + Termstatistiken
    data, term_frequencies, total_terms, total_answers = load_and_calculate_statistics(data_file_path)
    beta = total_terms / total_answers

    # 2. Tags aus CSV extrahieren
    tags = extract_tags_from_csv(csv_file_path, min_occurrences=100)

    # 3. Optional: Filter Tags, die in den JSON-Daten überhaupt vorkommen
    valid_tags = [
        tag for tag in tags
        if any(tag.lower() in [t.lower() for t in content.get("tags", [])] for content in data.values())
    ]
    print(f"Valid Tags: {valid_tags}")

    # 4. Metriken sammeln
    total_map, total_mrr, total_p1, total_p2, total_p5 = 0, 0, 0, 0, 0
    num_queries = 0

    for tag in valid_tags:
        print(f"\nProcessing Query: {tag}")

        # Ground Truth
        relevant_experts = calculate_ground_truth(data, category=tag)
        print(f"Relevant Experts for '{tag}': {relevant_experts}")

        if not relevant_experts:
            print(f"No relevant experts for '{tag}'. Skipping...")
            continue

        # Ranking
        ranking = rank_experts_doc_level(tag, data, term_frequencies, total_terms, beta)

        # Metriken
        map_score, mrr_score, p1, p2, p5 = calculate_map_mrr_and_precision(relevant_experts, ranking)
        print(f"MAP: {map_score:.4f}, MRR: {mrr_score:.4f}")
        print(f"P@1: {p1:.4f}, P@2: {p2:.4f}, P@5: {p5:.4f}")

        total_map += map_score
        total_mrr += mrr_score
        total_p1 += p1
        total_p2 += p2
        total_p5 += p5
        num_queries += 1

    # 5. Aggregierte Ergebnisse
    if num_queries > 0:
        avg_map = total_map / num_queries
        avg_mrr = total_mrr / num_queries
        avg_p1 = total_p1 / num_queries
        avg_p2 = total_p2 / num_queries
        avg_p5 = total_p5 / num_queries

        print("\nAggregated Results:")
        print(f"Average MAP: {avg_map:.4f}")
        print(f"Average MRR: {avg_mrr:.4f}")
        print(f"Average P@1: {avg_p1:.4f}")
        print(f"Average P@2: {avg_p2:.4f}")
        print(f"Average P@5: {avg_p5:.4f}")

if __name__ == "__main__":
    main()
