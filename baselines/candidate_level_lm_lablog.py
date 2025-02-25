import json
import csv
from collections import Counter

# Globale Dictionaries für Tag-Mappings und Experten pro Tag
experts_by_tag = {}
tag_to_id = {}

# 0. Hilfsfunktionen zum Laden der Ground-Truth-Dateien
def load_ground_truth_files(tag_ids_path, selection_file):
    """
    Lädt:
      - tagIDs.json (Dict: {"0": "Divorce", "4": "Dividing debts in a divorce", ...})
      - selection_tags_lawyers_experts.json (newline-delimited, pro Zeile {"tagID":"0","lawyerID":"19","expert":true}, ...)
    und füllt zwei globale Strukturen:
      - tag_to_id: Umkehrmapping von Tag-Text zu Tag-ID
      - experts_by_tag: Liste von Experten-Profilen (lawyerID) pro Tag-ID
    """
    global experts_by_tag, tag_to_id

    # 1) tagIDs.json laden (z.B. {"0": "Divorce", "4": "Dividing debts in a divorce", ...})
    with open(tag_ids_path, "r") as f:
        id_to_tag = json.load(f)  # key = ID als String, value = Tag-Name

    # Umkehrmapping: "Divorce" -> "0", "Dividing debts in a divorce" -> "4", ...
    tag_to_id = {v: k for k, v in id_to_tag.items()}

    # 2) selection_tags_lawyers_experts.json laden (NDJSON: pro Zeile ein JSON-Datensatz)
    with open(selection_file, "r") as f:
        for line in f:
            record = json.loads(line)
            tID = record["tagID"]      # z.B. "0"
            lID = record["lawyerID"]   # z.B. "19"
            is_expert = record["expert"]
            # Falls dieser Anwalt als Experte gekennzeichnet ist:
            if is_expert:
                if tID not in experts_by_tag:
                    experts_by_tag[tID] = []
                experts_by_tag[tID].append(lID)

# 1. Load data and aggregate answers by experts
def load_and_aggregate_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    expert_answers = {}
    for question_id, content in data.items():
        for answer in content["answers"]:
            expert_id = answer["attorney_link"]
            answer_text = answer["answer_text"]

            if expert_id not in expert_answers:
                expert_answers[expert_id] = []
            expert_answers[expert_id].append(answer_text)

    return expert_answers, data

# 2. *Neue* Ground-Truth-Funktion, die auf den geladenen JSON-Dateien basiert
def calculate_ground_truth(data, category):
    """
    Greift auf das globale Mapping (experts_by_tag, tag_to_id) zurück
    und liefert die Liste relevanter Experten (lawyerIDs) für den gesuchten Tag.
    Parameter 'data' wird nur beibehalten, damit der restliche Code unverändert bleibt.
    """
    global experts_by_tag, tag_to_id

    # Hole die Tag-ID (z.B. "0" für "Divorce") aus dem category-String
    tag_id = tag_to_id.get(category, None)
    if tag_id is None:
        # Falls der Tag-Text nicht in tagIDs.json vorkommt, ist die Relevanz-Menge leer
        return []

    # Liefere Liste aller Experten (lawyerIDs) für diese Tag-ID
    return experts_by_tag.get(tag_id, [])

# 3. Probability calculations
# P(t|d): Term frequency in an answer
def calculate_p_t_d(term, document):
    words = document.split()
    term_frequency = words.count(term)
    document_length = len(words)
    return term_frequency / document_length if document_length > 0 else 0

# P(t): Term frequency in the collection
def calculate_p_t(term, term_frequencies, total_terms):
    return term_frequencies[term] / total_terms if term in term_frequencies else 0

# Smoothing (\lambda):
def calculate_lambda(expert_id, expert_answers, beta=1500):
    expert_texts = " ".join(expert_answers[expert_id])
    n_ca = len(expert_texts.split())
    return n_ca / (n_ca + beta)

# Expert score for a query term
def calculate_expert_score(term, expert_id, expert_answers, term_frequencies, total_terms):
    # Background probability
    p_t = calculate_p_t(term, term_frequencies, total_terms)
    lambda_value = calculate_lambda(expert_id, expert_answers)
    background_score = lambda_value * p_t

    # Foreground probability
    foreground_score = 0
    for document in expert_answers[expert_id]:
        p_t_d = calculate_p_t_d(term, document)
        foreground_score += (1 - lambda_value) * p_t_d

    return background_score + foreground_score

# Total score for a query
def calculate_query_score(query, expert_id, expert_answers, term_frequencies, total_terms):
    query_terms = query.lower().split()
    total_score = 1  # Multiplikative Kombination der einzelnen Term-Wahrscheinlichkeiten
    for term in query_terms:
        score = calculate_expert_score(term, expert_id, expert_answers, term_frequencies, total_terms)
        if score == 0:
            # Verhindert Multiplikation mit 0
            total_score *= 1e-10
        else:
            total_score *= score
    return total_score

# Ranking experts
def rank_experts(query, expert_answers):
    # Prepare term statistics
    all_texts = " ".join([" ".join(answers) for answers in expert_answers.values()])
    all_words = all_texts.split()
    term_frequencies = Counter(all_words)
    total_terms = sum(term_frequencies.values())

    # Calculate ranking
    ranking = []
    for expert_id in expert_answers.keys():
        score = calculate_query_score(query, expert_id, expert_answers, term_frequencies, total_terms)
        ranking.append((expert_id, score))

    # Sort by score in descending order
    ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
    return ranking

# Display ranking results
def display_ranking(ranking, top_n=10):
    print("Top Experts for the Query:")
    for rank, (expert_id, score) in enumerate(ranking[:top_n], 1):
        print(f"Rank {rank}: Expert {expert_id}, Score: {score:.6f}")

# Calculate P@K
def calculate_precision_at_k(relevant_experts, ranking, k):
    top_k = [expert_id for expert_id, _ in ranking[:k]]
    relevant_in_top_k = sum(1 for expert in top_k if expert in relevant_experts)
    return relevant_in_top_k / k

# Calculate MAP, MRR, and P@K
def calculate_map_mrr_and_precision(relevant_experts, ranking):
    average_precision = 0
    reciprocal_rank = 0
    num_relevant_found = 0

    for rank, (expert_id, _) in enumerate(ranking, 1):
        if expert_id in relevant_experts:
            num_relevant_found += 1
            precision_at_k = num_relevant_found / rank
            average_precision += precision_at_k

            if reciprocal_rank == 0:  # First relevant expert
                reciprocal_rank = 1 / rank

    map_score = average_precision / len(relevant_experts) if relevant_experts else 0

    # Calculate P@1, P@2, P@5
    p1 = calculate_precision_at_k(relevant_experts, ranking, 1)
    p2 = calculate_precision_at_k(relevant_experts, ranking, 2)
    p5 = calculate_precision_at_k(relevant_experts, ranking, 5)

    return map_score, reciprocal_rank, p1, p2, p5

# Extract tags from CSV
def extract_tags_from_csv(csv_file, min_occurrences=700):
    tags = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["count_of_occurrences"]) > min_occurrences:
                tags.append(row["stag_name"])
    return tags

# Main function
def main():
    # Datei-Pfade anpassen
    data_file_path = "/Users/arthurwunder/PycharmProjects/EF_in_Legal_CQA-ECIR2022/data/data_with_ids.json"
    csv_file_path = "/Users/arthurwunder/PycharmProjects/EF_in_Legal_CQA-ECIR2022/data/all_tags_stat.csv"

    # *** Neue Dateien für Ground Truth ***
    tag_ids_file_path = "/Users/arthurwunder/PycharmProjects/EF_in_Legal_CQA-ECIR2022/data/tagIDs.json"
    selection_experts_file_path = "/Users/arthurwunder/PycharmProjects/EF_in_Legal_CQA-ECIR2022/data/selection_tags_lawyers_experts.json"

    # 0. Ground-Truth-Dateien laden
    load_ground_truth_files(tag_ids_file_path, selection_experts_file_path)

    # 1. Daten laden für das Ranking
    expert_answers, data = load_and_aggregate_data(data_file_path)

    # 2. Relevante Tags aus CSV extrahieren
    tags = extract_tags_from_csv(csv_file_path, min_occurrences=700)

    # 3. Metriken über alle Tags sammeln
    total_map, total_mrr, total_p1, total_p2, total_p5 = 0, 0, 0, 0, 0
    num_queries = 0

    for tag in tags:
        print(f"\nProcessing Query: {tag}")

        # Ground Truth aus den geladenen JSON-Dateien
        relevant_experts = calculate_ground_truth(data, category=tag)
        print(f"Relevant Experts for '{tag}': {relevant_experts}")

        # Experten ranken
        ranking = rank_experts(tag, expert_answers)

        # Ergebnisse anzeigen
        display_ranking(ranking)

        # Metriken berechnen
        map_score, mrr_score, p1, p2, p5 = calculate_map_mrr_and_precision(relevant_experts, ranking)
        print(f"MAP: {map_score:.4f}, MRR: {mrr_score:.4f}")
        print(f"P@1: {p1:.4f}, P@2: {p2:.4f}, P@5: {p5:.4f}")

        # Werte aufsummieren
        total_map += map_score
        total_mrr += mrr_score
        total_p1 += p1
        total_p2 += p2
        total_p5 += p5
        num_queries += 1

    # 4. Durchschnittliche Metriken ausgeben
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
