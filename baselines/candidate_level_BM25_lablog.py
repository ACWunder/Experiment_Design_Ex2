import json
import csv
from collections import Counter
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Globale Variablen zum Speichern der Ground Truth
experts_by_tag = {}  # { "0": ["19", "25", ...], "4": [...], ... }
tag_to_id = {}       # { "Divorce": "0", "Dividing debts in a divorce": "4", ... }


# ---------------------------------------------------
# 0. Hilfsfunktion: Lade Ground-Truth-Dateien
# ---------------------------------------------------
def load_ground_truth_files(tag_ids_path, selection_file):
    """
    Lädt:
      - tagIDs.json (Dict: {"0": "Divorce", "4": "Dividing debts in a divorce", ...})
      - selection_tags_lawyers_experts.json (NDJSON: pro Zeile {"tagID": "...", "lawyerID": "...", "expert": bool})
    Schreibt in:
      - experts_by_tag: Liste von Lawyer-IDs (Strings) je Tag-ID
      - tag_to_id: Umkehrmapping von Tag-Name -> Tag-ID
    """
    global experts_by_tag, tag_to_id

    # 1) Lade tagIDs.json
    with open(tag_ids_path, "r") as f:
        id_to_tag = json.load(f)  # z.B. {"0": "Divorce", "4": "Dividing debts in a divorce", ...}
    # Umkehrmapping: "Divorce" -> "0", ...
    tag_to_id = {v: k for k, v in id_to_tag.items()}

    # 2) Lade selection_tags_lawyers_experts.json (NDJSON)
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


# Preprocess text (stopword removal and stemming)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return filtered_tokens

# 1. Load data and aggregate answers by experts
def load_and_aggregate_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    expert_answers = {}
    for question_id, content in data.items():
        for answer in content["answers"]:
            expert_id = answer["attorney_link"]
            answer_text = answer["answer_text"]
            # Apply preprocessing if desired:
            #answer_text = preprocess_text(answer_text)
            #answer_text = " ".join(answer_text)

            if expert_id not in expert_answers:
                expert_answers[expert_id] = []
            expert_answers[expert_id].append(answer_text)

    return expert_answers, data

# 2. Neue Ground-Truth-Funktion, die nur auf die geladenen Dateien zugreift
def calculate_ground_truth(data, category):
    """
    Ermittelt relevante Experten anhand der zuvor geladenen Dateien:
      - tagIDs.json
      - selection_tags_lawyers_experts.json
    'data' und 'category' bleiben im Funktionskopf, um den bestehenden Code unverändert zu lassen.
    """
    global experts_by_tag, tag_to_id

    # Hole die zugehörige Tag-ID
    tag_id = tag_to_id.get(category, None)
    if tag_id is None:
        # Tag nicht gefunden => Leere Liste
        return []

    # Liefere alle Lawyer-IDs, die in selection_tags_lawyers_experts.json als "expert":true markiert sind
    return experts_by_tag.get(tag_id, [])

# 3. BM25 calculations
def calculate_bm25_score(query, document, term_frequencies, total_documents, avg_doc_length, k1=1.5, b=0.75):
    words = document.split()
    doc_length = len(words)
    score = 0

    # Preprocessing nur für den Query
    query_terms = preprocess_text(query)
    for term in query_terms:
        term_frequency = words.count(term)
        if term_frequency == 0:
            continue

        # IDF nach BM25
        idf = math.log((total_documents - term_frequencies.get(term, 0) + 0.5) /
                       (term_frequencies.get(term, 0) + 0.5) + 1)

        # Rechenformel BM25
        term_score = idf * (term_frequency * (k1 + 1)) / \
                     (term_frequency + k1 * (1 - b + b * (doc_length / avg_doc_length)))
        score += term_score

    return score

# Rank experts using BM25
def rank_experts_bm25(query, expert_answers):
    # Gesamten Text für Frequenzstatistik zusammenfassen
    all_texts = " ".join([" ".join(answers) for answers in expert_answers.values()])
    all_words = all_texts.split()
    term_frequencies = Counter(all_words)
    total_documents = len(expert_answers)
    avg_doc_length = sum(len(" ".join(answers).split()) for answers in expert_answers.values()) / total_documents

    ranking = []
    for expert_id, answers in expert_answers.items():
        expert_score = 0
        for answer in answers:
            expert_score += calculate_bm25_score(query, answer, term_frequencies, total_documents, avg_doc_length)
        # Normalisierung durch Anzahl der Antworten
        ranking.append((expert_id, expert_score / len(answers)))

    ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
    return ranking

# Display ranking results
def display_ranking(ranking, top_n=10):
    print("Top Experts for the Query:")
    for rank, (expert_id, score) in enumerate(ranking[:top_n], 1):
        print(f"Rank {rank}: Expert {expert_id}, Score: {score:.6f}")

# P@K calculations
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

            if reciprocal_rank == 0:  # First relevant expert found
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
    # Relative/Absolute file paths for JSON and CSV files
    data_file_path = "../data/data_with_ids.json"
    csv_file_path = "../data/all_tags_stat.csv"

    # Dateien für die Ground Truth
    tag_ids_path = "../data/tagIDs.json"
    selection_experts_path = "../data/selection_tags_lawyers_experts.json"

    print("Program started")

    # 0. Ground-Truth-Dateien laden
    load_ground_truth_files(tag_ids_path, selection_experts_path)

    # 1. Daten für das Ranking laden
    expert_answers, data = load_and_aggregate_data(data_file_path)

    # 2. Relevante Tags aus CSV extrahieren
    tags = extract_tags_from_csv(csv_file_path, min_occurrences=700)

    # 3. Aggregate metrics over all tags
    total_map, total_mrr, total_p1, total_p2, total_p5 = 0, 0, 0, 0, 0
    num_queries = 0

    for tag in tags:
        print(f"\nProcessing Query: {tag}")

        # Ground Truth aus den JSON-Dateien
        relevant_experts = calculate_ground_truth(data, category=tag)
        print(f"Relevant Experts for '{tag}': {relevant_experts}")

        # BM25-Ranking
        ranking = rank_experts_bm25(tag, expert_answers)

        # Anzeige
        display_ranking(ranking)

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

    # 4. Durchschn. Ergebnisse berechnen
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
