import json
import csv
from collections import Counter
import math

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

# 2. Calculate Ground Truth
def calculate_ground_truth(data, category):
    relevant_experts = Counter()

    for question_id, content in data.items():
        if category.lower() in [tag.lower() for tag in content.get("tags", [])]:
            for answer in content["answers"]:
                expert_id = answer["attorney_link"]

                # Local engagement condition: at least 2 accepted answers
                local_engagement = answer.get("best_answer") or answer.get("lawyers_agreed", 0) >= 3

                # Local quality ratio (calculated based on upvotes and agreement)
                local_quality_ratio = answer.get("lawyers_agreed", 0) > 3

                # Quality condition: more high-quality answers than the average (e.g., based on lawyers_agreed)
                quality = answer.get("lawyers_agreed", 0) > 0

                # Check if all conditions are met
                if local_engagement and local_quality_ratio and quality:
                    relevant_experts[expert_id] += 1

    # Filter experts with at least 10 relevant answers (or another threshold if needed)
    relevant_experts = [expert_id for expert_id, count in relevant_experts.items() if count >= 10]

    return relevant_experts

# 3. BM25 calculations
def calculate_bm25_score(query, document, term_frequencies, total_documents, avg_doc_length, k1=3, b=0.5):
    words = document.split()
    doc_length = len(words)
    score = 0

    for term in query.split():
        term_frequency = words.count(term)
        if term_frequency == 0:
            continue

        idf = math.log((total_documents + 0.5) / (term_frequencies.get(term, 0) + 0.5) + 1)  # Adjusted IDF weighting
        term_score = idf * (term_frequency * (k1 + 1)) / (term_frequency + k1 * (1 - b + b * (doc_length / avg_doc_length)))
        score += term_score

    return score

# Rank experts using BM25
def rank_experts_bm25(query, expert_answers):
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
        ranking.append((expert_id, expert_score / len(answers)))  # Normalize by number of answers

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
    # File paths for JSON and CSV files
    data_file_path = "/Users/arthurwunder/PycharmProjects/EF_in_Legal_CQA-ECIR2022/data/data_with_ids.json"
    csv_file_path = "/Users/arthurwunder/PycharmProjects/EF_in_Legal_CQA-ECIR2022/data/all_tags_stat.csv"

    # Load data
    expert_answers, data = load_and_aggregate_data(data_file_path)

    # Extract tags
    tags = extract_tags_from_csv(csv_file_path, min_occurrences=700)

    # Aggregate metrics over all tags
    total_map, total_mrr, total_p1, total_p2, total_p5 = 0, 0, 0, 0, 0
    num_queries = 0

    for tag in tags:
        print(f"\nProcessing Query: {tag}")

        # Calculate Ground Truth
        relevant_experts = calculate_ground_truth(data, category=tag)
        print(f"Relevant Experts for '{tag}': {relevant_experts}")

        # Rank experts using BM25
        ranking = rank_experts_bm25(tag, expert_answers)

        # Display results
        display_ranking(ranking)

        # Calculate MAP, MRR, and P@K
        map_score, mrr_score, p1, p2, p5 = calculate_map_mrr_and_precision(relevant_experts, ranking)
        print(f"MAP: {map_score:.4f}, MRR: {mrr_score:.4f}")
        print(f"P@1: {p1:.4f}, P@2: {p2:.4f}, P@5: {p5:.4f}")

        # Summing metrics
        total_map += map_score
        total_mrr += mrr_score
        total_p1 += p1
        total_p2 += p2
        total_p5 += p5
        num_queries += 1

    # Calculate average metrics
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
