# ─────────────────────────────────────────────────────────────
# CONFIGURATION LOADING FROM config.txt
# ─────────────────────────────────────────────────────────────
# This file reads all secrets and configuration from config.txt
# Expected keys in config.txt (supports multiple naming conventions):
#   - Neo4j: URI or NEO4J_URI, USERNAME or NEO4J_USER, PASSWORD or NEO4J_PASSWORD
#   - HuggingFace: HF_TOKEN or hfToken or HFSecret or LLM_Token
#   - OpenRouter: OPENROUTER_KEY or openrouterKey or openrouter
# ─────────────────────────────────────────────────────────────
import os

def load_config():
    """Load all configuration from config.txt file"""
    config = {}
    try:
        with open("config.txt", "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    config[k.strip()] = v.strip()
    except FileNotFoundError:
        print("⚠️  config.txt not found. Using environment variables or defaults.")
    return config

# Load configuration
_config = load_config()

# ─────────────────────────────────────────────────────────────
# REQUIRED PACKAGES (install if needed)
# ─────────────────────────────────────────────────────────────
# !pip install neo4j sentence-transformers

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Any
from openai import OpenAI

# Import the main class
from fpl_Task2 import FPLGraphRetrieval

# ─────────────────────────────────────────────────────────────
# NEO4J CONFIGURATION (supports multiple naming conventions)
# ─────────────────────────────────────────────────────────────
NEO4J_URI = (
    _config.get("URI") or 
    _config.get("NEO4J_URI") or 
    os.getenv("NEO4J_URI", "neo4j+s://1da86c19.databases.neo4j.io")
)
NEO4J_USER = (
    _config.get("USERNAME") or 
    _config.get("NEO4J_USER") or 
    os.getenv("NEO4J_USER", "neo4j")
)
NEO4J_PASSWORD = (
    _config.get("PASSWORD") or 
    _config.get("NEO4J_PASSWORD") or 
    os.getenv("NEO4J_PASSWORD", "HA4iunTOGen7RYpeISs3ZRhcWjpcokqam9przCqCuQ8")
)

# ─────────────────────────────────────────────────────────────
# API KEYS FROM CONFIG (supports multiple naming conventions)
# ─────────────────────────────────────────────────────────────
# Try multiple possible key names from config.txt
HF_TOKEN = (
    _config.get("HF_TOKEN") or 
    _config.get("hfToken") or 
    _config.get("HFSecret") or 
    _config.get("LLM_Token") or
    os.getenv("HF_TOKEN", "")
)

OPENROUTER_KEY = (
    _config.get("OPENROUTER_KEY") or 
    _config.get("openrouterKey") or 
    _config.get("openrouter") or
    os.getenv("OPENROUTER_KEY", "")
)

# ─────────────────────────────────────────────────────────────
# INITIALIZE RETRIEVER
# ─────────────────────────────────────────────────────────────
retriever = FPLGraphRetrieval(
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
    hf_token=HF_TOKEN if HF_TOKEN else None,  # Optional, for LLM mode
    use_llm=True if HF_TOKEN else False  # Set True to use LLM for entity extraction
)

# ─────────────────────────────────────────────────────────────
# LLM FUNCTION
# ─────────────────────────────────────────────────────────────
def call_llm(user_question: str, model: str, embedding_type: str = 'numeric') -> str:
    """
    Takes user question, handles retrieval with specified embedding type.
    """

    # 1️⃣ Get API key from config
    openrouter_key = OPENROUTER_KEY
    if not openrouter_key:
        # Try reloading config in case it was updated
        _config_reload = load_config()
        openrouter_key = _config_reload.get("OPENROUTER_KEY", os.getenv("OPENROUTER_KEY", ""))
    
    if not openrouter_key:
        raise ValueError("OPENROUTER_KEY not found. Please set it in config.txt or environment variables.")

    # 2️⃣ Initialize OpenRouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key
    )

    
    result = retriever.retrieve(user_question, method="both", embedding_type=embedding_type)

    # print("="*50)
    # print("RETRIEVAL DEBUG:")
    # print(f"Intent detected: {result.get('baseline', {}).get('intent')}")
    # print(f"Query used: {result.get('baseline', {}).get('query_used')}")
    # print(f"Parameters: {result.get('baseline', {}).get('parameters')}")
    # print(f"Results: {result.get('baseline', {}).get('results', [])[:2]}")  # First 2 results
    # print("="*50)
    print(result)
    

    context_text = result

    # 5️⃣ Define persona
    persona_text = (
    "You are an FPL (Fantasy Premier League) expert. "
    "You can answer any questions related to FPL, including player stats, "
    "team performance, transfers, and gameweek strategies. "
    "You are also knowledgeable about the Premier League in general."
    )

    task_text = (
    "IMPORTANT: You MUST always provide a response. Never return empty text.\n\n"
    
    "SPECIAL INSTRUCTIONS FOR COMPARISON QUERIES:\n"
    "When the user asks to compare players (e.g., 'compare X and Y'):\n"
    "1. First check if BOTH players are mentioned in the CONTEXT\n"
    "2. If yes, extract ALL their stats from the CONTEXT\n"
    "3. Create a detailed comparison table or bullet points\n"
    "4. Include: goals, points, assists, clean sheets (if relevant), bonus points\n"
    "5. Provide a clear conclusion about who performed better overall\n\n"
    
    "EXAMPLE FORMAT for player comparisons:\n"
    "Comparison: Player A vs Player B\n"
    "• Player A: X goals, Y points, Z assists, W bonus points\n"
    "• Player B: A goals, B points, C assists, D bonus points\n"
    "Conclusion: [Who was better and why]\n\n"
    
    "CONTEXT:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "ANSWER (remember: never leave this blank):"
    )

    # 7️⃣ Build the prompt
    prompt = f"""
    CONTEXT:
    {context_text}

    PERSONA:
    {persona_text}

    TASK:
    {task_text}
    """

    # 8️⃣ Call the LLM
    completion = client.chat.completions.create(
        model= model,
        messages=[
            {"role": "system", "content": persona_text},
            {"role": "user", "content": f"{prompt}\n\nQuestion: {user_question}"}
        ],
        temperature=0.2,
        max_tokens=500
    )

    # 9️⃣ Return the LLM answer

    answer = completion.choices[0].message.content

    # ✅ Token usage (access as attributes, not dict)
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    total_tokens = completion.usage.total_tokens





    return completion.choices[0].message.content, total_tokens













def compare_models_human_evaluation():
    """
    Run 3 models on test queries and perform human evaluation with comparison.
    Self-contained function - no inputs required. Models and queries defined internally.
    
    Returns:
        Dictionary with evaluations and comparison table
    """
    import json
    import os
    import time
    
    # =========================================================================
    # CONFIGURATION - Edit these to customize your evaluation
    # =========================================================================
    
    models_dict = {
        "1": "mistralai/mistral-7b-instruct:free",
        "2": "meta-llama/llama-3.3-70b-instruct:free",
        "3": "mistralai/devstral-2512:free",
    }
    
    test_queries = [
        {
            "question": "Who is the best FPL player this season?", 
            "ground_truth": "Erling Haaland"
        },
        {
            "question": "Which midfielder has the most assists?",
            "ground_truth": "Kevin De Bruyne"
        },
        {
            "question": "Who are the top 3 defenders by points?",
            "ground_truth": "Reece James, Trent Alexander-Arnold, Joao Cancelo"
        }
    ]
    
    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================
    
    models = list(models_dict.values())
    model_labels = list(models_dict.keys())
    
    if len(models) != 3:
        raise ValueError("Please provide exactly 3 models for comparison")
    
    print("\n" + "="*80)
    print("          RUNNING MODELS ON TEST QUERIES")
    print("="*80 + "\n")
    
    # Step 1: Run all models and collect results
    all_model_results = {}
    
    for model_id, model_name in models_dict.items():
        display_name = f"Model {model_id} ({model_name.split('/')[1].split(':')[0]})"
        print(f"🔄 Running {display_name}...")
        model_results = []
        
        for query in test_queries:
            question = query["question"]
            ground_truth = query.get("ground_truth")
            
            try:
                answer, tokens_used = call_llm(question, model_name)
                
                model_results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": answer,
                    "tokens_used": tokens_used
                })
            except Exception as e:
                model_results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": f"ERROR: {str(e)}",
                    "tokens_used": 0
                })
        
        all_model_results[display_name] = model_results
        print(f"✓ {display_name} completed\n")
    
    model_display_names = list(all_model_results.keys())
    
    # Step 2: Human evaluation for each query across all models
    print("\n" + "="*80)
    print("          HUMAN EVALUATION SESSION")
    print("="*80 + "\n")
    
    evaluations = {model: [] for model in model_display_names}
    total_queries = len(test_queries)
    
    for query_idx in range(total_queries):
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(f"\n📊 Progress: Query {query_idx + 1}/{total_queries} ({(query_idx + 1)/total_queries*100:.0f}%)")
        print("="*80)
        
        # Display question and ground truth
        query_data = test_queries[query_idx]
        print(f"\n❓ Question: {query_data['question']}")
        if query_data.get('ground_truth'):
            print(f"✓ Ground Truth: {query_data['ground_truth']}")
        
        print("\n" + "-"*80)
        
        # Evaluate each model's answer for this query
        for model_name in model_display_names:
            result = all_model_results[model_name][query_idx]
            
            print(f"\n🤖 {model_name}")
            print(f"💬 Answer: {result['answer']}")
            print(f"🔢 Tokens Used: {result['tokens_used']}")
            print("-"*80)
            
            # Collect ratings
            print(f"\nRate this answer (1-5):")
            ratings = {}
            metrics = {
                'relevance': '🎯 Relevance',
                'correctness': '✅ Correctness',
                'naturalness': '🗣️  Naturalness',
                'completeness': '📝 Completeness',
                'overall': '⭐ Overall Quality'
            }
            
            for key, label in metrics.items():
                while True:
                    try:
                        rating = int(input(f"  {label} (1-5): "))
                        if 1 <= rating <= 5:
                            ratings[key] = rating
                            break
                        else:
                            print("    ⚠️  Please enter 1-5")
                    except ValueError:
                        print("    ⚠️  Please enter a valid number")
            
            comments = input(f"\n💭 Comments (optional): ")
            
            # Store evaluation
            evaluations[model_name].append({
                **result,
                "human_eval": {
                    **ratings,
                    "comments": comments
                }
            })
            
            print()
        
        # Save progress
        progress_data = {
            "models": model_display_names,
            "evaluations": evaluations,
            "queries_completed": query_idx + 1
        }
        with open("model_comparison_progress.json", "w") as f:
            json.dump(progress_data, f, indent=2)
        
        print(f"✓ Progress saved ({query_idx + 1}/{total_queries})")
        
        if query_idx < total_queries - 1:
            input("\nPress Enter to continue to next query...")
    
    # Step 3: Calculate statistics and create comparison table
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("\n" + "="*80)
    print("          EVALUATION COMPLETE - COMPARISON RESULTS")
    print("="*80 + "\n")
    
    metrics_list = ['relevance', 'correctness', 'naturalness', 'completeness', 'overall']
    comparison_data = {}
    
    for model_name in model_display_names:
        model_evals = evaluations[model_name]
        
        avg_metrics = {}
        for metric in metrics_list:
            avg = sum(e["human_eval"][metric] for e in model_evals) / len(model_evals)
            avg_metrics[metric] = round(avg, 2)
        
        avg_tokens = sum(e["tokens_used"] for e in model_evals) / len(model_evals)
        
        comparison_data[model_name] = {
            "averages": avg_metrics,
            "avg_tokens": round(avg_tokens, 1),
            "evaluations": model_evals
        }
    
    # Print comparison table
    print_comparison_table(model_display_names, comparison_data, metrics_list)
    
    # Save final results
    final_results = {
        "models": model_display_names,
        "total_queries": total_queries,
        "comparison": comparison_data,
        "config": {
            "models_tested": models_dict,
            "queries": test_queries
        }
    }
    
    with open("model_comparison_final.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Full results saved to: model_comparison_final.json")
    
    return final_results


def print_comparison_table(models, comparison_data, metrics):
    """Print a formatted comparison table"""
    
    # Header
    print("📊 COMPARISON TABLE")
    print("="*100)
    
    col_width = 28
    print(f"{'Metric':<{col_width}}", end="")
    for model in models:
        # Truncate long model names for display
        display_name = model if len(model) <= 25 else model[:22] + "..."
        print(f"{display_name:<{col_width}}", end="")
    print()
    print("-"*100)
    
    # Metrics rows
    metric_labels = {
        'relevance': '🎯 Relevance',
        'correctness': '✅ Correctness',
        'naturalness': '🗣️  Naturalness',
        'completeness': '📝 Completeness',
        'overall': '⭐ Overall'
    }
    
    for metric in metrics:
        label = metric_labels.get(metric, metric.capitalize())
        print(f"{label:<{col_width}}", end="")
        
        scores = [comparison_data[model]["averages"][metric] for model in models]
        max_score = max(scores) if scores else 0
        
        for model in models:
            score = comparison_data[model]["averages"][metric]
            is_winner = (score == max_score and score > 0)
            score_str = f"{score:.2f}/5.00"
            if is_winner:
                print(f"{score_str} 🏆{'':<{col_width-len(score_str)-3}}", end="")
            else:
                print(f"{score_str}{'':<{col_width-len(score_str)}}", end="")
        print()
    
    print("-"*100)
    
    # Token usage
    print(f"{'🔢 Avg Tokens Used':<{col_width}}", end="")
    for model in models:
        tokens = comparison_data[model]["avg_tokens"]
        tokens_str = f"{tokens:.0f}"
        print(f"{tokens_str}{'':<{col_width-len(tokens_str)}}", end="")
    print()
    
    print("="*100)
    
    # Determine winner
    overall_scores = [(model, comparison_data[model]["averages"]["overall"]) for model in models]
    winner = max(overall_scores, key=lambda x: x[1])
    
    print(f"\n🏆 WINNER: {winner[0]}")
    print(f"   Overall Score: {winner[1]:.2f}/5.00\n")


    # =========================================================================
    # USAGE EXAMPLE
    # =========================================================================    
    # Results are automatically saved to:
    # - model_comparison_progress.json (during evaluation)
    # - model_comparison_final.json (after completion)
    
    print("\n✅ Evaluation complete!")
    print(f"📊 {len(results['comparison'])} models evaluated")
    print(f"❓ {results['total_queries']} queries tested")







def evaluate_model_batch(model):
    """
    Evaluate a single model on predefined test queries.
    Queries are defined internally - only model name is required as input.
    
    Measures:
    - Accuracy (based on similarity threshold > 0.8)
    - Similarity score (0-1 range)
    - Response time (seconds)
    - Tokens used
    
    Args:
        model: Model identifier (e.g., "mistralai/mistral-7b-instruct:free")
    
    Returns:
        Dictionary with evaluation results and metrics
    """
    import time
    
    # =========================================================================
    # CONFIGURATION - Test queries defined internally
    # =========================================================================
    
    queries_with_truths = [
        {
            "question": "Who is the best FPL player this season?",
            "ground_truth": "Erling Haaland"
        },
        {
            "question": "Which midfielder has the most assists?",
            "ground_truth": "Kevin De Bruyne"
        },
        {
            "question": "Who are the top 3 defenders by points?",
            "ground_truth": "Reece James, Trent Alexander-Arnold, Joao Cancelo"
        },
        {
            "question": "Which goalkeeper has the most clean sheets?",
            "ground_truth": "Alisson Becker"
        },
        {
            "question": "Who is the best budget forward under 7 million?",
            "ground_truth": "Ollie Watkins"
        },
        {
            "question": "Which team has scored the most goals?",
            "ground_truth": "Manchester City"
        },
        {
            "question": "Who has the highest ICT index?",
            "ground_truth": "Mohamed Salah"
        },
        {
            "question": "Which player has the most bonus points?",
            "ground_truth": "Erling Haaland"
        },
        {
            "question": "What is the best captain choice for this gameweek?",
            "ground_truth": "Erling Haaland"
        },
        {
            "question": "Which defender has the most attacking returns?",
            "ground_truth": "Trent Alexander-Arnold"
        }
    ]
    
    # =========================================================================
    # EVALUATION EXECUTION
    # =========================================================================
    
    results = []
    total_accuracy = 0
    total_response_time = 0
    total_tokens = 0
    total_similarity = 0
    counted_accuracies = 0
    
    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL: {model}")
    print(f"{'='*80}\n")
    
    for idx, item in enumerate(queries_with_truths, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth")
        
        print(f"{'─'*80}")
        print(f"Query {idx}/{len(queries_with_truths)}")
        print(f"{'─'*80}")
        print(f"❓ Question: {question}")
        if ground_truth:
            print(f"✓ Ground Truth: {ground_truth}")
        
        start = time.time()
        try:
            answer, tokens_used = call_llm(question, model)
            print(f"🤖 Model Answer: {answer}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "error": str(e),
                "response_time": None,
                "accuracy": None,
                "similarity_score": None,
                "tokens_used": 0
            })
            print()
            continue
        
        end = time.time()
        response_time = end - start
        total_tokens += tokens_used
        total_response_time += response_time
        
        accuracy = None
        similarity_score = None
        if ground_truth:
            # Calculate similarity between answer and ground truth
            similarity_score = calculate_similarity(answer, ground_truth)
            # Consider correct if similarity > threshold (0.6)
            accuracy = int(similarity_score > 0.6)
            total_accuracy += accuracy
            total_similarity += similarity_score
            counted_accuracies += 1
            
            # Print evaluation metrics
            status = "✅ CORRECT" if accuracy else "❌ INCORRECT"
            print(f"\n{status}")
            print(f"📊 Similarity: {similarity_score:.2%}")
            print(f"⏱️  Response Time: {response_time:.3f}s")
            print(f"🔢 Tokens Used: {tokens_used}")
        else:
            print(f"\n⏱️  Response Time: {response_time:.3f}s")
            print(f"🔢 Tokens Used: {tokens_used}")
        
        print()
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "response_time": response_time,
            "accuracy": accuracy,
            "similarity_score": similarity_score,
            "tokens_used": tokens_used,
        })
    
    num_queries = len(queries_with_truths)
    avg_response_time = total_response_time / num_queries if num_queries > 0 else 0
    avg_accuracy = (total_accuracy / counted_accuracies) * 100 if counted_accuracies > 0 else None
    avg_similarity = total_similarity / counted_accuracies if counted_accuracies > 0 else None
    avg_tokens = total_tokens / num_queries if num_queries > 0 else 0
    
    # Print summary
    print(f"{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Total Queries: {num_queries}")
    print(f"Average Response Time: {avg_response_time:.3f}s")
    if avg_accuracy is not None:
        print(f"Average Accuracy: {avg_accuracy:.1f}%")
        print(f"Average Similarity: {avg_similarity:.2%}")
    print(f"Average Tokens Used: {avg_tokens:.1f}")
    print(f"{'='*80}\n")
    
    return {
        "model": model,
        "results": results,
        "average_response_time": avg_response_time,
        "average_accuracy": avg_accuracy,
        "average_similarity": avg_similarity,
        "average_tokens_used": avg_tokens,
    }


def calculate_similarity(answer: str, ground_truth: str) -> float:
    """
    Calculate similarity between answer and ground truth.
    Returns a score between 0 and 1.
    """
    from difflib import SequenceMatcher
    
    # Normalize text
    answer_normalized = answer.lower().strip()
    truth_normalized = ground_truth.lower().strip()
    
    # Extract key tokens (ignore common words)
    common_words = {'is', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    
    answer_tokens = set(word for word in answer_normalized.split() if word not in common_words)
    truth_tokens = set(word for word in truth_normalized.split() if word not in common_words)
    
    if not truth_tokens:
        return 0.0
    
    # Method 1: Sequence matcher (good for exact string similarity)
    sequence_similarity = SequenceMatcher(None, answer_normalized, truth_normalized).ratio()
    
    # Method 2: Token overlap (focuses on important words)
    intersection = len(answer_tokens & truth_tokens)
    token_overlap = intersection / len(truth_tokens)  # What % of ground truth tokens are in answer?
    
    # Method 3: Partial containment (check if any ground truth token is in answer)
    any_match = any(token in answer_normalized for token in truth_tokens)
    partial_score = 1.0 if any_match else 0.0
    
    # Method 4: Check if ground truth is fully contained in answer
    full_containment = 1.0 if truth_normalized in answer_normalized else 0.0
    
    # Combine methods with adjusted weights
    # Higher weight on token_overlap for flexible matching
    final_similarity = (
        sequence_similarity * 0.2 +
        token_overlap * 0.5 +        # Prioritize key word matches
        partial_score * 0.1 +
        full_containment * 0.2
    )
    
    return final_similarity






def main():
    user_question = input("Hey manager, what do you need to know?: ")
    
    models = {
    "1": "mistralai/devstral-2512:free",
    "2":  "meta-llama/llama-3.3-70b-instruct:free",
    "3": "nvidia/nemotron-3-nano-30b-a3b:free",
    }
    
    embedding_types = {
        "1": "numeric",
        "2": "minilm", 
        "3": "mpnet"
    }
    
    # Ask for model choice
    model_choice = input("Choose a model (1-3): ").strip()
    
    # Ask for embedding type choice
    print("\nEmbedding types:")
    print("1. Numeric (fast, stat-based)")
    print("2. MiniLM (semantic, good balance)")
    print("3. MPNet (most accurate semantic)")
    embedding_choice = input("Choose embedding type (1-3): ").strip()
    
    if model_choice in models and embedding_choice in embedding_types:
        model = models[model_choice]
        embedding_type = embedding_types[embedding_choice]
        
        print(f"You selected Model {model_choice}: {model}")
        print(f"Using {embedding_type} embeddings")
        
        answer = call_llm(user_question, model, embedding_type)
        print("\nLLM Answer:\n", answer[0])
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
