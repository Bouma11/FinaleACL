from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("preprocesskey")
secret_value_1 = user_secrets.get_secret("preprocesskey")
# secret_value_2 = user_secrets.get_secret("j")
# secret_value_3 = user_secrets.get_secret("LLM Token")
secret_value_4 = user_secrets.get_secret("openrouterkey")
secret_value_5 = user_secrets.get_secret("openrouterkey")
secret_value_6 = user_secrets.get_secret("preprocesskey")




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

# Import the main class
from fpl_Task2 import FPLGraphRetrieval

# ─────────────────────────────────────────────────────────────
# NEO4J CONFIGURATION
# ─────────────────────────────────────────────────────────────
NEO4J_URI = "neo4j+s://1da86c19.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "HA4iunTOGen7RYpeISs3ZRhcWjpcokqam9przCqCuQ8"

# ─────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────
# Initialize
retriever = FPLGraphRetrieval(
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
    hf_token=secret_value_1,  # Optional, for LLM mode
    use_llm=True  # Set True to use LLM for entity extraction
)




from openai import OpenAI
from kaggle_secrets import UserSecretsClient
def call_llm(user_question: str, model: str, embedding_type: str = 'numeric') -> str:
    """
    Takes user question, handles retrieval with specified embedding type.
    """

    # 1️⃣ Get API key
    user_secrets = UserSecretsClient()
    openrouter_key = user_secrets.get_secret("openrouterkey")

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
    
    "=== QUERY TYPE INSTRUCTIONS ===\n\n"
    
    "1. COMPARISON QUERIES (e.g., 'compare X and Y'):\n"
    "   • First check if BOTH players are mentioned in the CONTEXT\n"
    "   • Extract ALL their stats from the CONTEXT\n"
    "   • Create a detailed comparison table or bullet points\n"
    "   • Include: goals, points, assists, clean sheets, bonus points\n"
    "   • Provide a clear conclusion about who performed better\n"
    "   EXAMPLE:\n"
    "   Comparison: Player A vs Player B\n"
    "   • Player A: X goals, Y points, Z assists, W bonus\n"
    "   • Player B: A goals, B points, C assists, D bonus\n"
    "   Conclusion: [Who was better and why]\n\n"
    
    "2. TOP/RANKING QUERIES (e.g., 'top 10 scorers', 'best defenders'):\n"
    "   • Extract the exact ranking from the CONTEXT\n"
    "   • List players in numbered order (1, 2, 3...)\n"
    "   • Include key stats: goals/points/assists next to each name\n"
    "   • Format: '1. Player Name - X goals, Y points'\n"
    "   • If asking for top N, provide EXACTLY N players\n\n"
    
    "3. PLAYER SUMMARY QUERIES (e.g., 'how did X perform', 'X's stats'):\n"
    "   • Extract ALL available stats for that player from CONTEXT\n"
    "   • Include: total points, goals, assists, minutes, clean sheets, bonus, cards\n"
    "   • Present in bullet format or table\n"
    "   • Add brief analysis if stats are exceptional\n\n"
    
    "4. TEAM/FIXTURE QUERIES (e.g., 'team fixtures', 'upcoming matches'):\n"
    "   • List fixtures chronologically by gameweek\n"
    "   • Format: 'GW X: Home Team vs Away Team (Date)'\n"
    "   • Include scores if available in CONTEXT\n"
    "   • Group by team if querying specific team\n\n"
    
    "5. GAMEWEEK PERFORMANCE QUERIES (e.g., 'GW 20 performance'):\n"
    "   • Focus on specific gameweek data from CONTEXT\n"
    "   • Include: points scored, goals, assists, minutes played\n"
    "   • Mention opponent if available\n"
    "   • Note if player didn't play (0 minutes)\n\n"
    
    "6. LIST QUERIES (e.g., 'players on Manchester City'):\n"
    "   • Provide complete list from CONTEXT\n"
    "   • Group by position if relevant (GK, DEF, MID, FWD)\n"
    "   • Use bullet points or numbered list\n"
    "   • Include position labels\n\n"
    
    "7. STATISTICAL QUERIES (e.g., 'most cards', 'clean sheet leaders'):\n"
    "   • Focus on the specific statistic asked\n"
    "   • Rank players by that stat\n"
    "   • Show the actual numbers clearly\n"
    "   • Format: 'Player Name: X [stat]'\n\n"
    
    "=== GENERAL RULES ===\n"
    "• ALWAYS extract data from CONTEXT - never make up stats\n"
    "• If data is not in CONTEXT, say 'Information not available in context'\n"
    "• Use clear formatting: bullet points, numbering, or tables\n"
    "• Include units (points, goals, assists) with numbers\n"
    "• Be concise but complete - include all relevant data\n"
    "• For numerical data, preserve exact values from CONTEXT\n"
    "• If multiple entries exist (e.g., player played multiple GWs), show all\n\n"

    
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
        "1": "mistralai/devstral-2512:free",
        "2":  "meta-llama/llama-3.3-70b-instruct:free",
        "3": "nvidia/nemotron-3-nano-30b-a3b:free",
    }
    
    test_queries = [
        {
            "question": "Who are the top 10 forwards in the 2022-23 season?",
            "ground_truth": """
Erling Haaland,272
Harry Kane,263
Ivan Toney,182
Ollie Watkins,175
Callum Wilson,157
Bryan Mbeumo,150
Dominic Solanke,130
Gabriel Fernando de Jesus,125
Brennan Johnson,122
Aleksandar Mitrović,107"""
        },
        {
            "question": "How did Erling Haaland perform in gameweek 20 of the 2022-23 season?",
            "ground_truth": """
Erling Haaland,points 2,goals 0,assists 0,minutes 90
Erling Haaland,points 6,goals 1, assists 0,minutes 89"""
        },
        {
            "question": "Compare Erling Haaland and Harry Kane's performance in the 2022-23 season",
            "ground_truth": """player,total_points,goals,assists
Erling Haaland,total_points 272,goals 36,assists 9
Harry Kane,total_points 263,goals 30,assists 9"""
        },
        {
            "question": "What were Manchester City's fixtures in the 2022-23 season?",
            "ground_truth": """
1,West Ham,Man City,2022-08-07 15:30:00+00:00
3,Newcastle,Man City,2022-08-21 15:30:00+00:00
6,Aston Villa,Man City,2022-09-03 16:30:00+00:00
8,Wolves,Man City,2022-09-17 11:30:00+00:00
11,Liverpool,Man City,2022-10-16 15:30:00+00:00
14,Leicester,Man City,2022-10-29 11:30:00+00:00
17,Leeds,Man City,2022-12-28 20:00:00+00:00
19,Chelsea,Man City,2023-01-05 20:00:00+00:00
20,Man Utd,Man City,2023-01-14 12:30:00+00:00
22,Spurs,Man City,2023-02-05 16:30:00+00:00
23,Arsenal,Man City,2023-02-15 19:30:00+00:00
24,Nott'm Forest,Man City,2023-02-18 15:00:00+00:00
25,Bournemouth,Man City,2023-02-25 17:30:00+00:00
27,Crystal Palace,Man City,2023-03-11 17:30:00+00:00
30,Southampton,Man City,2023-04-08 16:30:00+00:00
34,Fulham,Man City,2023-04-30 13:00:00+00:00
36,Everton,Man City,2023-05-14 13:00:00+00:00
37,Brighton,Man City,2023-05-24 19:00:00+00:00
38,Brentford,Man City,2023-05-28 15:30:00+00:00"""
        },
        {
            "question": "Which players are on Manchester City?",
            "ground_truth": """
player,position
Alex Robertson,MID
Aymeric Laporte,DEF
Aymeric Laporte,DEF
Ben Knight,MID
Benjamin Mendy,DEF
Bernardo Mota Veiga de Carvalho e Silva,MID
Bernardo Veiga de Carvalho e Silva,MID
Cieran Slicker,GK
Claudio Gomes,MID
Cole Palmer,MID
Cole Palmer,MID
Conrad Egan-Riley,DEF
Ederson Santana de Moraes,GK
Ederson Santana de Moraes,GK
Erling Haaland,FWD
Fernando Luiz Rosa,MID
Ferran Torres,MID
Gabriel Fernando de Jesus,FWD
Ilkay Gündogan,MID
Ilkay Gündogan,MID
Jack Grealish,MID
Jack Grealish,MID
James McAtee,MID
James McAtee,MID
John Stones,DEF
John Stones,FWD
John Stones,DEF
Josh Wilson-Esbrand,DEF
Joshua Wilson-Esbrand,DEF
João Cancelo,DEF
João Pedro Cavaco Cancelo,DEF
Julián Álvarez,FWD
Kalvin Phillips,MID
Kayky da Silva Chagas,FWD
Kevin De Bruyne,MID
Kevin De Bruyne,MID
Kyle Walker,DEF
Kyle Walker,DEF
Liam Delap,FWD
Liam Delap,FWD
Luke Mbete,DEF
Luke Mbete-Tabu,DEF
Manuel Akanji,DEF
Máximo Perrone,MID
Nathan Aké,DEF
Nathan Aké,DEF
Nico O'Reilly,MID
Oleksandr Zinchenko,DEF
Phil Foden,MID
Phil Foden,MID
Raheem Sterling,MID
Rico Lewis,DEF
Riyad Mahrez,MID
Riyad Mahrez,MID
Rodrigo Hernandez,MID
Rodrigo Hernandez,MID
Romeo Lavia,MID
Rúben Gato Alves Dias,DEF
Rúben Santos Gato Alves Dias,DEF
Samuel Edozie,MID
Scott Carson,GK
Scott Carson,GK
Sergio Gómez,DEF
Shea Charles,MID
Stefan Ortega Moreno,GK
Tommy Doyle,MID
Zack Steffen,GK
Zack Steffen,GK"""
        },
        {
            "question": "Who were the top 10 goal scorers in the 2022-23 season?",
            "ground_truth": """
Erling Haaland,36
Harry Kane,30
Ivan Toney,20
Mohamed Salah,19
Callum Wilson,18
Marcus Rashford,17
Ollie Watkins,15
Gabriel Martinelli Silva,15
Martin Ødegaard,15
Aleksandar Mitrović,14"""
        },
        {
            "question": "Who provided the most assists in the 2022-23 season? Show me the top 10?",
            "ground_truth": """
Kevin De Bruyne,18
Mohamed Salah,13
Morgan Gibbs-White,12
Riyad Mahrez,12
Bukayo Saka,12
Michael Olise,11
Trent Alexander-Arnold,11
Jack Grealish,10
Andreas Hoelgebaum Pereira,10
Solly March,10"""
        },
        {
            "question": "Which players had the most clean sheets in the 2022-23 season?",
            "ground_truth": """
Bruno Borges Fernandes,18
David De Gea Quintana,17
Kieran Trippier,16
Fabian Schär,15
Miguel Almirón Rejala,15
Benjamin White,15
Dan Burn,14
Alisson Ramses Becker,14
Gabriel Martinelli Silva,14
Aaron Ramsdale,14"""
        },
        {
            "question": "Who are the top 10 players in form as of gameweek 20 in the 2022-23 season?",
            "ground_truth": """
Martin Ødegaard,51
Solly March,46
Kieran Trippier,44
Harry Kane,43
Marcus Rashford,41
Bruno Borges Fernandes,40
Riyad Mahrez,39
Ivan Toney,39
Luke Shaw,38
Christian Eriksen,35"""
        },
       {
            "question": "Give me a complete summary of Erling Haaland's 2022-23 season?",
            "ground_truth": """player,total_points,goals,assists,clean_sheets,minutes,bonus_points
Erling Haaland,272,36,9,13,2767,40"""
        },
       {
            "question": "Who earned the most bonus points in the 2022-23 season?",
            "ground_truth": """player,total_bonus
Harry Kane,48
Erling Haaland,40
Kieran Trippier,39
Ivan Toney,35
Martin Ødegaard,30
Kevin De Bruyne,26
Ollie Watkins,25
Mohamed Salah,23
Bruno Borges Fernandes,23
Trent Alexander-Arnold,21"""
        },
      
        {
            "question": "Which players received the most yellow and red cards in the 2022-23 season?",
            "ground_truth": """player,total_yellow,total_red,total_cards
João Palhinha Gonçalves,total_yellow 14,total_red 0,total_cards 14
Rúben da Silva Neves,total_yellow 12,total_red 0,total_cards 12
Nélson Cabral Semedo,total_yellow 11,total_red 1,12
Joelinton Cássio Apolinário de Lira,total_yellow 12,total_red 0,total_cards 12
Fabio Henrique Tavares,total_yellow 11,total_red 0,total_cards 11
Adam Smith,total_yellow 11,total_red 0,total_cards 11
James Maddison,total_yellow 10,total_red 0,total_cards 10
Moisés Caicedo Corozo,total_yellow 10,total_red 0,total_cards 10
Conor Gallagher,total_yellow 9,total_red 1,total_cards 10
Cristian Romero,total_yellow 9,total_red 1,total_cards 10"""
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




def extract_entities_and_numbers(text: str) -> dict:
    """
    Extract player names, team names, and numbers from text.
    Returns dictionary with entities and numbers.
    """
    import re
    
    text_lower = text.lower()
    
    # Extract numbers (integers and decimals)
    numbers = set(re.findall(r'\b\d+\.?\d*\b', text))
    
    # Extract entities more aggressively
    entities = set()
    
    # Method 1: Extract from lines (handles lists)
    lines = text.split('\n')
    for line in lines:
        # Remove common prefixes (numbers, bullets, etc)
        clean_line = re.sub(r'^\s*[\d\.\-\*]+\.?\s*', '', line)
        clean_line = re.sub(r'\([^)]*\)', '', clean_line)  # Remove parentheses content
        
        # Extract capitalized word sequences
        words = clean_line.split()
        current_name = []
        
        for word in words:
            clean_word = re.sub(r'[^\w\s\-\']', '', word)
            
            # Stop at position markers or common words
            if clean_word.lower() in ['gk', 'def', 'mid', 'fwd', 'goalkeeper', 'defender', 'midfielder', 'forward']:
                if current_name:
                    entities.add(' '.join(current_name).lower())
                    current_name = []
                continue
            
            # Check if looks like a name part
            if clean_word and len(clean_word) > 1:
                # Add if capitalized or part of ongoing name
                if clean_word[0].isupper() or current_name:
                    current_name.append(clean_word.lower())
                elif current_name:
                    # End of name
                    entities.add(' '.join(current_name).lower())
                    current_name = []
        
        if current_name:
            entities.add(' '.join(current_name).lower())
    
    # Method 2: Also extract last names (single capitalized words > 3 chars)
    all_words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
    for word in all_words:
        entities.add(word.lower())
    
    # Method 3: Extract hyphenated names
    hyphenated = re.findall(r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+)+\b', text)
    for name in hyphenated:
        entities.add(name.lower())
    
    return {
        'entities': entities,
        'numbers': numbers
    }


def calculate_accuracy(answer: str, ground_truth: str) -> dict:
    """
    Calculate accuracy by comparing entities and numbers.
    More lenient matching - focuses on whether entities are present.
    Returns dict with accuracy score and details.
    """
    import re
    
    # Check if model refused to answer or has no information
    refusal_patterns = [
        'does not contain', 'not contain', 'no information', 'not available',
        'cannot find', 'unable to find', 'unfortunately', 'i don\'t have',
        'not provided', 'insufficient information', 'no data'
    ]
    
    answer_lower = answer.lower()
    if any(pattern in answer_lower for pattern in refusal_patterns):
        # Model refused to answer - automatic fail
        return {
            'accuracy': 0.0,
            'is_correct': False,
            'entity_precision': 0.0,
            'entity_recall': 0.0,
            'entity_f1': 0.0,
            'number_accuracy': 0.0,
            'entity_matches': "0/0",
            'number_matches': "0/0",
            'details': 'Model refused to answer or stated no information available'
        }
    
    # Parse ground truth (CSV format)
    truth_lines = [line.strip() for line in ground_truth.strip().split('\n') if line.strip()]
    
    if len(truth_lines) <= 1:
        return {
            'accuracy': 0.0,
            'entity_precision': 0.0,
            'entity_recall': 0.0,
            'number_accuracy': 0.0,
            'details': 'No ground truth data'
        }
    
    # Extract expected entities (skip header)
    expected_entities = set()
    expected_numbers = set()
    
    for line in truth_lines[1:]:
        parts = [p.strip() for p in line.split(',')]
        if parts:
            # First column is usually the entity name
            entity_name = parts[0].lower()
            expected_entities.add(entity_name)
            
            # Extract last name (most important for matching)
            name_parts = entity_name.split()
            if name_parts:
                # Add last name
                expected_entities.add(name_parts[-1])
                # Add first name if exists
                if len(name_parts) > 1:
                    expected_entities.add(name_parts[0])
                # Add middle parts
                for part in name_parts:
                    if len(part) > 3:
                        expected_entities.add(part)
            
            # Extract numbers
            for part in parts[1:]:
                if re.match(r'^\d+\.?\d*$', part):
                    expected_numbers.add(part)
    
    # Remove duplicates from expected entities (since ground truth has dupes)
    unique_expected = set()
    for entity in expected_entities:
        # Only keep substantial entities
        if len(entity) > 2:
            unique_expected.add(entity)
    
    # Extract entities from answer
    answer_data = extract_entities_and_numbers(answer)
    found_entities = answer_data['entities']
    found_numbers = answer_data['numbers']
    
    # Calculate entity matching with flexible approach
    matched_entities = set()
    for expected in unique_expected:
        for found in found_entities:
            # Exact match
            if expected == found:
                matched_entities.add(expected)
                break
            # Substring match (for partial names)
            elif expected in found or found in expected:
                matched_entities.add(expected)
                break
            # Check if all words of expected are in found
            expected_words = set(expected.split())
            found_words = set(found.split())
            if expected_words and expected_words.issubset(found_words):
                matched_entities.add(expected)
                break
    
    entity_matches = len(matched_entities)
    
    # For lists without numbers, don't penalize
    has_numbers = len(expected_numbers) > 0
    
    if has_numbers:
        # Calculate entity precision/recall
        entity_precision = entity_matches / len(found_entities) if found_entities else 0
        entity_recall = entity_matches / len(unique_expected) if unique_expected else 0
        entity_f1 = 2 * (entity_precision * entity_recall) / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0
        
        # Calculate number matching
        number_matches = len(expected_numbers & found_numbers)
        number_accuracy = number_matches / len(expected_numbers) if expected_numbers else 0
        
        # Combined accuracy (70% entities, 30% numbers)
        final_accuracy = (entity_f1 * 0.7) + (number_accuracy * 0.3)
    else:
        # For pure list queries (like "players on team"), use recall only
        # If we found at least 50% of expected entities, it's good
        entity_recall = entity_matches / len(unique_expected) if unique_expected else 0
        entity_precision = 1.0  # Don't penalize for finding extra entities
        entity_f1 = entity_recall  # Just use recall
        number_accuracy = 1.0  # No numbers expected
        
        # For list queries, lower threshold
        final_accuracy = entity_recall
    
    # Determine if correct (lower threshold for list queries)
    threshold = 0.3 if not has_numbers else 0.5
    is_correct = final_accuracy > threshold
    
    return {
        'accuracy': final_accuracy,
        'is_correct': is_correct,
        'entity_precision': entity_precision,
        'entity_recall': entity_recall,
        'entity_f1': entity_f1,
        'number_accuracy': number_accuracy,
        'entity_matches': f"{entity_matches}/{len(unique_expected)}",
        'number_matches': f"{len(expected_numbers & found_numbers)}/{len(expected_numbers)}" if expected_numbers else "N/A",
        'details': f"Entities: {entity_matches}/{len(unique_expected)}" + (f", Numbers: {len(expected_numbers & found_numbers)}/{len(expected_numbers)}" if expected_numbers else " (list query)")
    }


def evaluate_all_models():
    """
    Evaluate multiple models on predefined test queries.
    
    Measures:
    - Accuracy (NER-based entity and number matching)
    - Response time (seconds)
    - Tokens used
    
    Returns:
        Dictionary with individual results and comparison table
    """
    import time
    import pandas as pd
    
    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================
    
    models = {
        "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct:free",
        "Devstral 2512": "mistralai/devstral-2512:free",
        "Nemotron Nano 30B": "nvidia/nemotron-3-nano-30b-a3b:free",
    }
    
    # =========================================================================
    # TEST QUERIES CONFIGURATION
    # =========================================================================
    
    queries_with_truths = queries_with_truths = [
    {
        "question": "Who are the top 10 forwards in the 2022-23 season?",
        "ground_truth": """player,season_points
Erling Haaland,272
Harry Kane,263
Ivan Toney,182
Ollie Watkins,175
Callum Wilson,157
Bryan Mbeumo,150
Dominic Solanke,130
Gabriel Fernando de Jesus,125
Brennan Johnson,122
Aleksandar Mitrović,107"""
    },
    {
        "question": "How did Erling Haaland perform in gameweek 20 of the 2022-23 season?",
        "ground_truth": """player,points,goals,assists,minutes
Erling Haaland,2,0,0,90
Erling Haaland,6,1,0,89"""
    },
    {
        "question": "Compare Erling Haaland and Harry Kane's performance in the 2022-23 season",
        "ground_truth": """player,total_points,goals,assists
Erling Haaland,272,36,9
Harry Kane,263,30,9"""
    },
    {
        "question": "What were Manchester City's fixtures in the 2022-23 season?",
        "ground_truth": """gameweek,home_team,away_team,kickoff_time
1,West Ham,Man City,2022-08-07 15:30:00+00:00
3,Newcastle,Man City,2022-08-21 15:30:00+00:00
6,Aston Villa,Man City,2022-09-03 16:30:00+00:00
8,Wolves,Man City,2022-09-17 11:30:00+00:00
11,Liverpool,Man City,2022-10-16 15:30:00+00:00
14,Leicester,Man City,2022-10-29 11:30:00+00:00
17,Leeds,Man City,2022-12-28 20:00:00+00:00
19,Chelsea,Man City,2023-01-05 20:00:00+00:00
20,Man Utd,Man City,2023-01-14 12:30:00+00:00
22,Spurs,Man City,2023-02-05 16:30:00+00:00
23,Arsenal,Man City,2023-02-15 19:30:00+00:00
24,Nott'm Forest,Man City,2023-02-18 15:00:00+00:00
25,Bournemouth,Man City,2023-02-25 17:30:00+00:00
27,Crystal Palace,Man City,2023-03-11 17:30:00+00:00
30,Southampton,Man City,2023-04-08 16:30:00+00:00
34,Fulham,Man City,2023-04-30 13:00:00+00:00
36,Everton,Man City,2023-05-14 13:00:00+00:00
37,Brighton,Man City,2023-05-24 19:00:00+00:00
38,Brentford,Man City,2023-05-28 15:30:00+00:00"""
    },
    {
        "question": "Which players are on Manchester City?",
        "ground_truth": """player,position
Alex Robertson,MID
Aymeric Laporte,DEF
Aymeric Laporte,DEF
Ben Knight,MID
Benjamin Mendy,DEF
Bernardo Mota Veiga de Carvalho e Silva,MID
Bernardo Veiga de Carvalho e Silva,MID
Cieran Slicker,GK
Claudio Gomes,MID
Cole Palmer,MID
Cole Palmer,MID
Conrad Egan-Riley,DEF
Ederson Santana de Moraes,GK
Ederson Santana de Moraes,GK
Erling Haaland,FWD
Fernando Luiz Rosa,MID
Ferran Torres,MID
Gabriel Fernando de Jesus,FWD
Ilkay Gündogan,MID
Ilkay Gündogan,MID
Jack Grealish,MID
Jack Grealish,MID
James McAtee,MID
James McAtee,MID
John Stones,DEF
John Stones,FWD
John Stones,DEF
Josh Wilson-Esbrand,DEF
Joshua Wilson-Esbrand,DEF
João Cancelo,DEF
João Pedro Cavaco Cancelo,DEF
Julián Álvarez,FWD
Kalvin Phillips,MID
Kayky da Silva Chagas,FWD
Kevin De Bruyne,MID
Kevin De Bruyne,MID
Kyle Walker,DEF
Kyle Walker,DEF
Liam Delap,FWD
Liam Delap,FWD
Luke Mbete,DEF
Luke Mbete-Tabu,DEF
Manuel Akanji,DEF
Máximo Perrone,MID
Nathan Aké,DEF
Nathan Aké,DEF
Nico O'Reilly,MID
Oleksandr Zinchenko,DEF
Phil Foden,MID
Phil Foden,MID
Raheem Sterling,MID
Rico Lewis,DEF
Riyad Mahrez,MID
Riyad Mahrez,MID
Rodrigo Hernandez,MID
Rodrigo Hernandez,MID
Romeo Lavia,MID
Rúben Gato Alves Dias,DEF
Rúben Santos Gato Alves Dias,DEF
Samuel Edozie,MID
Scott Carson,GK
Scott Carson,GK
Sergio Gómez,DEF
Shea Charles,MID
Stefan Ortega Moreno,GK
Tommy Doyle,MID
Zack Steffen,GK
Zack Steffen,GK"""
    },
    {
        "question": "Who were the top 10 goal scorers in the 2022-23 season?",
        "ground_truth": """player,total_goals
Erling Haaland,36
Harry Kane,30
Ivan Toney,20
Mohamed Salah,19
Callum Wilson,18
Marcus Rashford,17
Ollie Watkins,15
Gabriel Martinelli Silva,15
Martin Ødegaard,15
Aleksandar Mitrović,14"""
    },
    {
        "question": "Who provided the most assists in the 2022-23 season? Show me the top 10",
        "ground_truth": """player,total_assists
Kevin De Bruyne,18
Mohamed Salah,13
Morgan Gibbs-White,12
Riyad Mahrez,12
Bukayo Saka,12
Michael Olise,11
Trent Alexander-Arnold,11
Jack Grealish,10
Andreas Hoelgebaum Pereira,10
Solly March,10"""
    },
    {
        "question": "Which players had the most clean sheets in the 2022-23 season?",
        "ground_truth": """player,total_clean_sheets
Bruno Borges Fernandes,18
David De Gea Quintana,17
Kieran Trippier,16
Fabian Schär,15
Miguel Almirón Rejala,15
Benjamin White,15
Dan Burn,14
Alisson Ramses Becker,14
Gabriel Martinelli Silva,14
Aaron Ramsdale,14"""
    },
    {
        "question": "Who are the top 10 players in form as of gameweek 20 in the 2022-23 season?",
        "ground_truth": """player,form_points
Martin Ødegaard,51
Solly March,46
Kieran Trippier,44
Harry Kane,43
Marcus Rashford,41
Bruno Borges Fernandes,40
Riyad Mahrez,39
Ivan Toney,39
Luke Shaw,38
Christian Eriksen,35"""
    },
    {
        "question": "Give me a complete summary of Erling Haaland's 2022-23 season",
        "ground_truth": """player,total_points,goals,assists,clean_sheets,minutes,bonus_points
Erling Haaland,272,36,9,13,2767,40"""
    },
    {
        "question": "Who earned the most bonus points in the 2022-23 season?",
        "ground_truth": """player,total_bonus
Harry Kane,48
Erling Haaland,40
Kieran Trippier,39
Ivan Toney,35
Martin Ødegaard,30
Kevin De Bruyne,26
Ollie Watkins,25
Mohamed Salah,23
Bruno Borges Fernandes,23
Trent Alexander-Arnold,21"""
    },
    {
        "question": "Which players received the most yellow and red cards in the 2022-23 season?",
        "ground_truth": """player,total_yellow,total_red,total_cards
João Palhinha Gonçalves,14,0,14
Rúben da Silva Neves,12,0,12
Nélson Cabral Semedo,11,1,12
Joelinton Cássio Apolinário de Lira,12,0,12
Fabio Henrique Tavares,11,0,11
Adam Smith,11,0,11
James Maddison,10,0,10
Moisés Caicedo Corozo,10,0,10
Conor Gallagher,9,1,10
Cristian Romero,9,1,10"""
    }
]
    
    # =========================================================================
    # EVALUATE ALL MODELS
    # =========================================================================
    
    all_model_results = {}
    
    for model_name, model_id in models.items():
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"{'='*80}\n")
        
        results = []
        total_response_time = 0
        total_tokens = 0
        total_accuracy = 0
        correct_count = 0
        evaluated_count = 0
        
        for idx, item in enumerate(queries_with_truths, 1):
            question = item["question"]
            ground_truth = item.get("ground_truth")
            
            print(f"{'─'*80}")
            print(f"Query {idx}/{len(queries_with_truths)}")
            print(f"{'─'*80}")
            print(f"❓ Question: {question}")
            
            start = time.time()
            try:
                answer, tokens_used = call_llm(question, model_id)
                answer_preview = str(answer)[:300] + "..." if len(str(answer)) > 300 else str(answer)
                print(f"🤖 Model Answer: {answer_preview}")
                
                # Print ground truth for comparison
                if ground_truth:
                    truth_preview = ground_truth[:200] + "..." if len(ground_truth) > 200 else ground_truth
                    print(f"📋 Ground Truth: {truth_preview}")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": None,
                    "error": str(e),
                    "response_time": None,
                    "tokens_used": 0,
                    "accuracy": None
                })
                print()
                continue
            
            end = time.time()
            response_time = end - start
            total_tokens += tokens_used
            total_response_time += response_time
            
            # Calculate accuracy
            accuracy_result = calculate_accuracy(str(answer), ground_truth)
            accuracy_score = accuracy_result['accuracy']
            is_correct = accuracy_result['is_correct']
            
            total_accuracy += accuracy_score
            if is_correct:
                correct_count += 1
            evaluated_count += 1
            
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"\n{status}")
            print(f"📊 Accuracy: {accuracy_score:.2%}")
            print(f"   Entity F1: {accuracy_result['entity_f1']:.2%}")
            print(f"   Number Match: {accuracy_result['number_accuracy']:.2%}")
            print(f"   Details: {accuracy_result['details']}")
            print(f"⏱️  Response Time: {response_time:.3f}s")
            print(f"🔢 Tokens Used: {tokens_used}")
            print()
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": str(answer),
                "response_time": response_time,
                "tokens_used": tokens_used,
                "accuracy": accuracy_score,
                "is_correct": is_correct,
                "accuracy_details": accuracy_result
            })
        
        num_queries = len(queries_with_truths)
        avg_response_time = total_response_time / num_queries if num_queries > 0 else 0
        avg_tokens = total_tokens / num_queries if num_queries > 0 else 0
        avg_accuracy = (total_accuracy / evaluated_count) * 100 if evaluated_count > 0 else 0
        correct_percentage = (correct_count / evaluated_count) * 100 if evaluated_count > 0 else 0
        
        all_model_results[model_name] = {
            "model_id": model_id,
            "results": results,
            "average_response_time": avg_response_time,
            "average_tokens_used": avg_tokens,
            "average_accuracy": avg_accuracy,
            "correct_percentage": correct_percentage,
            "correct_count": correct_count,
            "total_queries": evaluated_count
        }
        
        print(f"{'='*80}")
        print(f"SUMMARY FOR {model_name}")
        print(f"{'='*80}")
        print(f"Total Queries: {num_queries}")
        print(f"Correct Answers: {correct_count}/{evaluated_count} ({correct_percentage:.1f}%)")
        print(f"Average Accuracy Score: {avg_accuracy:.1f}%")
        print(f"Average Response Time: {avg_response_time:.3f}s")
        print(f"Average Tokens Used: {avg_tokens:.1f}")
        print(f"{'='*80}\n")
    
    # =========================================================================
    # CREATE COMPARISON TABLE
    # =========================================================================
    
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    comparison_data = []
    for model_name, metrics in all_model_results.items():
        comparison_data.append({
            "Model": model_name,
            "Correct (%)": f"{metrics['correct_percentage']:.1f}",
            "Avg Accuracy": f"{metrics['average_accuracy']:.1f}%",
            "Avg Time (s)": f"{metrics['average_response_time']:.3f}",
            "Avg Tokens": f"{metrics['average_tokens_used']:.0f}"
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    print(f"\n{'='*80}\n")
    
    # Find best model for each metric
    print("🏆 BEST PERFORMERS:")
    print(f"{'─'*80}")
    
    best_accuracy = max(all_model_results.items(), key=lambda x: x[1]['average_accuracy'])
    print(f"Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['average_accuracy']:.1f}%)")
    
    best_correct = max(all_model_results.items(), key=lambda x: x[1]['correct_percentage'])
    print(f"Most Correct: {best_correct[0]} ({best_correct[1]['correct_count']}/{best_correct[1]['total_queries']})")
    
    best_speed = min(all_model_results.items(), key=lambda x: x[1]['average_response_time'])
    print(f"Fastest: {best_speed[0]} ({best_speed[1]['average_response_time']:.3f}s)")
    
    best_efficiency = min(all_model_results.items(), key=lambda x: x[1]['average_tokens_used'])
    print(f"Most Efficient: {best_efficiency[0]} ({best_efficiency[1]['average_tokens_used']:.0f} tokens)")
    
    print(f"{'='*80}\n")
    
    return {
        "individual_results": all_model_results,
        "comparison_table": df,
        "summary": {
            "total_models_evaluated": len(models),
            
        }
    }



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


