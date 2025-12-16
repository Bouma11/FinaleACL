"""
Milestone 3 - System Requirements Part 2: Graph Retrieval Layer
Implements:
  2.a - Baseline: Cypher query templates (10+ queries)
  2.b - Embeddings: Semantic similarity search (2 models for comparison)
"""

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Any
from fpl_input_preprocessing import FPLInputPreprocessorLLM as FPLInputPreprocessor


class FPLGraphRetrieval:
    """
    Graph Retrieval Layer for FPL Graph-RAG system.
    Combines baseline Cypher queries with embedding-based semantic search.
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 hf_token: str = None, use_llm: bool = False):
        """
        Initialize Neo4j connection and embedding models.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            hf_token: HuggingFace API token (required if use_llm=True)
            use_llm: Use LLM for intent/entity extraction (default: False for rule-based)
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        self.preprocessor = FPLInputPreprocessor(
            hf_token=hf_token or "",
            use_llm=use_llm and hf_token is not None
        )
        
        # Two embedding models for comparison (requirement 2.b)
        self.embedding_models = {
            'model_1': SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),
            'model_2': SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        }
        self.active_model = 'model_1'
        
        self._init_query_templates()
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()

    # =========================================================================
    # SECTION 2.a: BASELINE - CYPHER QUERY TEMPLATES
    # =========================================================================
    
    def _init_query_templates(self):
        """Initialize 12 FIXED Cypher query templates for different intents."""
        self.query_templates = {
            
            'top_players_by_position': """
                // Top players by position for a specific season
                MATCH (gw:Gameweek {season: $season})<-[:HAS_FIXTURE]-(f:Fixture)
                MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: $position})
                MATCH (p)-[played:PLAYED_IN]->(f)
                MATCH (p)-[:PLAYS_FOR]->(t:Team)
                RETURN p.player_name AS player,
                    t.name AS team,
                    SUM(played.total_points) AS total_points,
                    SUM(played.goals_scored) AS goals,
                    SUM(played.assists) AS assists,
                    COUNT(DISTINCT f) AS games_played
                ORDER BY total_points DESC
                LIMIT $limit
            """,
            
            'player_gameweek_performance': """
                // Player performance in specific gameweek
                MATCH (gw:Gameweek {season: $season, GW_number: $gameweek})<-[:HAS_FIXTURE]-(f:Fixture)
                MATCH (p:Player {player_name: $player_name})-[played:PLAYED_IN]->(f)
                MATCH (p)-[:PLAYS_FOR]->(t:Team)
                RETURN p.player_name AS player,
                    t.name AS team,
                    played.total_points AS points,
                    played.goals_scored AS goals,
                    played.assists AS assists,
                    played.minutes AS minutes,
                    played.bonus AS bonus,
                    played.clean_sheets AS clean_sheets,
                    played.ict_index AS ict_index
            """,
            
            'compare_players': """
                // Compare two players for a specific season - FIXED VERSION
                MATCH (p1:Player {player_name: $player_name})
                MATCH (p2:Player {player_name: $player_name2})
                
                // Get player 1 stats for the season
                OPTIONAL MATCH (p1)-[played1:PLAYED_IN]->(f1:Fixture)
                WHERE EXISTS {
                    MATCH (f1)<-[:HAS_FIXTURE]-(:Gameweek {season: $season})
                }
                
                // Get player 2 stats for the season
                OPTIONAL MATCH (p2)-[played2:PLAYED_IN]->(f2:Fixture)
                WHERE EXISTS {
                    MATCH (f2)<-[:HAS_FIXTURE]-(:Gameweek {season: $season})
                }
                
                // Get teams
                OPTIONAL MATCH (p1)-[:PLAYS_FOR]->(t1:Team)
                OPTIONAL MATCH (p2)-[:PLAYS_FOR]->(t2:Team)
                
                // Aggregate with DISTINCT to avoid duplicates
                WITH p1, t1, 
                    SUM(DISTINCT played1.total_points) AS p1_points,
                    SUM(DISTINCT played1.goals_scored) AS p1_goals,
                    SUM(DISTINCT played1.assists) AS p1_assists,
                    COUNT(DISTINCT f1) AS p1_games,
                    p2, t2,
                    SUM(DISTINCT played2.total_points) AS p2_points,
                    SUM(DISTINCT played2.goals_scored) AS p2_goals,
                    SUM(DISTINCT played2.assists) AS p2_assists,
                    COUNT(DISTINCT f2) AS p2_games
                
                RETURN p1.player_name AS player1, 
                    COALESCE(t1.name, 'Unknown') AS team1, 
                    COALESCE(p1_points, 0) AS p1_points,
                    COALESCE(p1_goals, 0) AS p1_goals,
                    COALESCE(p1_assists, 0) AS p1_assists,
                    p1_games,
                    
                    p2.player_name AS player2, 
                    COALESCE(t2.name, 'Unknown') AS team2, 
                    COALESCE(p2_points, 0) AS p2_points,
                    COALESCE(p2_goals, 0) AS p2_goals,
                    COALESCE(p2_assists, 0) AS p2_assists,
                    p2_games
            """,
            
            'team_fixtures': """
                // Team fixtures for a specific season
                MATCH (t:Team {name: $team})
                MATCH (f:Fixture)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)
                MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek {season: $season})
                MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
                MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
                RETURN gw.GW_number AS gameweek,
                    home.name AS home_team,
                    away.name AS away_team,
                    f.kickoff_time AS kickoff,
                    CASE WHEN home = t THEN 'home' ELSE 'away' END AS venue
                ORDER BY gw.GW_number
            """,
            
            'players_by_team': """
                // Players by team
                MATCH (t:Team {name: $team})
                MATCH (p:Player)-[:PLAYS_FOR]->(t)
                OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                RETURN p.player_name AS player,
                    COALESCE(pos.name, 'Unknown') AS position,
                    p.player_element AS player_id
                ORDER BY position, p.player_name
            """,

            'player_team': """
                // Get team for a specific player
                MATCH (p:Player {player_name: $player_name})-[:PLAYS_FOR]->(t:Team)
                RETURN p.player_name AS player, 
                    t.name AS team,
                    p.player_element AS player_id
            """,
            
            'top_scorers': """
                // Top scorers for a specific season
                MATCH (gw:Gameweek {season: $season})<-[:HAS_FIXTURE]-(f:Fixture)
                MATCH (p:Player)-[played:PLAYED_IN]->(f)
                MATCH (p)-[:PLAYS_FOR]->(t:Team)
                RETURN p.player_name AS player,
                    t.name AS team,
                    SUM(played.goals_scored) AS goals,
                    SUM(played.total_points) AS points,
                    SUM(played.assists) AS assists,
                    COUNT(DISTINCT f) AS games_played
                ORDER BY goals DESC
                LIMIT $limit
            """,
            
            'top_assisters': """
                // Top assisters for a specific season
                MATCH (gw:Gameweek {season: $season})<-[:HAS_FIXTURE]-(f:Fixture)
                MATCH (p:Player)-[played:PLAYED_IN]->(f)
                MATCH (p)-[:PLAYS_FOR]->(t:Team)
                RETURN p.player_name AS player,
                    t.name AS team,
                    SUM(played.assists) AS assists,
                    SUM(played.total_points) AS points,
                    SUM(played.goals_scored) AS goals,
                    COUNT(DISTINCT f) AS games_played
                ORDER BY assists DESC
                LIMIT $limit
            """,
            
            'clean_sheet_leaders': """
                // Clean sheet leaders for a specific season (GK and DEF only)
                MATCH (gw:Gameweek {season: $season})<-[:HAS_FIXTURE]-(f:Fixture)
                MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
                WHERE pos.name IN ['GK', 'DEF']
                MATCH (p)-[played:PLAYED_IN]->(f)
                MATCH (p)-[:PLAYS_FOR]->(t:Team)
                RETURN p.player_name AS player,
                    pos.name AS position,
                    t.name AS team,
                    SUM(played.clean_sheets) AS clean_sheets,
                    SUM(played.total_points) AS points,
                    COUNT(DISTINCT f) AS games_played
                ORDER BY clean_sheets DESC
                LIMIT $limit
            """,
            
            'players_by_form': """
                // Players by recent form (last 5 gameweeks)
                MATCH (gw:Gameweek {season: $season})
                WHERE gw.GW_number >= $gameweek - 5 AND gw.GW_number <= $gameweek
                MATCH (gw)<-[:HAS_FIXTURE]-(f:Fixture)
                MATCH (p:Player)-[played:PLAYED_IN]->(f)
                MATCH (p)-[:PLAYS_FOR]->(t:Team)
                RETURN p.player_name AS player,
                    t.name AS team,
                    AVG(played.form) AS avg_form,
                    SUM(played.total_points) AS recent_points,
                    SUM(played.goals_scored) AS recent_goals,
                    SUM(played.assists) AS recent_assists,
                    COUNT(DISTINCT f) AS games_count
                ORDER BY avg_form DESC
                LIMIT $limit
            """,
            
            'player_season_summary': """
                // Player season summary
                MATCH (p:Player {player_name: $player_name})
                MATCH (gw:Gameweek {season: $season})<-[:HAS_FIXTURE]-(f:Fixture)
                MATCH (p)-[played:PLAYED_IN]->(f)
                MATCH (p)-[:PLAYS_AS]->(pos:Position)
                MATCH (p)-[:PLAYS_FOR]->(t:Team)
                RETURN p.player_name AS player,
                    pos.name AS position,
                    t.name AS team,
                    COUNT(DISTINCT f) AS games_played,
                    SUM(played.minutes) AS total_minutes,
                    SUM(played.total_points) AS total_points,
                    SUM(played.goals_scored) AS goals,
                    SUM(played.assists) AS assists,
                    SUM(played.clean_sheets) AS clean_sheets,
                    SUM(played.bonus) AS bonus_points,
                    AVG(played.ict_index) AS avg_ict,
                    AVG(played.form) AS avg_form
            """,
            
            'bonus_leaders': """
                // Bonus points leaders for a specific season
                MATCH (gw:Gameweek {season: $season})<-[:HAS_FIXTURE]-(f:Fixture)
                MATCH (p:Player)-[played:PLAYED_IN]->(f)
                MATCH (p)-[:PLAYS_FOR]->(t:Team)
                RETURN p.player_name AS player,
                    t.name AS team,
                    SUM(played.bonus) AS total_bonus,
                    SUM(played.total_points) AS total_points,
                    SUM(played.bps) AS bps,
                    COUNT(DISTINCT f) AS games_played
                ORDER BY total_bonus DESC
                LIMIT $limit
            """,
            
            'most_cards': """
                // Players with most cards (yellow/red) for a specific season
                MATCH (gw:Gameweek {season: $season})<-[:HAS_FIXTURE]-(f:Fixture)
                MATCH (p:Player)-[played:PLAYED_IN]->(f)
                MATCH (p)-[:PLAYS_FOR]->(t:Team)
                RETURN p.player_name AS player,
                    t.name AS team,
                    SUM(played.yellow_cards) AS yellows,
                    SUM(played.red_cards) AS reds,
                    SUM(played.yellow_cards) + SUM(played.red_cards) * 2 AS card_score,
                    COUNT(DISTINCT f) AS games_played
                ORDER BY card_score DESC
                LIMIT $limit
            """
        }
        
        # Map intents to query templates (ordered by preference)
        self.intent_to_query = {
        'recommendation': ['top_scorers', 'top_assisters', 'clean_sheet_leaders', 'bonus_leaders', 'top_players_by_position'],
        'performance_query': ['player_gameweek_performance', 'player_season_summary', 'players_by_form'],
        'comparison': ['compare_players', 'top_scorers', 'player_season_summary'],
        'player_search': ['player_season_summary', 'player_gameweek_performance', 'players_by_form'],
        'fixture_query': ['team_fixtures', 'players_by_team'],
        'form_query': ['players_by_form', 'player_season_summary', 'top_scorers'],
        'team_analysis': ['players_by_team', 'team_fixtures', 'top_scorers'],
        'value_analysis': ['top_players_by_position', 'players_by_form', 'bonus_leaders'],
        'general_query': ['top_scorers', 'top_assisters', 'bonus_leaders', 'clean_sheet_leaders'],
        'player_team_query': ['player_team', 'player_season_summary', 'players_by_team']
        }
        self.intent_primary_query = {
        'recommendation': 'top_scorers',
        'performance_query': 'player_gameweek_performance',
        'comparison': 'compare_players',  # This is the bug! Should be 'compare_players'
        'player_search': 'player_season_summary',
        'fixture_query': 'team_fixtures',
        'form_query': 'players_by_form',
        'team_analysis': 'players_by_team',
        'player_team_query': 'player_team',
        'value_analysis': 'top_players_by_position',
        'general_query': 'top_scorers'
    }
    
    def select_query(self, intent: str, entities: Dict) -> str:
        """Select appropriate query template based on intent and available entities.
        
        Priority order (highest to lowest):
        1. Intent-specific with complete entities (highest confidence)
        2. Entity combinations that clearly indicate a specific query type
        3. Intent-based fallback
        4. Generic fallback
        """
        
        metrics = entities.get('metrics', [])
        positions = entities.get('positions', [])
        teams = entities.get('teams', [])
        players = entities.get('players', [])
        gameweeks = entities.get('gameweeks', [])
        
        # =====================================================================
        # PRIORITY 1: HIGH-CONFIDENCE INTENT + ENTITY COMBINATIONS
        # =====================================================================
        
        # 1A: PERFORMANCE QUERY - Single player + specific gameweek
        if intent == 'performance_query' and len(players) >= 1 and len(gameweeks) >= 1:
            return 'player_gameweek_performance'
        
        # 1B: COMPARISON QUERY - Exactly 2 players (no gameweek needed for comparison)
        if intent == 'comparison' and len(players) == 2:
            return 'compare_players'
        
        # 1C: PLAYER SEARCH - Single player, no gameweek
        if intent == 'player_search' and len(players) >= 1 and len(gameweeks) == 0:
            return 'player_season_summary'
        
        # =====================================================================
        # PRIORITY 2: CLEAR ENTITY PATTERNS (regardless of intent)
        # =====================================================================
        
        # 2A: Single player + gameweek = ALWAYS gameweek performance
        if len(players) == 1 and len(gameweeks) >= 1:
            return 'player_gameweek_performance'
        
        # 2B: Two players = comparison (but check if it's REALLY a comparison)
        if len(players) == 2:
            # Extra validation: check if query has comparison words
            query_text = str(entities).lower()
            comparison_words = ['vs', 'versus', 'compare', 'comparison', 'versus']
            if any(word in query_text for word in comparison_words):
                return 'compare_players'
        
                # Two players without comparison words? Might be a mistake
                # Fall through to lower priority checks
        
        # 2C: Team + gameweek = team fixtures
        if len(teams) >= 1 and len(gameweeks) >= 1:
            return 'team_fixtures'
        
        # 2D: Team alone (depends on context)
        if len(teams) >= 1:
            # If asking about "players on team"
            if 'players' in str(entities).lower() or 'roster' in str(entities).lower():
                return 'players_by_team'
            return 'team_fixtures'  # Default for team queries
        
        # 2E: Position alone = top players by position
        if len(positions) >= 1 and len(players) == 0:
            return 'top_players_by_position'
        
        # =====================================================================
        # PRIORITY 3: INTENT-BASED FALLBACK (using corrected mapping)
        # =====================================================================
        
        intent_query_map = {
            'recommendation': 'top_scorers',
            'performance_query': 'player_gameweek_performance',
            'comparison': 'compare_players',
            'player_search': 'player_season_summary',
            'fixture_query': 'team_fixtures',
            'form_query': 'players_by_form',
            'team_analysis': 'players_by_team',
            'value_analysis': 'top_players_by_position',
            'general_query': 'top_scorers',
            'player_team_query': 'player_team'
        }
        
        if intent in intent_query_map:
            return intent_query_map[intent]
        
        # =====================================================================
        # PRIORITY 4: METRIC-SPECIFIC FALLBACK
        # =====================================================================
        
        metric_to_query = {
            'clean_sheets': 'clean_sheet_leaders',
            'assists': 'top_assisters',
            'goals_scored': 'top_scorers',
            'bonus': 'bonus_leaders',
            'cards': 'most_cards',
            'form': 'players_by_form',
            'value': 'top_players_by_position'
        }
        
        for metric in metrics:
            if metric in metric_to_query:
                return metric_to_query[metric]
        
        # =====================================================================
        # PRIORITY 5: SAFE GENERIC FALLBACK
        # =====================================================================
        return 'top_scorers'
    
    def execute_cypher(self, query_name: str, params: Dict) -> List[Dict]:
        """Execute a Cypher query template with parameters."""
        if query_name not in self.query_templates:
            raise ValueError(f"Unknown query template: {query_name}")
        
        query = self.query_templates[query_name]
        
        params.setdefault('limit', 10)
        params.setdefault('season', '2022-23')
        
        # Convert year to season format (e.g., '2022' -> '2022-23')
        if 'season' in params:
            season = params['season']
            if len(season) == 4 and season.isdigit():
                year = int(season)
                params['season'] = f"{year}-{str(year + 1)[-2:]}"
        
        with self.driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]
    
    def baseline_retrieve(self, user_input: str) -> Dict[str, Any]:
        """Baseline retrieval using Cypher queries only."""
        preprocessed = self.preprocessor.preprocess(user_input, include_embedding=False)
        params = self.preprocessor.get_cypher_params(preprocessed)
        query_name = self.select_query(preprocessed['intent'], preprocessed['entities'])
        
        try:
            results = self.execute_cypher(query_name, params)
            return {
                'method': 'baseline',
                'intent': preprocessed['intent'],
                'query_used': query_name,
                'parameters': params,
                'results': results,
                'cypher': self.query_templates[query_name]
            }
        except Exception as e:
            return {
                'method': 'baseline',
                'error': str(e),
                'intent': preprocessed['intent'],
                'query_used': query_name,
                'parameters': params
            }

    # =========================================================================
    # SECTION 2.b: NODE EMBEDDINGS - NUMERIC FEATURE VECTORS
    # =========================================================================
    #
    # APPROACH: Create numeric embeddings from player statistics
    # 
    # Why numeric embeddings for FPL?
    # - FPL data is purely numerical (goals, points, assists, etc.)
    # - No textual features to embed
    # - Direct numeric vectors preserve exact statistical relationships
    # - Faster computation than text-based embeddings
    #
    # Feature Vector Structure (12 dimensions):
    # [goals_norm, assists_norm, points_norm, clean_sheets_norm, minutes_norm,
    #  bonus_norm, form_norm, ict_norm, influence_norm, creativity_norm, 
    #  threat_norm, games_norm]
    #
    # Each feature is normalized to [0, 1] range for fair comparison
    # =========================================================================
    
    def set_embedding_model(self, model_name: str):
        """Switch between embedding models for comparison."""
        if model_name not in self.embedding_models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.embedding_models.keys())}")
        self.active_model = model_name
    
    def fetch_all_players_stats(self, season: str = None, use_per_game_avg: bool = True) -> List[Dict]:
        """
        Fetch ALL players with their stats and LATEST position.
        
        How it works:
        1. Match each unique player by player_element (unique ID)
        2. Get ONE position per player (collect and take first)
        3. Option A (use_per_game_avg=True): Compute PER-GAME AVERAGES
           - This ensures players with high performance per game rank higher
           - Avoids bias toward veterans who played more seasons
        4. Option B (use_per_game_avg=False): Use totals (for season-specific queries)
        5. Return one row per player (no duplicates)
        
        Args:
            season: Optional season filter (e.g., '2022-23'). If None, aggregates all.
            use_per_game_avg: If True, computes per-game averages for fair comparison.
        
        Returns:
            List of player dictionaries with stats
        """
        # Build season filter if provided
        season_filter = ""
        if season:
            season_filter = "MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek {season: $season})"
        
        if use_per_game_avg:
            # Use per-game averages - FAIR comparison across players with different game counts
            query = f"""
                // Get all unique players
                MATCH (p:Player)
                
                // Get position (take first if multiple)
                OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                
                // Aggregate stats from fixtures
                OPTIONAL MATCH (p)-[played:PLAYED_IN]->(f:Fixture)
                {season_filter}
                
                // Group by player_element (unique ID) to avoid duplicates
                // Use PER-GAME AVERAGES for fair comparison
                WITH p.player_element AS player_id,
                     COLLECT(DISTINCT p.player_name)[0] AS player_name,
                     COLLECT(DISTINCT pos.name)[0] AS position,
                     COUNT(played) AS games_played,
                     // Per-game averages (more fair for embedding comparison)
                     COALESCE(AVG(played.total_points), 0) AS avg_points_per_game,
                     COALESCE(SUM(played.goals_scored) * 1.0 / NULLIF(COUNT(played), 0), 0) AS goals_per_game,
                     COALESCE(SUM(played.assists) * 1.0 / NULLIF(COUNT(played), 0), 0) AS assists_per_game,
                     COALESCE(SUM(played.clean_sheets) * 1.0 / NULLIF(COUNT(played), 0), 0) AS clean_sheets_per_game,
                     COALESCE(AVG(played.minutes), 0) AS avg_minutes,
                     COALESCE(AVG(played.bonus), 0) AS avg_bonus,
                     COALESCE(AVG(played.form), 0) AS form,
                     COALESCE(AVG(played.ict_index), 0) AS ict_index,
                     COALESCE(AVG(played.influence), 0) AS influence,
                     COALESCE(AVG(played.creativity), 0) AS creativity,
                     COALESCE(AVG(played.threat), 0) AS threat,
                     // Also keep totals for reference
                     COALESCE(SUM(played.total_points), 0) AS total_points,
                     COALESCE(SUM(played.goals_scored), 0) AS goals,
                     COALESCE(SUM(played.assists), 0) AS assists,
                     COALESCE(SUM(played.clean_sheets), 0) AS clean_sheets
                
                // Filter out any null entries and return
                WHERE player_name IS NOT NULL AND position IS NOT NULL AND games_played > 0
                RETURN player_id, player_name, position, games_played,
                       avg_points_per_game, goals_per_game, assists_per_game,
                       clean_sheets_per_game, avg_minutes, avg_bonus,
                       form, ict_index, influence, creativity, threat,
                       total_points, goals, assists, clean_sheets
                ORDER BY avg_points_per_game DESC
            """
        else:
            # Use totals - for season-specific comparisons
            query = f"""
                MATCH (p:Player)
                OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                OPTIONAL MATCH (p)-[played:PLAYED_IN]->(f:Fixture)
                {season_filter}
                WITH p.player_element AS player_id,
                     COLLECT(DISTINCT p.player_name)[0] AS player_name,
                     COLLECT(DISTINCT pos.name)[0] AS position,
                     COALESCE(SUM(played.total_points), 0) AS total_points,
                     COALESCE(SUM(played.goals_scored), 0) AS goals,
                     COALESCE(SUM(played.assists), 0) AS assists,
                     COALESCE(SUM(played.clean_sheets), 0) AS clean_sheets,
                     COALESCE(SUM(played.minutes), 0) AS minutes,
                     COALESCE(SUM(played.bonus), 0) AS bonus,
                     COALESCE(AVG(played.form), 0) AS form,
                     COALESCE(AVG(played.ict_index), 0) AS ict_index,
                     COALESCE(AVG(played.influence), 0) AS influence,
                     COALESCE(AVG(played.creativity), 0) AS creativity,
                     COALESCE(AVG(played.threat), 0) AS threat,
                     COUNT(played) AS games_played
                WHERE player_name IS NOT NULL AND position IS NOT NULL
                RETURN player_id, player_name, position,
                       total_points, goals, assists, clean_sheets,
                       minutes, bonus, form, ict_index,
                       influence, creativity, threat, games_played
                ORDER BY total_points DESC
            """
        
        params = {'season': season} if season else {}
        
        with self.driver.session() as session:
            result = session.run(query, params)
            players = [dict(record) for record in result]
            
        print(f"📊 Fetched {len(players)} unique players" + (f" (per-game averages)" if use_per_game_avg else ""))
        return players
    
    def compute_normalization_stats(self, players: List[Dict]) -> Dict[str, Dict]:
        """
        Compute min/max for each feature to normalize to [0, 1] range.
        
        Why normalize?
        - Goals per game range: 0-1, Points per game range: 0-15
        - Without normalization, points would dominate similarity
        - Normalized: each feature contributes equally
        
        Uses PER-GAME AVERAGES for fair comparison across players with different game counts.
        
        Returns:
            Dictionary with min/max for each feature
        """
        # Use per-game average features for fair comparison
        features = ['goals_per_game', 'assists_per_game', 'avg_points_per_game', 
                    'clean_sheets_per_game', 'avg_minutes', 'avg_bonus', 
                    'form', 'ict_index', 'influence', 'creativity', 'threat', 'games_played']
        
        # Fallback features if per-game stats not available
        fallback_features = ['goals', 'assists', 'total_points', 'clean_sheets', 
                             'minutes', 'bonus', 'form', 'ict_index',
                             'influence', 'creativity', 'threat', 'games_played']
        
        stats = {}
        for i, feat in enumerate(features):
            # Try per-game feature first, fallback to totals
            values = [p.get(feat, 0) or p.get(fallback_features[i], 0) or 0 for p in players]
            stats[feat] = {
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values) if max(values) != min(values) else 1
            }
        
        return stats
    
    def create_numeric_embedding(self, player: Dict, norm_stats: Dict) -> np.ndarray:
        """
        Create a normalized numeric embedding vector for a player.
        
        Uses PER-GAME AVERAGES for fair comparison:
        - A striker with 1 goal per game ranks higher than one with 0.2 goals/game
        - This prevents bias toward players who simply played more games
        
        Process:
        1. Extract per-game average values (or fallback to totals)
        2. Normalize each to [0, 1] using min-max scaling
        3. Return as numpy array
        
        Formula: normalized = (value - min) / (max - min)
        
        Args:
            player: Player dictionary with stats
            norm_stats: Normalization statistics from compute_normalization_stats
            
        Returns:
            numpy array of shape (12,) with normalized features
        """
        # Per-game average features
        features = ['goals_per_game', 'assists_per_game', 'avg_points_per_game', 
                    'clean_sheets_per_game', 'avg_minutes', 'avg_bonus', 
                    'form', 'ict_index', 'influence', 'creativity', 'threat', 'games_played']
        
        # Fallback features if per-game not available
        fallback_features = ['goals', 'assists', 'total_points', 'clean_sheets', 
                             'minutes', 'bonus', 'form', 'ict_index',
                             'influence', 'creativity', 'threat', 'games_played']
        
        embedding = []
        for i, feat in enumerate(features):
            # Try per-game feature first, fallback to totals
            raw_value = player.get(feat, 0) or player.get(fallback_features[i], 0) or 0
            stats = norm_stats[feat]
            # Min-max normalization to [0, 1]
            normalized = (raw_value - stats['min']) / stats['range']
            embedding.append(normalized)
        
        return np.array(embedding, dtype=np.float32)
    
    def create_all_embeddings(self, players: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Create embeddings for all players in batch.
        
        Returns:
            Dictionary mapping player_name to embedding vector
        """
        print("🔢 Computing normalization statistics...")
        norm_stats = self.compute_normalization_stats(players)
        
        print("📐 Creating embeddings for all players...")
        embeddings = {}
        for player in players:
            name = player['player_name']
            embeddings[name] = self.create_numeric_embedding(player, norm_stats)
        
        print(f"✅ Created {len(embeddings)} embeddings of dimension {len(list(embeddings.values())[0])}")
        return embeddings, norm_stats
    
    # ═════════════════════════════════════════════════════════════════════════
    # FIXTURE EMBEDDINGS
    # ═════════════════════════════════════════════════════════════════════════
    
    def fetch_all_fixtures(self, season: str = None) -> List[Dict]:
        """Fetch all fixtures with their stats."""
        season_filter = ""
        params = {}
        
        if season:
            season_filter = "MATCH (gw:Gameweek {season: $season})<-[:HAS_FIXTURE]-(f:Fixture)"
            params['season'] = season
        else:
            season_filter = "MATCH (f:Fixture)<-[:HAS_FIXTURE]-(gw:Gameweek)"
        
        query = f"""
            {season_filter}
            OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
            OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
            WITH f, gw, home.name AS home_team, away.name AS away_team
            OPTIONAL MATCH (p)-[played:PLAYED_IN]->(f)
            WITH f, gw, home_team, away_team,
                 COUNT(DISTINCT p) AS players_in_fixture,
                 AVG(played.total_points) AS avg_points,
                 MAX(played.total_points) AS max_points,
                 MIN(played.total_points) AS min_points
            RETURN DISTINCT 
                   f.fixture_number AS fixture_number,
                   gw.GW_number AS gameweek,
                   gw.season AS season,
                   home_team,
                   away_team,
                   players_in_fixture,
                   avg_points,
                   max_points,
                   min_points
        """
        
        with self.driver.session() as session:
            result = session.run(query, params)
            return [dict(record) for record in result]
    
    def create_fixture_embedding(self, fixture: Dict) -> np.ndarray:
        """Create embedding for a fixture based on team strength and match context."""
        # 8-dimensional embedding for fixtures:
        # [gameweek_norm, home_strength, away_strength, avg_points, max_points, min_points, num_players, fixture_quality]
        
        embedding = np.zeros(8, dtype=np.float32)
        
        # Normalize gameweek (assuming max 38 gameweeks)
        gameweek = fixture.get('gameweek', 1)
        embedding[0] = min(gameweek / 38.0, 1.0)
        
        # Team strength (proxy: use team name hash for consistency)
        home_team = fixture.get('home_team', '')
        away_team = fixture.get('away_team', '')
        
        # Hash teams to get pseudo-strength (0-1)
        home_strength = (hash(home_team) % 100) / 100.0 if home_team else 0.5
        away_strength = (hash(away_team) % 100) / 100.0 if away_team else 0.5
        
        embedding[1] = home_strength
        embedding[2] = away_strength
        
        # Points statistics (normalized to 0-1 range, assuming max ~100 points)
        embedding[3] = min((fixture.get('avg_points', 0) or 0) / 100.0, 1.0)
        embedding[4] = min((fixture.get('max_points', 0) or 0) / 150.0, 1.0)
        embedding[5] = min((fixture.get('min_points', 0) or 0) / 50.0, 1.0)
        
        # Number of players in fixture (normalized, assuming max ~1000)
        num_players = fixture.get('players_in_fixture', 0) or 0
        embedding[6] = min(num_players / 1000.0, 1.0)
        
        # Fixture "quality" - higher when there's good scoring potential
        quality = ((embedding[3] + embedding[4]) / 2.0) * (1.0 + embedding[6] / 2.0)
        embedding[7] = min(quality, 1.0)
        
        return embedding
    
    def store_fixture_embeddings_in_neo4j(self, season: str = None) -> Dict:
        """Store fixture embeddings in Neo4j."""
        fixtures = self.fetch_all_fixtures(season)
        
        if not fixtures:
            return {'error': 'No fixtures found', 'count': 0}
        
        # Create embeddings
        embeddings = {}
        for fixture in fixtures:
            fixture_key = (fixture['fixture_number'], fixture['season'])
            embeddings[fixture_key] = self.create_fixture_embedding(fixture)
        
        # Store in Neo4j
        update_query = """
            MATCH (f:Fixture {fixture_number: $fixture_number})
            WHERE (f)<-[:HAS_FIXTURE]-(gw:Gameweek {season: $season})
            SET f.embedding = $embedding,
                f.embedding_type = 'fixture',
                f.embedding_dim = 8
        """
        
        stored = 0
        with self.driver.session() as session:
            for fixture in fixtures:
                fixture_key = (fixture['fixture_number'], fixture['season'])
                if fixture_key in embeddings:
                    try:
                        session.run(update_query, {
                            'fixture_number': fixture['fixture_number'],
                            'season': fixture['season'],
                            'embedding': embeddings[fixture_key].tolist()
                        })
                        stored += 1
                    except Exception as e:
                        print(f"⚠️ Error storing fixture {fixture['fixture_number']}: {e}")
        
        return {
            'fixtures_processed': len(fixtures),
            'embeddings_stored': stored,
            'embedding_dimensions': 8
        }
    
    # ═════════════════════════════════════════════════════════════════════════
    # GAMEWEEK EMBEDDINGS
    # ═════════════════════════════════════════════════════════════════════════
    
    def fetch_all_gameweeks(self, season: str = None) -> List[Dict]:
        """Fetch all gameweeks with their stats."""
        season_filter = ""
        params = {}
        
        if season:
            season_filter = "WHERE gw.season = $season"
            params['season'] = season
        
        query = f"""
            MATCH (gw:Gameweek)
            {season_filter}
            OPTIONAL MATCH (gw)<-[:HAS_FIXTURE]-(f:Fixture)
            OPTIONAL MATCH (p)-[played:PLAYED_IN]->(f)
            WITH gw,
                 COUNT(DISTINCT f) AS num_fixtures,
                 COUNT(DISTINCT p) AS num_players,
                 AVG(played.total_points) AS avg_points,
                 MAX(played.total_points) AS max_points,
                 MIN(played.total_points) AS min_points,
                 STDEV(played.total_points) AS points_variance
            RETURN gw.GW_number AS gameweek,
                   gw.season AS season,
                   num_fixtures,
                   num_players,
                   avg_points,
                   max_points,
                   min_points,
                   COALESCE(points_variance, 0) AS points_variance
        """
        
        with self.driver.session() as session:
            result = session.run(query, params)
            return [dict(record) for record in result]
    
    def create_gameweek_embedding(self, gameweek: Dict) -> np.ndarray:
        """Create embedding for a gameweek based on match activity and scoring stats."""
        # 8-dimensional embedding for gameweeks:
        # [gameweek_norm, fixture_density, player_coverage, avg_points, max_points, min_points, variance, excitement]
        
        embedding = np.zeros(8, dtype=np.float32)
        
        # Gameweek number (normalized to 0-1, assuming max 38)
        gw_num = gameweek.get('gameweek', 1)
        embedding[0] = min(gw_num / 38.0, 1.0)
        
        # Fixture density (max 10 fixtures per gameweek)
        num_fixtures = gameweek.get('num_fixtures', 0) or 0
        embedding[1] = min(num_fixtures / 10.0, 1.0)
        
        # Player coverage (normalized, assuming ~1000 players across all fixtures)
        num_players = gameweek.get('num_players', 0) or 0
        embedding[2] = min(num_players / 1000.0, 1.0)
        
        # Points statistics (normalized)
        embedding[3] = min((gameweek.get('avg_points', 0) or 0) / 100.0, 1.0)
        embedding[4] = min((gameweek.get('max_points', 0) or 0) / 150.0, 1.0)
        embedding[5] = min((gameweek.get('min_points', 0) or 0) / 50.0, 1.0)
        
        # Points variance (normalized, high variance = more variability)
        variance = (gameweek.get('points_variance', 0) or 0)
        embedding[6] = min(variance / 100.0, 1.0)
        
        # "Excitement" score - combination of activity and variance
        excitement = (embedding[1] * 0.3 + embedding[2] * 0.3 + embedding[6] * 0.4)
        embedding[7] = excitement
        
        return embedding
    
    def store_gameweek_embeddings_in_neo4j(self, season: str = None) -> Dict:
        """Store gameweek embeddings in Neo4j."""
        gameweeks = self.fetch_all_gameweeks(season)
        
        if not gameweeks:
            return {'error': 'No gameweeks found', 'count': 0}
        
        # Create embeddings
        embeddings = {}
        for gw in gameweeks:
            gw_key = (gw['gameweek'], gw['season'])
            embeddings[gw_key] = self.create_gameweek_embedding(gw)
        
        # Store in Neo4j
        update_query = """
            MATCH (gw:Gameweek {GW_number: $gameweek, season: $season})
            SET gw.embedding = $embedding,
                gw.embedding_type = 'gameweek',
                gw.embedding_dim = 8
        """
        
        stored = 0
        with self.driver.session() as session:
            for gw in gameweeks:
                gw_key = (gw['gameweek'], gw['season'])
                if gw_key in embeddings:
                    try:
                        session.run(update_query, {
                            'gameweek': gw['gameweek'],
                            'season': gw['season'],
                            'embedding': embeddings[gw_key].tolist()
                        })
                        stored += 1
                    except Exception as e:
                        print(f"⚠️ Error storing gameweek {gw['gameweek']}: {e}")
        
        return {
            'gameweeks_processed': len(gameweeks),
            'embeddings_stored': stored,
            'embedding_dimensions': 8
        }
    
    def create_vector_index(self, embedding_type: str = 'numeric'):
        """
        Create a vector index in Neo4j for fast similarity search.
        
        Args:
            embedding_type: 'numeric' (12 dims), 'minilm' (384 dims), or 'mpnet' (768 dims)
        """
        dims_map = {'numeric': 12, 'minilm': 384, 'mpnet': 768}
        property_map = {'numeric': 'embedding', 'minilm': 'embedding_minilm', 'mpnet': 'embedding_mpnet'}
        
        dims = dims_map.get(embedding_type, 12)
        prop = property_map.get(embedding_type, 'embedding')
        index_name = f"player_{embedding_type}_embeddings"
        
        print(f"📁 Creating vector index '{index_name}' in Neo4j...")
        
        drop_query = f"DROP INDEX {index_name} IF EXISTS"
        create_query = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (p:Player) ON (p.{prop})
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dims},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """
        
        with self.driver.session() as session:
            try:
                session.run(drop_query)
            except:
                pass
            session.run(create_query)
        
        print(f"✅ Vector index created ({dims} dimensions, cosine similarity)")
    
    def create_text_representation(self, player: Dict) -> str:
        """Create text description from player stats for text embedding."""
        return (
            f"Football player {player.get('player_name', 'Unknown')} "
            f"plays as {player.get('position', 'Unknown')} position. "
            f"Season statistics: {player.get('total_points', 0)} total FPL points, "
            f"{player.get('goals', 0)} goals scored, "
            f"{player.get('assists', 0)} assists provided, "
            f"{player.get('clean_sheets', 0)} clean sheets. "
            f"Playing time: {player.get('minutes', 0)} minutes in {player.get('games_played', 0)} games. "
            f"Bonus points earned: {player.get('bonus', 0)}. "
            f"Performance metrics: form {player.get('form', 0):.2f}, "
            f"ICT index {player.get('ict_index', 0):.2f}."
        )
    
    def store_text_embeddings_in_neo4j(self, model_name: str = 'model_1', batch_size: int = 32) -> Dict:
        """
        Store TEXT-BASED embeddings using sentence transformers.
        
        This satisfies the requirement to "experiment with at least TWO different
        embedding models for comparison" - storing both MiniLM and MPNet embeddings.
        
        Args:
            model_name: 'model_1' (MiniLM, 384 dims) or 'model_2' (MPNet, 768 dims)
            batch_size: Batch size for encoding
            
        Returns:
            Dictionary with statistics
        """
        model_info = {
            'model_1': {'name': 'MiniLM', 'dims': 384, 'property': 'embedding_minilm'},
            'model_2': {'name': 'MPNet', 'dims': 768, 'property': 'embedding_mpnet'}
        }
        
        if model_name not in model_info:
            raise ValueError(f"Unknown model: {model_name}")
        
        info = model_info[model_name]
        
        print("=" * 70)
        print(f"STORING TEXT EMBEDDINGS ({info['name']}) IN NEO4J")
        print("=" * 70)
        
        # Fetch players
        print("📊 Fetching players...")
        players = self.fetch_all_players_stats()
        
        if not players:
            return {'error': 'No players found'}
        
        # Create text representations
        print("📝 Creating text representations...")
        texts = [self.create_text_representation(p) for p in players]
        
        # Generate embeddings in batches
        print(f"🔢 Generating embeddings with {info['name']} (batch_size={batch_size})...")
        model = self.embedding_models[model_name]
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(batch_emb)
            if (i + batch_size) % 200 == 0:
                print(f"   Encoded {min(i + batch_size, len(texts))}/{len(texts)} texts...")
        
        embeddings = np.vstack(all_embeddings)
        print(f"✅ Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        
        # Store in Neo4j
        print(f"💾 Storing embeddings as '{info['property']}' in Neo4j...")
        
        update_query = f"""
            MATCH (p:Player {{player_name: $player_name}})
            SET p.{info['property']} = $embedding,
                p.{info['property']}_model = $model_name
        """
        
        stored = 0
        with self.driver.session() as session:
            for i, player in enumerate(players):
                if i < len(embeddings):
                    session.run(update_query, {
                        'player_name': player['player_name'],
                        'embedding': embeddings[i].tolist(),
                        'model_name': info['name']
                    })
                    stored += 1
                    
                    if stored % 200 == 0:
                        print(f"   Stored {stored}/{len(players)} embeddings...")
        
        print(f"✅ Stored {stored} embeddings")
        
        # Create index
        idx_type = 'minilm' if model_name == 'model_1' else 'mpnet'
        self.create_vector_index(idx_type)
        
        print("=" * 70)
        
        return {
            'model': info['name'],
            'players_processed': len(players),
            'embeddings_stored': stored,
            'dimensions': info['dims']
        }
    
    def store_all_embeddings(self) -> Dict:
        """
        Store ALL THREE types of embeddings for complete comparison:
        1. Numeric (12 dims)
        2. MiniLM text (384 dims)
        3. MPNet text (768 dims)
        
        This fully satisfies the project requirement to experiment with
        at least TWO different embedding models.
        """
        print("=" * 80)
        print("STORING ALL EMBEDDING TYPES FOR COMPARISON")
        print("=" * 80)
        
        results = {}
        
        # 1. Numeric embeddings
        print("\n[1/3] NUMERIC EMBEDDINGS")
        results['numeric'] = self.store_embeddings_in_neo4j()
        
        # 2. MiniLM text embeddings
        print("\n[2/3] TEXT EMBEDDINGS (MiniLM)")
        results['minilm'] = self.store_text_embeddings_in_neo4j('model_1')
        
        # 3. MPNet text embeddings
        print("\n[3/3] TEXT EMBEDDINGS (MPNet)")
        results['mpnet'] = self.store_text_embeddings_in_neo4j('model_2')
        
        print("\n" + "=" * 80)
        print("ALL EMBEDDINGS STORED SUCCESSFULLY")
        print("=" * 80)
        
        print("\nSummary:")
        print(f"  • Numeric: {results['numeric'].get('embeddings_stored', 0)} players, 12 dims")
        print(f"  • MiniLM:  {results['minilm'].get('embeddings_stored', 0)} players, 384 dims")
        print(f"  • MPNet:   {results['mpnet'].get('embeddings_stored', 0)} players, 768 dims")
        
        return results
    
    def store_embeddings_in_neo4j(self) -> Dict:
        """
        Main function to create and store embeddings for ALL players.
        
        Process:
        1. Fetch all players with latest position (no duplicates)
        2. Compute normalization statistics
        3. Create numeric embeddings for each player
        4. Store embeddings in Neo4j Player nodes
        5. Create vector index for fast search
        
        Returns:
            Dictionary with statistics about the operation
        """
        print("=" * 70)
        print("STORING NUMERIC NODE EMBEDDINGS IN NEO4J")
        print("=" * 70)
        
        # Step 1: Fetch all players
        players = self.fetch_all_players_stats()
        
        if not players:
            return {'error': 'No players found', 'count': 0}
        
        # Step 2 & 3: Create embeddings
        embeddings, norm_stats = self.create_all_embeddings(players)
        
        # Step 4: Store in Neo4j
        print("💾 Storing embeddings in Neo4j...")
        
        update_query = """
            MATCH (p:Player {player_name: $player_name})
            SET p.embedding = $embedding,
                p.embedding_type = 'numeric',
                p.embedding_dim = 12
        """
        
        stored = 0
        with self.driver.session() as session:
            for player in players:
                name = player['player_name']
                if name in embeddings:
                    session.run(update_query, {
                        'player_name': name,
                        'embedding': embeddings[name].tolist()
                    })
                    stored += 1
                    
                    if stored % 200 == 0:
                        print(f"   Stored {stored}/{len(players)} embeddings...")
        
        print(f"✅ Stored {stored} embeddings in Neo4j")
        
        # Step 5: Create vector index
        self.create_vector_index()
        
        # Store normalization stats for later use
        self._norm_stats = norm_stats
        
        print("=" * 70)
        print("✅ EMBEDDING STORAGE COMPLETE")
        print("=" * 70)
        
        return {
            'players_processed': len(players),
            'embeddings_stored': stored,
            'embedding_dimensions': 12,
            'features': list(norm_stats.keys()),
            'normalization_stats': {k: {'min': v['min'], 'max': v['max']} 
                                    for k, v in norm_stats.items()}
        }
    
    def create_query_embedding(self, criteria: Dict) -> np.ndarray:
        """
        Create an embedding vector from search criteria.
        
        Example criteria:
        {'goals_per_game': 'high', 'assists_per_game': 'high', 'position': 'FWD'}
        
        Uses per-game average features to match the stored embeddings.
        Converts to normalized vector for similarity search.
        """
        # Default: mid-range values (0.5)
        embedding = np.full(12, 0.5, dtype=np.float32)
        
        # Feature indices (per-game averages)
        feature_map = {
            # Per-game features (primary)
            'goals_per_game': 0, 'assists_per_game': 1, 'avg_points_per_game': 2, 
            'clean_sheets_per_game': 3, 'avg_minutes': 4, 'avg_bonus': 5,
            'form': 6, 'ict_index': 7, 'influence': 8, 'creativity': 9, 
            'threat': 10, 'games_played': 11,
            # Legacy aliases (for backward compatibility)
            'goals': 0, 'assists': 1, 'total_points': 2, 'points': 2,
            'clean_sheets': 3, 'minutes': 4, 'bonus': 5
        }
        
        for feature, value in criteria.items():
            if feature in feature_map:
                idx = feature_map[feature]
                if value == 'high':
                    embedding[idx] = 1.0
                elif value == 'low':
                    embedding[idx] = 0.0
                elif isinstance(value, (int, float)):
                    embedding[idx] = min(max(value, 0), 1)
        
        return embedding
    
    def semantic_search(self, query_embedding: np.ndarray, top_k: int = 10, 
                    position: str = None, embedding_type: str = 'numeric') -> List[Dict]:
        """
        Find similar players using cosine similarity with any embedding type.
        
        Args:
            query_embedding: Query vector (dims depend on embedding_type)
            top_k: Number of results to return
            position: Optional position filter (GK, DEF, MID, FWD)
            embedding_type: 'numeric' (12d), 'minilm' (384d), or 'mpnet' (768d)
        """
        # Map embedding type to Neo4j property
        property_map = {
            'numeric': 'embedding',
            'minilm': 'embedding_minilm',
            'mpnet': 'embedding_mpnet'
        }
        
        if embedding_type not in property_map:
            raise ValueError(f"Unsupported embedding_type: {embedding_type}")
        
        emb_prop = property_map[embedding_type]
        
        # Build query
        if position:
            query = f"""
                MATCH (p:Player)
                WHERE p.{emb_prop} IS NOT NULL
                OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position {{name: $position}})
                WITH p.player_element AS player_id,
                    COLLECT(DISTINCT p.player_name)[0] AS player,
                    COLLECT(DISTINCT pos.name)[0] AS position,
                    p.{emb_prop} AS embedding
                WHERE player IS NOT NULL AND position = $position
                RETURN player, position, embedding
            """
            params = {'position': position}
        else:
            query = f"""
                MATCH (p:Player)
                WHERE p.{emb_prop} IS NOT NULL
                OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                WITH p.player_element AS player_id,
                    COLLECT(DISTINCT p.player_name)[0] AS player,
                    COLLECT(DISTINCT pos.name)[0] AS position,
                    p.{emb_prop} AS embedding
                WHERE player IS NOT NULL
                RETURN player, position, embedding
            """
            params = {}
        
        with self.driver.session() as session:
            result = session.run(query, params)
            players = [dict(record) for record in result]
        
        if not players:
            return []
        
        # Compute cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        similarities = []
        seen_players = set()
        
        for player in players:
            player_name = player['player']
            if player_name in seen_players:
                continue
            seen_players.add(player_name)
            
            player_emb = np.array(player['embedding'], dtype=np.float32)
            player_norm = player_emb / (np.linalg.norm(player_emb) + 1e-10)
            similarity = float(np.dot(query_norm, player_norm))
            
            similarities.append({
                'player': player_name,
                'position': player['position'],
                'similarity_score': round(similarity, 4),
                'embedding_type': embedding_type
            })
        
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:top_k]
    
    def search_with_text_embedding(self, query: str, model_name: str = 'model_1',
                                   top_k: int = 10, position: str = None) -> Dict:
        """
        Search using text embeddings (MiniLM or MPNet).
        
        Args:
            query: Natural language query
            model_name: 'model_1' (MiniLM) or 'model_2' (MPNet)
            top_k: Number of results
            position: Optional position filter
        """
        model = self.embedding_models[model_name]
        query_emb = model.encode([query], convert_to_numpy=True)[0]
        
        emb_type = 'minilm' if model_name == 'model_1' else 'mpnet'
        results = self.semantic_search(query_emb, top_k, position, emb_type)
        
        return {
            'method': f'text_{emb_type}',
            'model': model_name,
            'query': query,
            'position_filter': position,
            'results': results
        }
    
    def compare_all_embedding_models(self, position: str = 'FWD', top_k: int = 10) -> Dict:
        """
        Compare search results across all THREE embedding approaches.
        
        This is the main comparison function for the project requirement:
        "experiment with at least TWO different embedding models for comparison"
        
        Args:
            position: Position to filter (FWD, MID, DEF, GK)
            top_k: Number of results per approach
        """
        print("=" * 80)
        print(f"COMPARING ALL EMBEDDING MODELS (Position: {position})")
        print("=" * 80)
        
        results = {}
        
        # 1. Numeric search (criteria-based)
        print("\n📊 [1/3] Numeric Embeddings (12 dims)")
        criteria = {'goals': 'high', 'total_points': 'high', 'assists': 'high'}
        query_numeric = self.create_query_embedding(criteria)
        results['numeric'] = {
            'dims': 12,
            'criteria': criteria,
            'results': self.semantic_search(query_numeric, top_k, position, 'numeric')
        }
        
        # 2. MiniLM search (semantic)
        print("📊 [2/3] MiniLM Text Embeddings (384 dims)")
        query_text = f"Best {position} player with high goals and assists"
        results['minilm'] = self.search_with_text_embedding(query_text, 'model_1', top_k, position)
        results['minilm']['dims'] = 384
        
        # 3. MPNet search (semantic)
        print("📊 [3/3] MPNet Text Embeddings (768 dims)")
        results['mpnet'] = self.search_with_text_embedding(query_text, 'model_2', top_k, position)
        results['mpnet']['dims'] = 768
        
        # Display results
        print("\n" + "─" * 80)
        print("SEARCH RESULTS COMPARISON")
        print("─" * 80)
        
        print(f"\n{'Rank':<6} {'Numeric (12d)':<25} {'MiniLM (384d)':<25} {'MPNet (768d)':<25}")
        print("─" * 85)
        
        for i in range(min(top_k, 10)):
            num_res = results['numeric']['results'][i] if i < len(results['numeric']['results']) else {}
            mini_res = results['minilm']['results'][i] if i < len(results['minilm']['results']) else {}
            mpn_res = results['mpnet']['results'][i] if i < len(results['mpnet']['results']) else {}
            
            num_name = num_res.get('player', '-')[:22] if num_res else '-'
            mini_name = mini_res.get('player', '-')[:22] if mini_res else '-'
            mpn_name = mpn_res.get('player', '-')[:22] if mpn_res else '-'
            
            print(f"{i+1:<6} {num_name:<25} {mini_name:<25} {mpn_name:<25}")
        
        # Calculate agreement
        print("\n" + "─" * 80)
        print("MODEL AGREEMENT ANALYSIS")
        print("─" * 80)
        
        top10_numeric = set(r['player'] for r in results['numeric']['results'][:10])
        top10_minilm = set(r['player'] for r in results['minilm']['results'][:10])
        top10_mpnet = set(r['player'] for r in results['mpnet']['results'][:10])
        
        print(f"\n   Numeric ∩ MiniLM: {len(top10_numeric & top10_minilm)}/10 players agree")
        print(f"   Numeric ∩ MPNet:  {len(top10_numeric & top10_mpnet)}/10 players agree")
        print(f"   MiniLM ∩ MPNet:   {len(top10_minilm & top10_mpnet)}/10 players agree")
        print(f"   All three agree:  {len(top10_numeric & top10_minilm & top10_mpnet)}/10 players")
        
        print("\n" + "=" * 80)
        
        return results
    
    def embedding_retrieve(self, criteria: Dict = None, position: str = None, 
                        top_k: int = 10, embedding_type: str = 'numeric') -> Dict[str, Any]:
        """
        Retrieve similar players based on criteria using specified embedding type.
        
        Args:
            criteria: Dictionary of desired attributes
            position: Filter by position (GK, DEF, MID, FWD)
            top_k: Number of results
            embedding_type: 'numeric', 'minilm', or 'mpnet'
        """
        if criteria is None:
            criteria = {'total_points': 'high'}
        
        if embedding_type == 'numeric':
            # For numeric embeddings, create query vector from criteria
            query_embedding = self.create_query_embedding(criteria)
        else:
            # For text embeddings, convert criteria to text query
            query_text = self._criteria_to_text_query(criteria, position)
            model_name = 'model_1' if embedding_type == 'minilm' else 'model_2'
            model = self.embedding_models[model_name]
            query_embedding = model.encode([query_text], convert_to_numpy=True)[0]
        
        results = self.semantic_search(query_embedding, top_k, position, embedding_type)
        
        return {
            'method': f'{embedding_type}_embedding',
            'criteria': criteria,
            'position_filter': position,
            'embedding_type': embedding_type,
            'embedding_dimensions': query_embedding.shape[0],
            'results': results
        }
    
    def find_similar_players(self, player_name: str, top_k: int = 10, 
                             same_position: bool = True) -> List[Dict]:
        """
        Find players similar to a given player.
        
        How it works:
        1. Get the target player's embedding from Neo4j
        2. Search for players with similar embeddings
        3. Optionally filter to same position
        
        Args:
            player_name: Name of the player to find similar to
            top_k: Number of similar players to return
            same_position: If True, only return players of same position
        """
        # Get target player's embedding and position (handle multiple matches)
        query = """
            MATCH (p:Player {player_name: $player_name})
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
            WHERE p.embedding IS NOT NULL
            WITH p.embedding AS embedding, 
                 COLLECT(DISTINCT pos.name)[0] AS position
            RETURN embedding, position
            LIMIT 1
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'player_name': player_name})
            record = result.single()
            
            if not record or record['embedding'] is None:
                return [{'error': f'Player "{player_name}" not found or has no embedding'}]
            
            target_embedding = np.array(record['embedding'], dtype=np.float32)
            target_position = record['position'] if same_position else None
        
        # Find similar players (excluding the target player)
        results = self.semantic_search(target_embedding, top_k + 1, target_position)
        
        # Remove the target player from results
        results = [r for r in results if r['player'] != player_name][:top_k]
        
        return results

    # =========================================================================
    # COMBINED RETRIEVAL
    # =========================================================================
    
    def retrieve(self, user_input: str, method: str = 'both', 
                embedding_type: str = 'numeric') -> Dict[str, Any]:
        """
        Main retrieval method combining baseline and embeddings.
        
        Args:
            user_input: Raw user query
            method: 'baseline', 'embedding', or 'both'
            embedding_type: 'numeric', 'minilm', or 'mpnet' - which embedding to use
        """
        results = {'user_input': user_input}
        
        # Always run baseline for intent/entity extraction
        baseline_result = self.baseline_retrieve(user_input)
        results['baseline'] = baseline_result
        
        preprocessed = self.preprocessor.preprocess(user_input, include_embedding=False)
        query_used = baseline_result.get('query_used', 'top_scorers')
        entities = preprocessed.get('entities', {})
        
        if method in ['embedding', 'both']:
            # Determine if this query type can benefit from embeddings
            # ALL player-centric queries should use embeddings
            player_centric_queries = {
                'top_players_by_position',      # ✓ Player similarity by position
                'top_scorers',                   # ✓ Player similarity by goals
                'top_assisters',                 # ✓ Player similarity by assists
                'clean_sheet_leaders',           # ✓ Defender/GK similarity
                'players_by_form',               # ✓ Player similarity by form
                'bonus_leaders',                 # ✓ Player similarity by bonus points
                'most_cards',                    # ✓ Player similarity by cards
                'player_season_summary',         # ✓ Find similar players to reference player
                'compare_players',               # ✓ Find similar players to both
                'players_by_team',               # ✓ Find similar players on other teams
                'player_gameweek_performance',   # ✓ Find similar players with similar performances
            }
            
            if query_used in player_centric_queries:
                # Extract position from query if available
                positions = entities.get('positions', [])
                position = positions[0] if positions else None
                
                # Get search criteria from preprocessed query
                search_criteria = preprocessed.get('search_criteria', {})
                
                if embedding_type == 'numeric':
                    # Use numeric embeddings
                    results['embedding'] = self.embedding_retrieve(
                        criteria=search_criteria, 
                        position=position, 
                        top_k=10
                    )
                elif embedding_type == 'minilm':
                    # Use MiniLM text embeddings
                    # Convert search criteria to natural language query
                    query_text = self._criteria_to_text_query(search_criteria, position)
                    results['embedding'] = self.search_with_text_embedding(
                        query=query_text,
                        model_name='model_1',
                        top_k=10,
                        position=position
                    )
                elif embedding_type == 'mpnet':
                    # Use MPNet text embeddings
                    query_text = self._criteria_to_text_query(search_criteria, position)
                    results['embedding'] = self.search_with_text_embedding(
                        query=query_text,
                        model_name='model_2',
                        top_k=10,
                        position=position
                    )
                
                results['embedding']['embedding_type'] = embedding_type
            else:
                results['embedding'] = {
                    'method': 'skipped',
                    'reason': f'Query "{query_used}" may not benefit from player embeddings',
                    'embedding_type': embedding_type
                }
        
        if method == 'both' and 'embedding' in results:
            # Merge results if embeddings were actually used
            embedding_results = results.get('embedding', {}).get('results', [])
            baseline_results = results.get('baseline', {}).get('results', [])
            
            if embedding_results:  # Only merge if embeddings returned results
                results['combined'] = self._merge_results(baseline_results, embedding_results)
            else:
                # Use only baseline results if embeddings skipped or empty
                results['combined'] = baseline_results
        
        return results

    def _criteria_to_text_query(self, criteria: Dict, position: str = None) -> str:
        """
        Convert search criteria to natural language query for text embeddings.
        
        Example: {'goals': 'high', 'assists': 'high'} -> 
                "Players with high goals and assists"
        """
        parts = []
        
        # Map criteria values to natural language
        value_map = {
            'high': 'high',
            'low': 'low',
            'good': 'good',
            'bad': 'poor'
        }
        
        # Map criteria keys to natural language
        criteria_map = {
            'goals': 'goals',
            'assists': 'assists', 
            'total_points': 'FPL points',
            'clean_sheets': 'clean sheets',
            'bonus': 'bonus points',
            'minutes': 'minutes played',
            'form': 'form',
            'ict_index': 'ICT index',
            'influence': 'influence',
            'creativity': 'creativity',
            'threat': 'threat',
            'cards': 'cards',
            'saves': 'saves',
            'value': 'value',
            'ownership': 'ownership'
        }
        
        for key, value in criteria.items():
            if key in criteria_map and value in value_map:
                parts.append(f"{value_map[value]} {criteria_map[key]}")
        
        # Build the query
        if position:
            position_map = {
                'FWD': 'forwards',
                'MID': 'midfielders', 
                'DEF': 'defenders',
                'GK': 'goalkeepers'
            }
            position_text = position_map.get(position, 'players')
            if parts:
                query = f"{position_text} with " + " and ".join(parts)
            else:
                query = f"best {position_text}"
        else:
            if parts:
                query = "Players with " + " and ".join(parts)
            else:
                query = "best players"
        
        return query    
    def _merge_results(self, baseline_results: List, embedding_results: List) -> List:
        """Merge results from both methods, removing duplicates."""
        combined = baseline_results.copy() if baseline_results else []
        seen_players = {r.get('player') for r in combined if r.get('player')}
        
        for result in (embedding_results or []):
            if result.get('player') not in seen_players:
                combined.append(result)
                seen_players.add(result.get('player'))
        
        return combined
    
    def compare_embedding_approaches(self, player_name: str = "Erling Haaland") -> Dict:
        """
        Compare numeric embeddings vs text-based embeddings.
        
        This demonstrates the two approaches:
        1. Numeric: Direct feature vectors [goals, assists, ...]
        2. Text-based: Convert stats to text, then embed with transformer
        """
        print("=" * 70)
        print("COMPARING EMBEDDING APPROACHES")
        print("=" * 70)
        
        # Get player stats
        query = """
            MATCH (p:Player {player_name: $name})-[:PLAYS_AS]->(pos:Position)
            OPTIONAL MATCH (p)-[played:PLAYED_IN]->(f:Fixture)
            RETURN p.player_name AS player_name,
                   pos.name AS position,
                   COALESCE(SUM(played.total_points), 0) AS total_points,
                   COALESCE(SUM(played.goals_scored), 0) AS goals,
                   COALESCE(SUM(played.assists), 0) AS assists,
                   p.embedding AS numeric_embedding
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'name': player_name})
            record = result.single()
        
        if not record:
            return {'error': f'Player {player_name} not found'}
        
        player_data = dict(record)
        
        # Approach 1: Numeric embedding (already stored)
        numeric_emb = player_data.get('numeric_embedding')
        
        # Approach 2: Text-based embedding
        text_repr = (
            f"Player {player_data['player_name']} is a {player_data['position']} "
            f"with {player_data['goals']} goals, {player_data['assists']} assists, "
            f"and {player_data['total_points']} total points."
        )
        
        text_embeddings = {}
        for model_name, model in self.embedding_models.items():
            text_emb = model.encode([text_repr], convert_to_numpy=True)[0]
            text_embeddings[model_name] = {
                'dimensions': len(text_emb),
                'sample': text_emb[:5].tolist()
            }
        
        return {
            'player': player_name,
            'numeric_embedding': {
                'approach': 'Direct numeric features',
                'dimensions': 12,
                'features': ['goals', 'assists', 'points', 'clean_sheets', 
                            'minutes', 'bonus', 'form', 'ict', 'influence', 
                            'creativity', 'threat', 'games'],
                'values': numeric_emb[:5] if numeric_emb else None,
                'pros': ['Fast computation', 'Preserves exact relationships', 'Small storage'],
                'cons': ['Cannot handle semantic queries', 'Fixed features only']
            },
            'text_embeddings': {
                'approach': 'Text-based (sentence transformers)',
                'models': text_embeddings,
                'pros': ['Handles semantic queries', 'Flexible input'],
                'cons': ['Slower', 'Larger storage', 'Indirect representation']
            }
        }


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_numeric_embeddings():
    """Test the numeric embedding functionality."""
    print("=" * 70)
    print("NUMERIC NODE EMBEDDINGS - TEST SUITE")
    print("=" * 70)
    
    # Initialize
    retriever = FPLGraphRetrieval(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD
    )
    
    # Test 1: Fetch all players
    print("\n" + "─" * 70)
    print("TEST 1: Fetch All Players (No Duplicates)")
    print("─" * 70)
    players = retriever.fetch_all_players_stats()
    print(f"✅ Total unique players: {len(players)}")
    
    # Check for duplicates
    names = [p['player_name'] for p in players]
    unique_names = set(names)
    print(f"✅ Unique names: {len(unique_names)} (duplicates: {len(names) - len(unique_names)})")
    
    # Position distribution
    positions = {}
    for p in players:
        pos = p['position']
        positions[pos] = positions.get(pos, 0) + 1
    print(f"✅ Position distribution: {positions}")
    
    # Test 2: Store embeddings
    print("\n" + "─" * 70)
    print("TEST 2: Store Embeddings in Neo4j")
    print("─" * 70)
    result = retriever.store_embeddings_in_neo4j()
    print(f"✅ Result: {result}")
    
    # Test 3: Search by criteria
    print("\n" + "─" * 70)
    print("TEST 3: Search by Criteria")
    print("─" * 70)
    
    test_searches = [
        {'criteria': {'goals': 'high', 'assists': 'high'}, 'position': 'FWD', 'desc': 'High-scoring forwards'},
        {'criteria': {'clean_sheets': 'high'}, 'position': 'GK', 'desc': 'Goalkeepers with clean sheets'},
        {'criteria': {'creativity': 'high', 'assists': 'high'}, 'position': 'MID', 'desc': 'Creative midfielders'},
        {'criteria': {'clean_sheets': 'high', 'bonus': 'high'}, 'position': 'DEF', 'desc': 'High-performing defenders'},
    ]
    
    for search in test_searches:
        print(f"\n🔍 {search['desc']} ({search['position']}):")
        results = retriever.embedding_retrieve(
            criteria=search['criteria'], 
            position=search['position'],
            top_k=5
        )
        for i, r in enumerate(results['results'][:5], 1):
            print(f"   {i}. {r['player']} ({r['position']}) - Similarity: {r['similarity_score']:.3f}")
    
    # Test 4: Find similar players
    print("\n" + "─" * 70)
    print("TEST 4: Find Similar Players")
    print("─" * 70)
    
    test_players = ["Erling Haaland", "Kevin De Bruyne", "Virgil van Dijk"]
    
    for player in test_players:
        print(f"\n🔍 Players similar to {player}:")
        similar = retriever.find_similar_players(player, top_k=5, same_position=True)
        if 'error' in similar[0] if similar else True:
            print(f"   ⚠️ {similar[0].get('error', 'Not found')}")
        else:
            for i, r in enumerate(similar[:5], 1):
                print(f"   {i}. {r['player']} ({r['position']}) - Similarity: {r['similarity_score']:.3f}")
    
    # Close connection
    retriever.close()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)


def test_plays_for_relationships():
    """Verify `PLAYS_FOR` relationships exist and are consistent with embeddings.

    - Runs a simple Cypher check: `MATCH p=()-[:PLAYS_FOR]->() RETURN p;`
    - Ensures that players which have embeddings in Neo4j also have a `PLAYS_FOR` relationship
    - Prints summary statistics and a small sample for manual inspection
    """
    print("\n" + "=" * 70)
    print("PLAYS_FOR RELATIONSHIP & EMBEDDING CONSISTENCY CHECK")
    print("=" * 70)

    retriever = FPLGraphRetrieval(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD
    )

    # Ensure embeddings exist (store numeric embeddings if missing)
    print("\n🔁 Ensuring numeric embeddings are stored (this may take a while)...")
    store_res = retriever.store_embeddings_in_neo4j()
    print(f"   Store result: {store_res}")

    # 1) Count PLAYS_FOR relationships and show a small sample
    with retriever.driver.session() as session:
        count_res = session.run("MATCH ()-[:PLAYS_FOR]->() RETURN count(*) AS rel_count")
        rel_count = count_res.single()['rel_count']

        sample_res = session.run(
            "MATCH (p:Player)-[r:PLAYS_FOR]->(t:Team) RETURN p.player_name AS player, t.name AS team LIMIT 10"
        )
        sample = [dict(rec) for rec in sample_res]

        # 2) Find how many players with embeddings are missing PLAYS_FOR
        missing_res = session.run(
            "MATCH (p:Player) WHERE p.embedding IS NOT NULL AND NOT (p)-[:PLAYS_FOR]->() RETURN count(p) AS missing"
        )
        missing = missing_res.single()['missing']

    print(f"\n✅ Total PLAYS_FOR relationships: {rel_count}")
    print(f"✅ Players with embeddings but missing PLAYS_FOR: {missing}")
    print("\nSample PLAYS_FOR rows:")
    for row in sample:
        print(f"   - {row['player']} -> {row['team']}")

    retriever.close()

    print("\n" + "=" * 70)
    print("PLAYS_FOR CHECK COMPLETE")
    print("=" * 70)


def test_all_embedding_approaches():
    """
    Compare ALL THREE embedding approaches:
    1. Numeric embeddings (direct feature vectors)
    2. Text embedding with MiniLM (model_1)
    3. Text embedding with MPNet (model_2)
    """
    import time
    
    print("=" * 80)
    print("COMPLETE EMBEDDING APPROACH COMPARISON")
    print("Numeric Features vs Text-Based (MiniLM) vs Text-Based (MPNet)")
    print("=" * 80)
    
    retriever = FPLGraphRetrieval(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD
    )
    
    print("\n📊 Fetching player data...")
    players = retriever.fetch_all_players_stats()[:200]
    print(f"   Loaded {len(players)} players for testing")
    
    # ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("APPROACH SPECIFICATIONS")
    print("─" * 80)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ NUMERIC FEATURES (12 dimensions)                                        │
    │ • Direct normalized stats: [goals, assists, points, clean_sheets, ...]  │
    │ • No text conversion needed                                             │
    │ • Fast and compact                                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ TEXT + MiniLM (384 dimensions)                                          │
    │ • Convert stats to text description                                     │
    │ • Embed with all-MiniLM-L6-v2 (22M params)                              │
    │ • Good speed/accuracy tradeoff                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ TEXT + MPNet (768 dimensions)                                           │
    │ • Convert stats to text description                                     │
    │ • Embed with all-mpnet-base-v2 (109M params)                            │
    │ • Best semantic understanding                                           │
    └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # ─────────────────────────────────────────────────────────────
    print("─" * 80)
    print("EMBEDDING GENERATION SPEED")
    print("─" * 80)
    
    embeddings = {}
    times = {}
    
    # 1. Numeric embeddings
    start = time.time()
    norm_stats = retriever.compute_normalization_stats(players)
    numeric_embs = [retriever.create_numeric_embedding(p, norm_stats) for p in players]
    times['numeric'] = time.time() - start
    embeddings['numeric'] = numeric_embs
    
    # 2. Create text representations
    texts = []
    for p in players:
        text = (f"Football player {p['player_name']} plays as {p['position']}. "
                f"Stats: {p['total_points']} points, {p['goals']} goals, "
                f"{p['assists']} assists, {p['clean_sheets']} clean sheets.")
        texts.append(text)
    
    # 3. MiniLM embeddings
    start = time.time()
    text_embs_minilm = retriever.embedding_models['model_1'].encode(texts, convert_to_numpy=True)
    times['minilm'] = time.time() - start
    embeddings['minilm'] = text_embs_minilm
    
    # 4. MPNet embeddings
    start = time.time()
    text_embs_mpnet = retriever.embedding_models['model_2'].encode(texts, convert_to_numpy=True)
    times['mpnet'] = time.time() - start
    embeddings['mpnet'] = text_embs_mpnet
    
    print(f"\n{'Approach':<20} {'Dims':<8} {'Time':<10} {'Speed':<15} {'Storage':<12}")
    print("─" * 70)
    specs = [
        ('Numeric', 12, times['numeric']),
        ('Text+MiniLM', 384, times['minilm']),
        ('Text+MPNet', 768, times['mpnet'])
    ]
    for name, dims, t in specs:
        storage = dims * 4  # bytes per float32
        print(f"{name:<20} {dims:<8} {t:<10.3f}s {len(players)/t:<15.0f}/sec {storage:<12} bytes")
    
    print(f"\n🏆 Numeric is {times['mpnet']/times['numeric']:.0f}x faster than MPNet")
    
    # ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("SEARCH RESULTS COMPARISON")
    print("─" * 80)
    
    # Filter to forwards
    fwd_idx = [i for i, p in enumerate(players) if p['position'] == 'FWD']
    print(f"\nSearching among {len(fwd_idx)} forwards for 'top goal scorer'")
    
    # Numeric search
    print("\n📊 NUMERIC APPROACH (criteria: goals=high, points=high):")
    query_num = retriever.create_query_embedding({'goals': 'high', 'total_points': 'high'})
    q_norm = query_num / (np.linalg.norm(query_num) + 1e-10)
    
    num_results = []
    for i in fwd_idx:
        e = embeddings['numeric'][i]
        e_norm = e / (np.linalg.norm(e) + 1e-10)
        sim = float(np.dot(q_norm, e_norm))
        num_results.append((players[i]['player_name'], sim, players[i]['goals']))
    num_results.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (name, sim, goals) in enumerate(num_results[:5], 1):
        print(f"   {rank}. {name:<25} Sim: {sim:.4f}  Goals: {goals}")
    
    # Text searches
    query_text = "Top scoring forward striker with many goals"
    
    for label, key, model_key in [('MiniLM', 'minilm', 'model_1'), ('MPNet', 'mpnet', 'model_2')]:
        print(f"\n📊 TEXT + {label} (query: \"{query_text}\"):")
        
        q_emb = retriever.embedding_models[model_key].encode([query_text], convert_to_numpy=True)[0]
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        
        txt_results = []
        for i in fwd_idx:
            e = embeddings[key][i]
            e_norm = e / (np.linalg.norm(e) + 1e-10)
            sim = float(np.dot(q_norm, e_norm))
            txt_results.append((players[i]['player_name'], sim, players[i]['goals']))
        txt_results.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (name, sim, goals) in enumerate(txt_results[:5], 1):
            print(f"   {rank}. {name:<25} Sim: {sim:.4f}  Goals: {goals}")
    
    # ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("TOP-10 RANKING AGREEMENT")
    print("─" * 80)
    
    # Get top 10 from each for all players
    all_top10 = {}
    
    # Numeric
    q = retriever.create_query_embedding({'goals': 'high', 'total_points': 'high', 'assists': 'high'})
    q_n = q / (np.linalg.norm(q) + 1e-10)
    res = [(players[i]['player_name'], float(np.dot(q_n, embeddings['numeric'][i]/np.linalg.norm(embeddings['numeric'][i]+1e-10)))) for i in range(len(players))]
    res.sort(key=lambda x: x[1], reverse=True)
    all_top10['Numeric'] = [x[0] for x in res[:10]]
    
    # Text models
    q_txt = "Best player with high goals assists and points"
    for label, key, mkey in [('MiniLM', 'minilm', 'model_1'), ('MPNet', 'mpnet', 'model_2')]:
        q_emb = retriever.embedding_models[mkey].encode([q_txt], convert_to_numpy=True)[0]
        q_n = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        res = []
        for i in range(len(players)):
            e = embeddings[key][i]
            e_n = e / (np.linalg.norm(e) + 1e-10)
            res.append((players[i]['player_name'], float(np.dot(q_n, e_n))))
        res.sort(key=lambda x: x[1], reverse=True)
        all_top10[label] = [x[0] for x in res[:10]]
    
    print(f"\n{'Rank':<6} {'Numeric':<22} {'MiniLM':<22} {'MPNet':<22}")
    print("─" * 75)
    for i in range(10):
        print(f"{i+1:<6} {all_top10['Numeric'][i]:<22} {all_top10['MiniLM'][i]:<22} {all_top10['MPNet'][i]:<22}")
    
    # Overlaps
    s1, s2, s3 = set(all_top10['Numeric']), set(all_top10['MiniLM']), set(all_top10['MPNet'])
    print(f"\n📊 Agreement:")
    print(f"   Numeric ∩ MiniLM: {len(s1&s2)}/10")
    print(f"   Numeric ∩ MPNet:  {len(s1&s3)}/10")
    print(f"   MiniLM ∩ MPNet:   {len(s2&s3)}/10")
    print(f"   All three:        {len(s1&s2&s3)}/10")
    
    retriever.close()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║ WHEN TO USE EACH APPROACH                                                 ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║ NUMERIC EMBEDDINGS:                                                       ║
    ║   → "Find players with high goals and low price"                          ║
    ║   → "Who has the best form among midfielders?"                            ║
    ║   → Stats-based filtering and ranking                                     ║
    ║                                                                           ║
    ║ TEXT + MiniLM:                                                            ║
    ║   → "Find a creative playmaker"                                           ║
    ║   → "Who is similar to De Bruyne?"                                        ║
    ║   → Real-time semantic search with good accuracy                          ║
    ║                                                                           ║
    ║ TEXT + MPNet:                                                             ║
    ║   → "Find a clinical finisher who performs in big games"                  ║
    ║   → Complex semantic understanding                                        ║
    ║   → When accuracy matters more than speed                                 ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)


def test_two_experiments():
    """
    Test the TWO EXPERIMENTS required by the project:
    
    Experiment 1: Baseline only (Cypher queries)
    Experiment 2: Baseline + Embeddings (combined results)
    """
    print("=" * 80)
    print("TWO EXPERIMENTS - BASELINE vs BASELINE + EMBEDDINGS")
    print("=" * 80)
    
    retriever = FPLGraphRetrieval(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD
    )

    # Ensure numeric embeddings are stored before running embedding experiments
    print("\n🔁 Ensuring numeric embeddings are stored (required for Experiment 2)...")
    store_info = retriever.store_embeddings_in_neo4j()
    print(f"   Embedding storage summary: {store_info}")
    
    test_queries = [
        "Who are the top forwards in 2022?",
        "Best midfielders with assists",
        "Top defenders this season",
        "Goalkeepers with clean sheets"
    ]
    
    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: \"{query}\"")
        print("=" * 80)
        
        # ─────────────────────────────────────────────────────────────
        # EXPERIMENT 1: BASELINE ONLY (Cypher Queries)
        # ─────────────────────────────────────────────────────────────
        print("\n📊 EXPERIMENT 1: BASELINE ONLY (Cypher Queries)")
        print("─" * 60)
        
        baseline_result = retriever.baseline_retrieve(query)
        
        print(f"   Intent: {baseline_result.get('intent')}")
        print(f"   Query Used: {baseline_result.get('query_used')}")
        print(f"   Parameters: {baseline_result.get('parameters')}")
        
        if baseline_result.get('error'):
            print(f"   ❌ Error: {baseline_result.get('error')}")
        else:
            print(f"   Results ({len(baseline_result.get('results', []))} found):")
            for i, r in enumerate(baseline_result.get('results', [])[:5], 1):
                player = r.get('player', r.get('player_name', 'Unknown'))
                points = r.get('total_points', r.get('points', '-'))
                print(f"      {i}. {player} - Points: {points}")

        # Post-check: verify that baseline result players have embeddings and PLAYS_FOR
        players_checked = baseline_result.get('results', [])[:10]
        with retriever.driver.session() as session:
            emb_missing = 0
            plays_for_missing = 0
            for p in players_checked:
                name = p.get('player') or p.get('player_name')
                rec = session.run(
                    "MATCH (p:Player {player_name: $name}) OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team) RETURN p.embedding IS NOT NULL AS has_emb, t.name AS team",
                    {'name': name}
                ).single()
                if not rec:
                    emb_missing += 1
                    plays_for_missing += 1
                else:
                    if not rec['has_emb']:
                        emb_missing += 1
                    if not rec['team']:
                        plays_for_missing += 1

        print(f"\n   Post-check: of top {len(players_checked)} baseline players -> missing embeddings: {emb_missing}, missing PLAYS_FOR: {plays_for_missing}")
        
        # ─────────────────────────────────────────────────────────────
        # EXPERIMENT 2: BASELINE + EMBEDDINGS (Combined)
        # ─────────────────────────────────────────────────────────────
        print("\n📊 EXPERIMENT 2: BASELINE + NUMERIC EMBEDDINGS")
        print("─" * 60)
        
        # Get embedding results
        preprocessed = retriever.preprocessor.preprocess(query, include_embedding=False)
        entities = preprocessed['entities']
        
        # Get position filter if available
        position = entities.get('positions', [None])[0]
        metrics = entities.get('metrics', [])
        
        # Create embedding criteria based on position and query
        # Use per-game average features (goals_per_game, etc.)
        criteria = {'avg_points_per_game': 'high'}
        
        # Position-specific criteria
        if position == 'FWD':
            # Forwards: prioritize goals
            criteria['goals_per_game'] = 'high'
            criteria['threat'] = 'high'
        elif position == 'MID':
            # Midfielders: balance goals and assists
            criteria['goals_per_game'] = 'high'
            criteria['assists_per_game'] = 'high'
            criteria['creativity'] = 'high'
        elif position == 'DEF':
            # Defenders: clean sheets and bonus
            criteria['clean_sheets_per_game'] = 'high'
            criteria['avg_bonus'] = 'high'
        elif position == 'GK':
            # Goalkeepers: clean sheets
            criteria['clean_sheets_per_game'] = 'high'
            criteria['avg_bonus'] = 'high'
        
        # Add explicit metric criteria from query
        if 'goals_scored' in metrics or 'goals' in str(query).lower():
            criteria['goals_per_game'] = 'high'
        if 'assists' in metrics or 'assists' in str(query).lower():
            criteria['assists_per_game'] = 'high'
        if 'clean_sheets' in metrics or 'clean' in str(query).lower():
            criteria['clean_sheets_per_game'] = 'high'
        
        embedding_result = retriever.embedding_retrieve(criteria=criteria, position=position, top_k=5)
        
        print(f"   Criteria: {embedding_result.get('criteria')}")
        print(f"   Position Filter: {embedding_result.get('position_filter')}")
        print(f"   Embedding Dims: {embedding_result.get('embedding_dimensions')}")
        print(f"   Results ({len(embedding_result.get('results', []))} found):")
        for i, r in enumerate(embedding_result.get('results', [])[:5], 1):
            print(f"      {i}. {r.get('player')} ({r.get('position')}) - Sim: {r.get('similarity_score', 0):.3f}")
        
        # ─────────────────────────────────────────────────────────────
        # COMBINED RESULTS
        # ─────────────────────────────────────────────────────────────
        print("\n📊 COMBINED RESULTS (Baseline + Embeddings merged)")
        print("─" * 60)
        
        baseline_players = baseline_result.get('results', [])
        embedding_players = embedding_result.get('results', [])
        
        combined = retriever._merge_results(baseline_players, embedding_players)
        
        print(f"   Baseline found: {len(baseline_players)} players")
        print(f"   Embeddings found: {len(embedding_players)} players")
        print(f"   Combined (deduplicated): {len(combined)} players")
        
        # Find overlap
        baseline_names = set(r.get('player', r.get('player_name', '')) for r in baseline_players[:10])
        embedding_names = set(r.get('player', '') for r in embedding_players[:10])
        overlap = baseline_names & embedding_names
        
        print(f"   Overlap (in top 10): {len(overlap)} players")
        if overlap:
            print(f"   Common players: {list(overlap)[:3]}")
    
    retriever.close()
    
    print("\n" + "=" * 80)
    print("✅ TWO EXPERIMENTS COMPLETE")
    print("=" * 80)
    print("""
    EXPERIMENT SUMMARY:
    
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ EXPERIMENT 1: BASELINE ONLY                                              │
    │ • Uses Cypher queries based on intent classification                     │
    │ • Exact match filtering (position, season, etc.)                         │
    │ • Returns structured data from Knowledge Graph                           │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ EXPERIMENT 2: BASELINE + EMBEDDINGS                                      │
    │ • Combines Cypher results with embedding similarity search               │
    │ • Numeric embeddings find statistically similar players                  │
    │ • Results are merged and deduplicated                                    │
    │ • Provides both exact matches AND similar alternatives                   │
    └──────────────────────────────────────────────────────────────────────────┘
    """)


def test_full_embedding_comparison():
    """
    Full test that stores ALL embedding types and compares them.
    This satisfies the project requirement for TWO embedding models.
    """
    print("=" * 80)
    print("FULL EMBEDDING MODEL COMPARISON TEST")
    print("Requirement: Experiment with at least TWO different embedding models")
    print("=" * 80)
    
    retriever = FPLGraphRetrieval(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD
    )
    
    # Step 1: Store ALL embedding types
    print("\n" + "─" * 80)
    print("STEP 1: STORE ALL EMBEDDING TYPES IN NEO4J")
    print("─" * 80)
    
    storage_results = retriever.store_all_embeddings()
    
    # Step 2: Compare search results across all models
    print("\n" + "─" * 80)
    print("STEP 2: COMPARE SEARCH RESULTS")
    print("─" * 80)
    
    for position in ['FWD', 'MID', 'DEF', 'GK']:
        print(f"\n{'='*80}")
        print(f"TESTING POSITION: {position}")
        print('='*80)
        retriever.compare_all_embedding_models(position=position, top_k=5)
    
    # Step 3: Find similar players comparison
    print("\n" + "─" * 80)
    print("STEP 3: FIND SIMILAR PLAYERS (ALL MODELS)")
    print("─" * 80)
    
    test_players = ["Erling Haaland", "Kevin De Bruyne", "Virgil van Dijk"]
    
    for player in test_players:
        print(f"\n🔍 Players similar to {player}:")
        
        # Numeric
        similar = retriever.find_similar_players(player, top_k=3, same_position=True)
        print(f"   Numeric: {[r['player'] for r in similar if 'error' not in r][:3]}")
    
    retriever.close()
    
    print("\n" + "=" * 80)
    print("✅ FULL COMPARISON TEST COMPLETE")
    print("=" * 80)
    print("""
    SUMMARY OF EMBEDDING MODELS COMPARED:
    
    ┌────────────────┬──────────┬─────────────────────────────────────────┐
    │ Model          │ Dims     │ Description                             │
    ├────────────────┼──────────┼─────────────────────────────────────────┤
    │ Numeric        │ 12       │ Direct normalized stats                 │
    │ MiniLM         │ 384      │ Text → sentence-transformers/all-MiniLM │
    │ MPNet          │ 768      │ Text → sentence-transformers/all-mpnet  │
    └────────────────┴──────────┴─────────────────────────────────────────┘
    
    This satisfies the requirement: "experiment with at least TWO different 
    embedding models for comparison" by comparing:
    1. MiniLM (all-MiniLM-L6-v2) - 384 dimensions
    2. MPNet (all-mpnet-base-v2) - 768 dimensions
    
    Additionally, we compare these TEXT embeddings with NUMERIC embeddings
    to show the tradeoffs between direct features vs semantic understanding.
    """)


def test_baseline_queries_run():
    """Run all 12 baseline Cypher queries with example parameters, verify
    they return results, and check that returned players have embeddings
    and a `PLAYS_FOR` relationship. Prints a concise report.
    """
    print("\n" + "=" * 80)
    print("RUNNING BASELINE QUERY TESTS (with embedding & PLAYS_FOR checks)")
    print("=" * 80)

    retriever = FPLGraphRetrieval(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD
    )

    # Ensure numeric embeddings exist for semantic checks
    print("\n🔁 Ensuring numeric embeddings are stored (required for embedding checks)...")
    try:
        store_info = retriever.store_embeddings_in_neo4j()
        print(f"   Embedding storage: {store_info}")
    except Exception as e:
        print(f"   ⚠️ Failed to store embeddings: {e}")

    # Define example parameters for each query template
    tests = [
        ('top_players_by_position', {'position': 'FWD', 'season': '2022-23', 'limit': 10}),
        ('player_gameweek_performance', {'player_name': 'Erling Haaland', 'gameweek': 20, 'season': '2022-23'}),
        ('compare_players', {'player_name': 'Erling Haaland', 'player_name2': 'Harry Kane', 'season': '2022-23'}),
        ('team_fixtures', {'team': 'Manchester City', 'season': '2022-23'}),
        ('players_by_team', {'team': 'Manchester City'}),
        ('top_scorers', {'season': '2022-23', 'limit': 10}),
        ('top_assisters', {'season': '2022-23', 'limit': 10}),
        ('clean_sheet_leaders', {'season': '2022-23', 'limit': 10}),
        ('players_by_form', {'season': '2022-23', 'gameweek': 20, 'limit': 10}),
        ('player_season_summary', {'player_name': 'Erling Haaland', 'season': '2022-23'}),
        ('bonus_leaders', {'season': '2022-23', 'limit': 10}),
        ('most_cards', {'season': '2022-23', 'limit': 10})
    ]

    summary = {}

    with retriever.driver.session() as session:
        for name, params in tests:
            print("\n" + "-" * 70)
            print(f"Running template: {name}  with params: {params}")
            try:
                results = retriever.execute_cypher(name, params)
            except Exception as e:
                print(f"   ❌ Query execution failed: {e}")
                summary[name] = {'status': 'error', 'error': str(e)}
                continue

            ok = bool(results)
            print(f"   Results returned: {len(results)}")

            # Extract up to 5 player names from results for per-player checks
            sample_players = []
            for r in results[:5]:
                # Common keys to look for
                if isinstance(r, dict):
                    if 'player' in r and r['player']:
                        sample_players.append(r['player'])
                    elif 'player_name' in r and r['player_name']:
                        sample_players.append(r['player_name'])
                    elif 'player1' in r and r['player1']:
                        sample_players.append(r['player1'])
                    elif 'player2' in r and r['player2']:
                        sample_players.append(r['player2'])
                    else:
                        # try any string-looking value
                        for v in r.values():
                            if isinstance(v, str) and len(v) > 1:
                                sample_players.append(v)
                                break

            sample_players = list(dict.fromkeys(sample_players))  # unique preserve order

            emb_missing = 0
            plays_for_missing = 0
            for pname in sample_players:
                rec = session.run(
                    "MATCH (p:Player {player_name: $name}) OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team) RETURN p.embedding IS NOT NULL AS has_emb, t.name AS team",
                    {'name': pname}
                ).single()
                if not rec:
                    emb_missing += 1
                    plays_for_missing += 1
                else:
                    if not rec['has_emb']:
                        emb_missing += 1
                    if not rec['team']:
                        plays_for_missing += 1

            summary[name] = {
                'results_count': len(results),
                'sample_players_checked': sample_players,
                'emb_missing_in_sample': emb_missing,
                'plays_for_missing_in_sample': plays_for_missing,
                'status': 'ok' if ok and emb_missing == 0 and plays_for_missing == 0 else 'warning' if ok else 'fail'
            }

            # Print quick per-query findings
            print(f"   Sample players checked: {sample_players}")
            print(f"   Embeddings missing in sample: {emb_missing}")
            print(f"   PLAYS_FOR missing in sample: {plays_for_missing}")

    retriever.close()

    print("\n" + "=" * 80)
    print("BASELINE QUERIES TEST SUMMARY")
    print("=" * 80)
    for k, v in summary.items():
        print(f"- {k}: status={v['status']}, results={v.get('results_count', 0)}, emb_missing={v.get('emb_missing_in_sample', 0)}, plays_for_missing={v.get('plays_for_missing_in_sample', 0)}")

    print("\nTests complete. Review warnings and errors above.")


def test_each_template_with_example_query():
    """Test each of the 12 baseline query templates with one hardcoded example query.
    
    For each template, this function:
    - Uses one realistic example query that will trigger that template
    - Calls retrieve(user_input, method='both')
    - Validates baseline/embedding/combined results exist and make sense
    - Checks up to 3 sample baseline players for p.embedding and PLAYS_FOR
    - Prints detailed per-template findings
    """
    print("\n" + "=" * 80)
    print("TESTING EACH TEMPLATE WITH HARDCODED EXAMPLE QUERIES")
    print("=" * 80)

    retriever = FPLGraphRetrieval(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD
    )

    # Ensure embeddings are stored before running tests
    print("\n🔁 Ensuring numeric embeddings are stored...")
    try:
        store_res = retriever.store_embeddings_in_neo4j()
        print(f"   Stored {store_res['embeddings_stored']} embeddings")
    except Exception as e:
        print(f"   ⚠️ Embedding storage failed: {e}")

    # Define one example query per template
    template_queries = {
        'top_players_by_position': "Who are the top forwards in 2022?",
        'player_gameweek_performance': "How did Erling Haaland perform in gameweek 20 of 2022-23?",
        'compare_players': "Compare Erling Haaland vs Harry Kane in 2022-23",
        'team_fixtures': "Show me Manchester City's fixtures in season 2022-23",
        'players_by_team': "Which players play for Manchester City?",
        'top_scorers': "Who are the top scorers in the 2022-23 season?",
        'top_assisters': "Who has the most assists in 2022-23?",
        'clean_sheet_leaders': "Which goalkeepers have the most clean sheets in 2022-23?",
        'players_by_form': "Who are the players with the best form around gameweek 20 in 2022-23?",
        'player_season_summary': "What is Erling Haaland's full season summary for 2022-23?",
        'bonus_leaders': "Who earned the most bonus points in 2022-23?",
        'most_cards': "Which players received the most cards (yellow/red) in 2022-23?"
    }

    results_summary = {}

    for template_name, example_query in template_queries.items():
        print("\n" + "=" * 80)
        print(f"TEMPLATE: {template_name}")
        print(f"QUERY: \"{example_query}\"")
        print("=" * 80)

        try:
            # Call retrieve with method='both' (baseline + embeddings)
            output = retriever.retrieve(example_query, method='both')
        except Exception as e:
            print(f"❌ ERROR: retrieve() raised exception: {e}")
            results_summary[template_name] = {
                'status': 'error',
                'error': str(e),
                'baseline_count': 0,
                'embedding_count': 0,
                'combined_count': 0,
                'notes': [f"exception: {str(e)}"]
            }
            continue

        baseline = output.get('baseline', {})
        embedding = output.get('embedding', {})
        combined = output.get('combined', [])

        # Extract counts
        baseline_results = baseline.get('results', []) if isinstance(baseline, dict) else []
        embedding_results = embedding.get('results', []) if isinstance(embedding, dict) else []

        baseline_count = len(baseline_results)
        embedding_count = len(embedding_results)
        combined_count = len(combined)

        print(f"\n📊 BASELINE RETRIEVAL:")
        print(f"   Intent: {baseline.get('intent', 'unknown')}")
        print(f"   Query Used: {baseline.get('query_used', 'unknown')}")
        print(f"   Results Count: {baseline_count}")
        if baseline_results:
            for i, r in enumerate(baseline_results[:3], 1):
                name = r.get('player') or r.get('player_name', 'Unknown')
                points = r.get('total_points') or r.get('points', '-')
                print(f"      {i}. {name} - Points: {points}")

        print(f"\n📊 EMBEDDING RETRIEVAL:")
        print(f"   Criteria: {embedding.get('criteria', {})}")
        print(f"   Position Filter: {embedding.get('position_filter')}")
        print(f"   Results Count: {embedding_count}")
        if embedding_results:
            for i, r in enumerate(embedding_results[:3], 1):
                name = r.get('player', 'Unknown')
                sim = r.get('similarity_score', 0)
                print(f"      {i}. {name} - Similarity: {sim:.3f}")

        print(f"\n📊 COMBINED RESULTS:")
        print(f"   Total (deduplicated): {combined_count}")

        # Check sample baseline players for embeddings and PLAYS_FOR
        sample_players = []
        for r in baseline_results[:3]:
            if isinstance(r, dict):
                name = r.get('player') or r.get('player_name') or r.get('player1') or r.get('player2')
                if name:
                    sample_players.append(name)

        emb_missing = 0
        plays_for_missing = 0
        player_status = []

        with retriever.driver.session() as session:
            for name in sample_players:
                rec = session.run(
                    "MATCH (p:Player {player_name: $name}) OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team) RETURN p.embedding IS NOT NULL AS has_emb, t.name AS team",
                    {'name': name}
                ).single()
                if not rec:
                    emb_missing += 1
                    plays_for_missing += 1
                    player_status.append({'player': name, 'has_embedding': False, 'has_plays_for': False})
                else:
                    has_emb = bool(rec['has_emb'])
                    team = rec['team']
                    if not has_emb:
                        emb_missing += 1
                    if not team:
                        plays_for_missing += 1
                    player_status.append({'player': name, 'has_embedding': has_emb, 'has_plays_for': bool(team)})

        print(f"\n✓ PLAYER VALIDATION (sample of {len(sample_players)}):")
        for ps in player_status:
            emb_icon = "✅" if ps['has_embedding'] else "❌"
            pf_icon = "✅" if ps['has_plays_for'] else "❌"
            print(f"   {ps['player']}: embedding {emb_icon}, PLAYS_FOR {pf_icon}")

        status = 'ok'
        notes = []

        if baseline_count == 0:
            notes.append('no baseline results')
            status = 'warning'
        if embedding_count == 0:
            notes.append('no embedding results')
            if status == 'ok':
                status = 'warning'
        if emb_missing > 0:
            notes.append(f'embedding missing: {emb_missing}')
        if plays_for_missing > 0:
            notes.append(f'PLAYS_FOR missing: {plays_for_missing}')

        results_summary[template_name] = {
            'status': status,
            'baseline_count': baseline_count,
            'embedding_count': embedding_count,
            'combined_count': combined_count,
            'sample_players_checked': len(sample_players),
            'embedding_missing': emb_missing,
            'plays_for_missing': plays_for_missing,
            'notes': notes
        }

        print(f"\n   Status: {status}, Notes: {notes}")

    retriever.close()

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL 12 TEMPLATES")
    print("=" * 80)
    print(f"\n{'Template':<35} {'Status':<10} {'Baseline':<10} {'Embedding':<10} {'Combined':<10}")
    print("-" * 95)
    for template, res in results_summary.items():
        status = res['status']
        baseline = res['baseline_count']
        embedding = res['embedding_count']
        combined = res['combined_count']
        print(f"{template:<35} {status:<10} {baseline:<10} {embedding:<10} {combined:<10}")

    print(f"\n{'Notes:'}")
    for template, res in results_summary.items():
        if res['notes']:
            print(f"  {template}: {', '.join(res['notes'])}")

    print("\n✅ TEMPLATE TEST COMPLETE")


if __name__ == "__main__":
    NEO4J_URI = "neo4j+s://1da86c19.databases.neo4j.io"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "HA4iunTOGen7RYpeISs3ZRhcWjpcokqam9przCqCuQ8"
    
    # Initialize retriever
    retriever = FPLGraphRetrieval(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD
    )
    
    print("=" * 80)
    print("STORING ALL EMBEDDINGS IN NEO4J")
    print("=" * 80)
    
    # Store ALL embedding types
    results = retriever.store_all_embeddings()
    
    print("\n✅ ALL EMBEDDINGS STORED")
    print(f"Numeric: {results['numeric'].get('embeddings_stored')} players")
    print(f"MiniLM: {results['minilm'].get('embeddings_stored')} players")
    print(f"MPNet: {results['mpnet'].get('embeddings_stored')} players")
    
    # Optional: Run the template tests after
    print("\n" + "=" * 80)
    print("RUNNING TEMPLATE TESTS")
    print("=" * 80)
    test_each_template_with_example_query()
