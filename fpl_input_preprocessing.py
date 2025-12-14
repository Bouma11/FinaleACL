
"""
Fixed FPL Input Preprocessing with improved intent classification and entity extraction
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
from huggingface_hub import InferenceClient


class FPLInputPreprocessorLLM:
    """
    Enhanced FPL preprocessor with improved intent classification and entity extraction
    """
    
    def __init__(self, 
                 hf_token: str,
                 llm_model: str = "google/gemma-2-2b-it",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_llm: bool = True):
        """Initialize the preprocessor"""
        print("🚀 Initializing FPL Input Preprocessor (Enhanced)...")
        
        self.use_llm = use_llm
        
        # Initialize LLM client if enabled
        if use_llm:
            try:
                self.llm_client = InferenceClient(model=llm_model, token=hf_token)
                print(f"✅ Connected to LLM: {llm_model}")
            except Exception as e:
                print(f"⚠️ LLM connection failed: {e}")
                print("⚠️ Falling back to rule-based approach")
                self.use_llm = False
        
        # Initialize embedding model
        self.embedder = SentenceTransformer(embedding_model_name)
        print(f"✅ Loaded embedding model: {embedding_model_name}")
        
        # Define entity schemas for LLM prompts
        self.intent_types = [
            'player_search',
            'performance_query', 
            'comparison',
            'recommendation',
            'team_analysis',
            'fixture_query',
            'value_analysis',
            'form_query',
            'general_query'
        ]
        
        self._init_rule_based_entities()
        self.error_log = []
        print("✅ System initialized\n")
    
    def _init_rule_based_entities(self):
        """Initialize rule-based entity dictionaries"""
        self.positions = {
            'GK': ['gk', 'goalkeeper', 'keeper', 'goalie'],
            'DEF': ['def', 'defender', 'defence', 'defense', 'back'],
            'MID': ['mid', 'midfielder', 'midfield'],
            'FWD': ['fwd', 'forward', 'striker', 'attacker', 'attack']
        }
        
        self.teams = {
            'Arsenal': ['arsenal', 'ars', 'gunners'],
            'Aston Villa': ['aston villa', 'villa', 'avl'],
            'Bournemouth': ['bournemouth', 'bou', 'cherries'],
            'Brentford': ['brentford', 'bre', 'bees'],
            'Brighton': ['brighton', 'bha', 'seagulls'],
            'Burnley': ['burnley', 'bur', 'clarets'],
            'Chelsea': ['chelsea', 'che', 'blues'],
            'Crystal Palace': ['crystal palace', 'palace', 'cry', 'eagles'],
            'Everton': ['everton', 'eve', 'toffees'],
            'Fulham': ['fulham', 'ful', 'cottagers'],
            'Liverpool': ['liverpool', 'liv', 'reds'],
            'Luton': ['luton', 'lut', 'hatters'],
            'Man City': ['man city', 'manchester city', 'city', 'mci', 'mcfc'],
            'Man Utd': ['man utd', 'manchester united', 'united', 'mun', 'mufc'],
            'Newcastle': ['newcastle', 'new', 'magpies', 'toon'],
            'Nottingham Forest': ['nottingham', 'forest', 'nfo', 'nottm forest'],
            'Sheffield Utd': ['sheffield', 'sheffield united', 'shu', 'blades'],
            'Tottenham': ['tottenham', 'spurs', 'tot', 'thfc'],
            'West Ham': ['west ham', 'whu', 'hammers'],
            'Wolves': ['wolves', 'wolverhampton', 'wol'],
            'Leeds': ['leeds', 'leeds united', 'lee', 'lufc'],
            'Leicester': ['leicester', 'leicester city', 'lei', 'lcfc', 'foxes'],
            'Southampton': ['southampton', 'saints', 'sou'],
        }
        
        self.metrics = {
            'total_points': ['points', 'total points', 'score'],
            'goals_scored': ['goals', 'goal', 'scored', 'scorer', 'scorers', 'scoring'],
            'assists': ['assists', 'assist', 'assisting'],
            'bonus': ['bonus', 'bonus points', 'bps'],
            'minutes': ['minutes', 'playing time'],
            'clean_sheets': ['clean sheets', 'clean sheet', 'cs', 'cleansheet'],
            'saves': ['saves', 'save'],
            'ict_index': ['ict', 'ict index'],
            'influence': ['influence'],
            'creativity': ['creativity'],
            'threat': ['threat'],
            'form': ['form', 'recent form', 'hot', 'recent', 'lately'],
            'cards': ['cards', 'yellow cards', 'red cards', 'yellow', 'red', 'disciplinary']
        }
    
    def classify_intent(self, text: str) -> Tuple[str, float]:
        """
        Enhanced intent classification with better pattern matching.
        """
        text_lower = text.lower()
        
        # Top/best/leading queries (MOST COMMON)
        top_keywords = ['top', 'best', 'leading', 'highest', 'most']
        if any(keyword in text_lower for keyword in top_keywords):
            # Check for specific metrics
            if any(word in text_lower for word in ['scorer', 'scorers', 'goals', 'goal']):
                return 'recommendation', 0.95  # Will trigger top_scorers
            if any(word in text_lower for word in ['assist', 'assists']):
                return 'recommendation', 0.95  # Will trigger top_assisters
            if any(word in text_lower for word in ['clean sheet', 'cleansheet']):
                return 'recommendation', 0.95  # Will trigger clean_sheet_leaders
            if any(word in text_lower for word in ['bonus', 'bonus points']):
                return 'recommendation', 0.95  # Will trigger bonus_leaders
            return 'recommendation', 0.90  # Generic top query
        
        # Comparison queries
        if any(word in text_lower for word in ['compare', 'vs', 'versus', 'between']):
            return 'comparison', 0.90
        
        # Fixture queries
        if any(word in text_lower for word in ['fixture', 'fixtures', 'schedule', 'match', 'matches']):
            return 'fixture_query', 0.90
        
        # Team queries
        if any(word in text_lower for word in ['players for', 'play for', 'roster', 'squad']):
            return 'team_analysis', 0.85
        
        # Form queries
        if any(word in text_lower for word in ['form', 'recent', 'lately', 'hot']):
            return 'form_query', 0.85
        
        # Specific player queries (gameweek performance)
        if 'gameweek' in text_lower or 'gw' in text_lower:
            return 'performance_query', 0.85
        
        # Performance queries (specific player)
        known_players = ['haaland', 'kane', 'salah', 'de bruyne', 'son']
        if any(player in text_lower for player in known_players):
            if 'gameweek' in text_lower or 'gw' in text_lower:
                return 'performance_query', 0.90
            return 'player_search', 0.80
        
        return 'general_query', 0.30
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Enhanced entity extraction with better metric detection.
        """
        entities = {
            'players': [],
            'teams': [],
            'positions': [],
            'seasons': [],
            'gameweeks': [],
            'metrics': [],
            'numerical_values': [],
            'comparators': []
        }
        
        text_lower = text.lower()
        
        # Metrics - EXPANDED
        for canonical_metric, variants in self.metrics.items():
            if any(v in text_lower for v in variants):
                if canonical_metric not in entities['metrics']:
                    entities['metrics'].append(canonical_metric)
        
        # Players - Known players
        known_players = [
            'Erling Haaland', 'Harry Kane', 'Mohamed Salah', 'Kevin De Bruyne',
            'Heung-Min Son', 'Ivan Toney', 'Marcus Rashford', 'Bukayo Saka',
            'Virgil van Dijk', 'Alisson', 'Bruno Fernandes', 'Phil Foden'
        ]
        for player in known_players:
            if player.lower() in text_lower:
                entities['players'].append(player)
        
        # If no known players found, try to extract capitalized names
        if not entities['players']:
            stopwords = {'who', 'what', 'find', 'show', 'tell', 'compare', 'the', 'vs', 'v', 
                        'and', 'or', 'in', 'for', 'of', 'did', 'are', 'have', 'has', 'is', 
                        'how', 'which', 'where', 'when', 'why', 'do', 'does', 'received', 
                        'earned', 'had', 'players', 'playing', 'play'}
            words = text.split()
            potential_players = []
            i = 0
            
            while i < len(words):
                word = words[i]
                clean_word = word.rstrip("'s,?!")
                
                if len(clean_word) >= 2 and clean_word[0].isupper() and clean_word.lower() not in stopwords:
                    name_parts = [clean_word]
                    j = i + 1
                    
                    while j < len(words) and len(name_parts) < 3:
                        next_word = words[j].rstrip("'s,?!")
                        if (len(next_word) >= 2 and next_word[0].isupper() 
                            and next_word.lower() not in stopwords):
                            name_parts.append(next_word)
                            j += 1
                        else:
                            break
                    
                    if len(name_parts) >= 1:
                        player_name = ' '.join(name_parts)
                        if len(player_name) >= 2:
                            potential_players.append(player_name)
                        i = j
                else:
                    i += 1
            
            entities['players'] = list(dict.fromkeys(potential_players))
        
        # Positions
        for canonical_pos, variants in self.positions.items():
            if any(v in text_lower for v in variants):
                if canonical_pos not in entities['positions']:
                    entities['positions'].append(canonical_pos)
        
        # Teams
        for canonical_team, variants in self.teams.items():
            for variant in variants:
                if variant in text_lower:
                    if canonical_team not in entities['teams']:
                        entities['teams'].append(canonical_team)
                    break
        
        # Seasons
        season_patterns = [
            r'20\d{2}[-/]20?\d{2}',  # 2022-23 or 2022/23
            r'20\d{2}'                # 2022
        ]
        for pattern in season_patterns:
            matches = re.findall(pattern, text)
            entities['seasons'].extend(matches)
        
        # Gameweeks
        gw_pattern = r'(?:gameweek|gw)\s*(\d+)'
        gw_matches = re.findall(gw_pattern, text_lower)
        entities['gameweeks'].extend(gw_matches)
        
        # Comparators
        comparator_pattern = r'(more than|less than|under|over|above|below)'
        entities['comparators'] = re.findall(comparator_pattern, text_lower)
        
        # Numbers
        number_pattern = r'\b(\d+(?:\.\d+)?)\b'
        numbers = re.findall(number_pattern, text)
        filtered_numbers = [n for n in numbers 
                          if n not in entities['seasons'] and n not in entities['gameweeks']]
        entities['numerical_values'] = [float(n) for n in filtered_numbers]
        
        return entities
    
    def generate_embedding(self, user_input: str) -> np.ndarray:
        """Generate semantic text embedding vector (384 dims for MiniLM)."""
        try:
            embedding = self.embedder.encode([user_input], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            self.error_log.append({
                'type': 'embedding_error',
                'query': user_input,
                'error': str(e)
            })
            return np.zeros(384)
    
    def entities_to_criteria(self, entities: Dict, position: str = None) -> Dict[str, str]:
        """Convert extracted entities to search criteria for numeric embeddings."""
        criteria = {'total_points': 'high'}  # Default: find high performers
        
        positions_list = entities.get('positions', [])
        pos = position if position else (positions_list[0] if positions_list else None)
        
        # Position-specific default criteria
        if pos == 'FWD':
            criteria['goals'] = 'high'
            criteria['threat'] = 'high'
        elif pos == 'MID':
            criteria['goals'] = 'high'
            criteria['assists'] = 'high'
            criteria['creativity'] = 'high'
        elif pos == 'DEF':
            criteria['clean_sheets'] = 'high'
            criteria['bonus'] = 'high'
        elif pos == 'GK':
            criteria['clean_sheets'] = 'high'
            criteria['bonus'] = 'high'
        
        # Override with explicit metrics from query
        metrics = entities.get('metrics', [])
        
        # Map extracted metrics to criteria features
        metric_to_criteria = {
            'goals_scored': 'goals',
            'total_points': 'total_points',
            'assists': 'assists',
            'clean_sheets': 'clean_sheets',
            'bonus': 'bonus',
            'minutes': 'minutes',
            'form': 'form',
            'ict_index': 'ict_index',
            'influence': 'influence',
            'creativity': 'creativity',
            'threat': 'threat',
            'cards': 'cards'
        }
        
        for metric in metrics:
            if metric in metric_to_criteria:
                criteria[metric_to_criteria[metric]] = 'high'
        
        return criteria
    
    def generate_numeric_embedding(self, criteria: Dict) -> np.ndarray:
        """Generate a 12-dimensional numeric query embedding."""
        embedding = np.full(12, 0.5, dtype=np.float32)
        
        feature_map = {
            'goals': 0, 
            'assists': 1, 
            'total_points': 2,
            'points': 2,
            'clean_sheets': 3, 
            'minutes': 4, 
            'bonus': 5,
            'form': 6, 
            'ict_index': 7, 
            'influence': 8, 
            'creativity': 9, 
            'threat': 10, 
            'games_played': 11,
            'cards': 11  # Map cards to games_played slot for now
        }
        
        for feature, value in criteria.items():
            if feature in feature_map:
                idx = feature_map[feature]
                if value == 'high':
                    embedding[idx] = 1.0
                elif value == 'low':
                    embedding[idx] = 0.0
                elif isinstance(value, (int, float)):
                    embedding[idx] = min(max(float(value), 0.0), 1.0)
        
        return embedding
    
    def preprocess(self, user_input: str, include_embedding: bool = True) -> Dict:
        """Complete preprocessing pipeline."""
        if not user_input or not user_input.strip():
            self.error_log.append({
                'type': 'empty_query',
                'query': user_input
            })
            return {
                'original_query': user_input,
                'intent': 'unknown',
                'intent_confidence': 0.0,
                'entities': {},
                'text_embedding': None,
                'numeric_embedding': None,
                'search_criteria': {},
                'method': 'none',
                'error': 'Empty query provided'
            }
        
        intent, confidence = self.classify_intent(user_input)
        entities = self.extract_entities(user_input)
        
        # Generate search criteria from entities
        search_criteria = self.entities_to_criteria(entities)
        
        result = {
            'original_query': user_input,
            'intent': intent,
            'intent_confidence': confidence,
            'entities': entities,
            'method': 'llm' if self.use_llm else 'rule-based',
            'search_criteria': search_criteria,
        }
        
        if include_embedding:
            result['text_embedding'] = self.generate_embedding(user_input)
            result['numeric_embedding'] = self.generate_numeric_embedding(search_criteria)
        else:
            result['text_embedding'] = None
            result['numeric_embedding'] = None
        
        result['embedding'] = result['text_embedding']
        
        return result
    
    def _convert_to_season_format(self, year_str: str) -> str:
        """Convert year to KG season format."""
        try:
            year = int(year_str)
            next_year = str(year + 1)[-2:]
            return f"{year}-{next_year}"
        except ValueError:
            return year_str
    
    def get_cypher_params(self, preprocessing_result: Dict) -> Dict[str, any]:
        """Convert entities to Cypher parameters."""
        entities = preprocessing_result['entities']
        
        params = {'intent': preprocessing_result['intent']}
        
        if entities.get('positions'):
            params['position'] = entities['positions'][0]
        
        if entities.get('teams'):
            params['team'] = entities['teams'][0]
            if len(entities['teams']) > 1:
                params['team2'] = entities['teams'][1]
        
        if entities.get('seasons'):
            raw_season = entities['seasons'][0]
            params['season'] = self._convert_to_season_format(raw_season)
        
        if entities.get('gameweeks'):
            params['gameweek'] = int(entities['gameweeks'][0])
        
        if entities.get('metrics'):
            params['metric'] = entities['metrics'][0]
            if len(entities['metrics']) > 1:
                params['metric2'] = entities['metrics'][1]
        
        if entities.get('players'):
            params['player_name'] = entities['players'][0]
            if len(entities['players']) > 1:
                params['player_name2'] = entities['players'][1]
        
        if entities.get('numerical_values'):
            params['threshold'] = entities['numerical_values'][0]
        
        if entities.get('comparators'):
            params['comparator'] = entities['comparators'][0]
        
        return params