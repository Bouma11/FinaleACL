# %%writefile fpl_input_preprocessing.py


"""
Enhanced FPL Input Preprocessor with Robust LLM-based Multi-Entity Extraction
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
    Enhanced FPL preprocessor with improved LLM-based multi-entity extraction
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
        
        # Define intent types
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
        self._init_player_database()
        self.error_log = []
        print("✅ System initialized\n")
    
    def _init_rule_based_entities(self):
        """Initialize rule-based entity dictionaries"""
        self.positions = {
            'GK': ['gk', 'goalkeeper', 'keeper', 'goalie', 'keepers', 'goalkeepers'],
            'DEF': ['def', 'defender', 'defence', 'defense', 'back', 'defenders', 'defensive'],
            'MID': ['mid', 'midfielder', 'midfield', 'midfielders'],
            'FWD': ['fwd', 'forward', 'striker', 'attacker', 'attack', 'forwards', 'strikers', 'attackers']
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
            'total_points': ['points', 'total points', 'score', 'scoring'],
            'goals_scored': ['goals', 'goal', 'scored', 'scorer', 'scorers', 'scoring goals'],
            'assists': ['assists', 'assist', 'assisting', 'assist provider'],
            'bonus': ['bonus', 'bonus points', 'bps'],
            'minutes': ['minutes', 'playing time', 'game time'],
            'clean_sheets': ['clean sheets', 'clean sheet', 'cs', 'cleansheet', 'cleansheets'],
            'saves': ['saves', 'save', 'saved'],
            'ict_index': ['ict', 'ict index'],
            'influence': ['influence', 'influential'],
            'creativity': ['creativity', 'creative'],
            'threat': ['threat', 'threatening'],
            'form': ['form', 'recent form', 'hot', 'recent', 'lately', 'in form'],
            'cards': ['cards', 'yellow cards', 'red cards', 'yellow', 'red', 'disciplinary'],
            'goals_conceded': ['goals conceded', 'conceded', 'goals against'],
            'own_goals': ['own goals', 'own goal', 'og'],
            'penalties_saved': ['penalties saved', 'penalty saves', 'pen saves'],
            'penalties_missed': ['penalties missed', 'penalty misses', 'pen misses'],
            'yellow_cards': ['yellow cards', 'yellows', 'bookings'],
            'red_cards': ['red cards', 'reds', 'sent off', 'dismissals'],
            'bps': ['bps', 'bonus point system'],
            'value': ['value', 'price', 'cost', 'worth', 'bargain', 'expensive'],
            'ownership': ['ownership', 'owned', 'selected', '% owned']
        }
    
    def _init_player_database(self):
        """Initialize comprehensive player database with variations"""
        self.known_players = {
            # Top Forwards
            'Erling Haaland': ['haaland', 'erling', 'erling haaland'],
            'Harry Kane': ['kane', 'harry kane'],
            'Ivan Toney': ['toney', 'ivan toney'],
            'Alexander Isak': ['isak', 'alexander isak'],
            'Ollie Watkins': ['watkins', 'ollie watkins', 'oliver watkins'],
            'Darwin Nunez': ['nunez', 'darwin', 'darwin nunez'],
            'Dominic Calvert-Lewin': ['calvert-lewin', 'dcl', 'dominic calvert-lewin'],
            'Callum Wilson': ['wilson', 'callum wilson'],
            
            # Top Midfielders
            'Mohamed Salah': ['salah', 'mo salah', 'mohamed salah'],
            'Kevin De Bruyne': ['de bruyne', 'kdb', 'kevin de bruyne'],
            'Heung-Min Son': ['son', 'heung-min son', 'sonny'],
            'Bukayo Saka': ['saka', 'bukayo saka'],
            'Phil Foden': ['foden', 'phil foden'],
            'Bruno Fernandes': ['bruno', 'fernandes', 'bruno fernandes'],
            'Martin Odegaard': ['odegaard', 'martin odegaard'],
            'James Maddison': ['maddison', 'james maddison', 'madders'],
            'Marcus Rashford': ['rashford', 'marcus rashford'],
            'Jack Grealish': ['grealish', 'jack grealish'],
            
            # Top Defenders
            'Trent Alexander-Arnold': ['trent', 'taa', 'alexander-arnold', 'trent alexander-arnold'],
            'Reece James': ['reece james', 'reece'],
            'Kieran Trippier': ['trippier', 'kieran trippier'],
            'Andrew Robertson': ['robertson', 'andy robertson', 'andrew robertson', 'robbo'],
            'Virgil van Dijk': ['van dijk', 'virgil', 'virgil van dijk', 'vvd'],
            'William Saliba': ['saliba', 'william saliba'],
            'Ruben Dias': ['dias', 'ruben dias'],
            'Gabriel Magalhaes': ['gabriel', 'gabriel magalhaes'],
            
            # Top Goalkeepers
            'Alisson': ['alisson', 'alisson becker'],
            'Ederson': ['ederson', 'ederson moraes'],
            'Aaron Ramsdale': ['ramsdale', 'aaron ramsdale'],
            'Nick Pope': ['pope', 'nick pope'],
            'David Raya': ['raya', 'david raya'],
            'Robert Sanchez': ['sanchez', 'robert sanchez']
        }
        
        # Create reverse mapping for quick lookup
        self.player_variants = {}
        for player, variants in self.known_players.items():
            for variant in variants:
                self.player_variants[variant] = player
    
    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Enhanced intent classification with LLM support"""
        text_lower = text.lower()
        
        # ===== EMERGENCY HOTFIX =====
        # Force performance_query for any question about specific gameweek performance
        gameweek_patterns = [
            r'performance.*gameweek.*\d+',
            r'gameweek.*\d+.*performance',
            r'how.*did.*gameweek.*\d+',
            r'what.*was.*gameweek.*\d+'
        ]
        for pattern in gameweek_patterns:
           if re.search(pattern, text_lower, re.IGNORECASE):
                return 'performance_query', 0.99  # Highest confidence
    # ===== END HOTFIX =====
        # Rule-based classification (fast path)
        
        # Top/best/leading queries
        top_keywords = ['top', 'best', 'leading', 'highest', 'most', 'who scored', 'who has']
        if any(keyword in text_lower for keyword in top_keywords):
            if any(word in text_lower for word in ['scorer', 'scorers', 'goals', 'goal']):
                return 'recommendation', 0.95
            if any(word in text_lower for word in ['assist', 'assists']):
                return 'recommendation', 0.95
            if any(word in text_lower for word in ['clean sheet', 'cleansheet']):
                return 'recommendation', 0.95
            if any(word in text_lower for word in ['bonus', 'bonus points']):
                return 'recommendation', 0.95
            return 'recommendation', 0.90
        
        # Comparison queries (check for multiple entities)
        comparison_keywords = ['compare', 'vs', 'versus', 'between', 'or', 'and', 'versus']
        if any(word in text_lower for word in comparison_keywords):
            # Check if there are at least two potential entities for comparison
            potential_players = self._extract_capitalized_names(text)
            if len(potential_players) >= 2 or len(re.findall(r'\b(vs|versus|compare)\b', text_lower)) > 0:
                return 'comparison', 0.95
            return 'comparison', 0.85
        
        # Fixture queries
        if any(word in text_lower for word in ['fixture', 'fixtures', 'schedule', 'match', 'matches', 'playing against', 'upcoming']):
            return 'fixture_query', 0.90
        
        # Team analysis queries
        if any(word in text_lower for word in ['players for', 'play for', 'roster', 'squad', 'team', 'from arsenal', 'from liverpool']):
            return 'team_analysis', 0.85
        
        # Form queries
        if any(word in text_lower for word in ['form', 'recent', 'lately', 'hot', 'in form', 'last 5', 'recently']):
            return 'form_query', 0.85
        
        # Performance queries (gameweek-specific)
        if 'gameweek' in text_lower or 'gw' in text_lower:
            return 'performance_query', 0.85
        
        # Player search (specific player names)
        if any(player.lower() in text_lower for variants in self.known_players.values() for player in variants):
            if 'gameweek' in text_lower or 'gw' in text_lower:
                return 'performance_query', 0.90
            return 'player_search', 0.80
        
        # Value analysis
        if any(word in text_lower for word in ['value', 'price', 'cost', 'worth', 'bargain', 'differential', 'cheap', 'expensive']):
            return 'value_analysis', 0.85
        
        # Use LLM for ambiguous cases
        if self.use_llm:
            try:
                return self._classify_intent_with_llm(text)
            except Exception as e:
                self.error_log.append({'type': 'llm_classification_error', 'error': str(e)})
        
        return 'general_query', 0.30
    
    def _classify_intent_with_llm(self, text: str) -> Tuple[str, float]:
        """Use LLM for intent classification"""
        prompt = f"""Analyze this Fantasy Premier League query and classify its intent:

Query: "{text}"

Available intents:
- player_search: Finding specific player information (e.g., "Haaland stats", "Salah price")
- performance_query: Player performance in specific gameweek/match (e.g., "Salah points GW5", "Haaland gameweek 3")
- comparison: Comparing multiple players or teams (e.g., "Haaland vs Salah", "compare TAA and Robertson")
- recommendation: Finding top/best players (e.g., "top scorers", "best defenders", "highest points")
- team_analysis: Team-related queries (e.g., "Arsenal players", "Liverpool squad")
- fixture_query: Match schedule and fixtures (e.g., "Man City fixtures", "upcoming matches")
- value_analysis: Player value and pricing (e.g., "best value defenders", "Haaland worth it")
- form_query: Recent player form (e.g., "players in form", "recent performance")
- general_query: General questions

Determine the SINGLE most appropriate intent. Return format: intent,confidence (0.0-1.0)

Response:"""
        
        response = self.llm_client.text_generation(prompt, max_new_tokens=50)
        parts = response.strip().split(',')
        if len(parts) == 2:
            intent = parts[0].strip().lower()
            try:
                confidence = float(parts[1].strip())
                # Validate intent
                if intent in self.intent_types:
                    return intent, confidence
            except ValueError:
                pass
        return 'general_query', 0.30
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Enhanced multi-entity extraction with improved LLM support"""
        # Always run rule-based extraction first for robustness
        rule_entities = self._extract_entities_rule_based(text)
        
        # Use LLM for enhanced extraction if available
        if self.use_llm:
            try:
                llm_entities = self._extract_entities_with_llm(text)
                return self._merge_entities(llm_entities, rule_entities)
            except Exception as e:
                self.error_log.append({'type': 'llm_extraction_error', 'error': str(e)})
        
        return rule_entities
    
    def _extract_entities_with_llm(self, text: str) -> Dict[str, List[str]]:
        """Use LLM for comprehensive entity extraction with multi-entity support"""
        prompt = f"""Extract ALL entities from this Fantasy Premier League query.

Query: "{text}"

Extract entities for ALL these categories:
1. players: ALL player names mentioned (extract ALL names)
2. teams: ALL team names mentioned
3. positions: GK, DEF, MID, FWD (only these values)
4. seasons: Year ranges like 2023-24, 2023/24, or single years
5. gameweeks: Gameweek numbers (extract ALL numbers mentioned as gameweeks)
6. metrics: Statistics mentioned (goals, assists, points, clean_sheets, bonus, form, etc.)
7. numerical_values: Any numbers that are thresholds, counts, or values
8. comparators: Comparison words (more than, less than, over, under, above, below)

CRITICAL INSTRUCTIONS:
- Extract ALL instances of each entity type. If multiple players are mentioned, extract ALL of them.
- For players: Use full canonical names when possible (e.g., "Erling Haaland" not just "Haaland")
- For teams: Use full canonical names (e.g., "Man City" not just "City")
- For metrics: Use standardized names from: total_points, goals_scored, assists, bonus, minutes, clean_sheets, saves, ict_index, influence, creativity, threat, form, cards, goals_conceded, own_goals, penalties_saved, penalties_missed, yellow_cards, red_cards, bps, value, ownership
- Return ONLY valid JSON with exactly these 8 keys.

Example response for "Compare Haaland and Salah goals vs assists in gameweek 5":
{{
  "players": ["Erling Haaland", "Mohamed Salah"],
  "teams": [],
  "positions": [],
  "seasons": [],
  "gameweeks": ["5"],
  "metrics": ["goals_scored", "assists"],
  "numerical_values": [],
  "comparators": []
}}

Now extract entities from the query above:"""

        try:
            response = self.llm_client.text_generation(prompt, max_new_tokens=500)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    entities = json.loads(json_match.group())
                    
                    # Validate and clean entities
                    cleaned_entities = self._clean_llm_entities(entities, text)
                    
                    # Ensure all required keys exist
                    required_keys = ['players', 'teams', 'positions', 'seasons', 'gameweeks', 
                                   'metrics', 'numerical_values', 'comparators']
                    for key in required_keys:
                        if key not in cleaned_entities:
                            cleaned_entities[key] = []
                        elif not isinstance(cleaned_entities[key], list):
                            cleaned_entities[key] = [str(cleaned_entities[key])]
                    
                    return cleaned_entities
                except json.JSONDecodeError as e:
                    self.error_log.append({
                        'type': 'llm_json_parse_error',
                        'response': response[:200],
                        'error': str(e)
                    })
        except Exception as e:
            self.error_log.append({
                'type': 'llm_extraction_failure',
                'error': str(e)
            })
        
        return {
            'players': [], 'teams': [], 'positions': [], 'seasons': [],
            'gameweeks': [], 'metrics': [], 'numerical_values': [], 'comparators': []
        }
    
    def _clean_llm_entities(self, entities: Dict, original_text: str) -> Dict:
        """Clean and validate LLM-extracted entities"""
        cleaned = {}
        text_lower = original_text.lower()
        
        for key, values in entities.items():
            if not isinstance(values, list):
                values = [values] if values else []
            
            cleaned_values = []
            seen = set()
            
            for value in values:
                if not value:
                    continue
                
                # Convert to string and strip whitespace
                str_value = str(value).strip()
                
                # Deduplicate (case-insensitive)
                val_lower = str_value.lower()
                if val_lower in seen:
                    continue
                
                seen.add(val_lower)
                
                # Validate specific entity types
                if key == 'players':
                    # Try to map to canonical player name
                    canonical_name = self._map_to_canonical_player(str_value)
                    if canonical_name:
                        cleaned_values.append(canonical_name)
                    elif len(str_value.split()) >= 2:  # Likely a player name
                        cleaned_values.append(str_value)
                
                elif key == 'teams':
                    # Map team aliases to canonical names
                    canonical_team = self._map_to_canonical_team(str_value)
                    if canonical_team:
                        cleaned_values.append(canonical_team)
                
                elif key == 'positions':
                    # Standardize position values
                    pos = str_value.upper()
                    if pos in ['GK', 'DEF', 'MID', 'FWD']:
                        cleaned_values.append(pos)
                
                elif key == 'metrics':
                    # Standardize metric names
                    metric = self._map_to_canonical_metric(str_value)
                    if metric:
                        cleaned_values.append(metric)
                
                elif key == 'gameweeks':
                    # Extract numbers only
                    match = re.search(r'(\d+)', str_value)
                    if match:
                        cleaned_values.append(match.group(1))
                
                elif key == 'numerical_values':
                    try:
                        num = float(str_value)
                        cleaned_values.append(num)
                    except ValueError:
                        pass
                
                else:
                    # Keep other values as-is
                    cleaned_values.append(str_value)
            
            cleaned[key] = cleaned_values
        
        return cleaned
    
    def _map_to_canonical_player(self, player_name: str) -> Optional[str]:
        """Map player name or alias to canonical name"""
        name_lower = player_name.lower()
        
        # Check exact variants first
        if name_lower in self.player_variants:
            return self.player_variants[name_lower]
        
        # Check partial matches
        for canonical, variants in self.known_players.items():
            if name_lower in canonical.lower() or any(variant in name_lower for variant in variants):
                return canonical
        
        # Check if name contains known player last names
        words = name_lower.split()
        for word in words:
            for canonical, variants in self.known_players.items():
                if any(word == v for v in variants):
                    return canonical
        
        return None
    
    def _map_to_canonical_team(self, team_name: str) -> Optional[str]:
        """Map team alias to canonical name"""
        team_lower = team_name.lower()
        for canonical, variants in self.teams.items():
            if team_lower == canonical.lower() or team_lower in variants:
                return canonical
        return None
    
    def _map_to_canonical_metric(self, metric_name: str) -> Optional[str]:
        """Map metric name to canonical form"""
        metric_lower = metric_name.lower()
        for canonical, variants in self.metrics.items():
            if metric_lower == canonical or metric_lower in variants:
                return canonical
        return None
    
    def _extract_entities_rule_based(self, text: str) -> Dict[str, List[str]]:
        """Enhanced rule-based entity extraction with multi-entity support"""
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
        
        # Extract ALL metrics mentioned
        for canonical_metric, variants in self.metrics.items():
            for variant in variants:
                # Use word boundary matching for better accuracy
                pattern = r'\b' + re.escape(variant) + r'\b'
                if re.search(pattern, text_lower):
                    if canonical_metric not in entities['metrics']:
                        entities['metrics'].append(canonical_metric)
                    break
        
        # Extract ALL known players (improved detection)
        for canonical_name, variants in self.known_players.items():
            for variant in variants:
                # Use word boundary and case-insensitive matching
                pattern = r'\b' + re.escape(variant) + r'\b'
                if re.search(pattern, text_lower, re.IGNORECASE):
                    if canonical_name not in entities['players']:
                        entities['players'].append(canonical_name)
                    break
        
        # Extract capitalized names (for players not in database)
        if 'vs' in text_lower or 'versus' in text_lower or 'compare' in text_lower or 'and' in text:
            potential_players = self._extract_capitalized_names(text)
            for player in potential_players:
                if player not in entities['players']:
                    entities['players'].append(player)
        
        # Extract ALL positions
        for canonical_pos, variants in self.positions.items():
            for variant in variants:
                pattern = r'\b' + re.escape(variant) + r'\b'
                if re.search(pattern, text_lower):
                    if canonical_pos not in entities['positions']:
                        entities['positions'].append(canonical_pos)
                    break
        
        # Extract ALL teams (with word boundary matching)
        for canonical_team, variants in self.teams.items():
            for variant in variants:
                pattern = r'\b' + re.escape(variant) + r'\b'
                if re.search(pattern, text_lower):
                    if canonical_team not in entities['teams']:
                        entities['teams'].append(canonical_team)
                    break
        
        # Extract ALL seasons
        season_patterns = [
            r'20\d{2}[-/]20?\d{2}',  # 2022-23 or 2022/23
            r'20\d{2}'                # 2022
        ]
        for pattern in season_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in entities['seasons']:
                    entities['seasons'].append(match)
        
        # Extract ALL gameweeks (improved patterns)
        gw_patterns = [
            r'(?:gameweek|gw|week)\s*(\d+)',
            r'round\s*(\d+)',
            r'(\d+)\s*(?:st|nd|rd|th)\s*(?:gameweek|gw|week)'
        ]
        for pattern in gw_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if str(match) not in entities['gameweeks']:
                    entities['gameweeks'].append(str(match))
        
        # Extract ALL comparators
        comparators = [
            'more than', 'less than', 'greater than', 'fewer than',
            'under', 'over', 'above', 'below', 'at least', 'at most',
            'higher than', 'lower than'
        ]
        for comp in comparators:
            if comp in text_lower:
                if comp not in entities['comparators']:
                    entities['comparators'].append(comp)
        
        # Extract ALL numerical values (excluding seasons and gameweeks)
        number_pattern = r'\b(\d+(?:\.\d+)?)\b'
        numbers = re.findall(number_pattern, text)
        seen_numbers = set()
        
        for num in numbers:
            # Skip if already captured as season or gameweek
            is_season = any(season_pattern in num for season_pattern in ['202', '201', '200'])
            is_gameweek = num in entities['gameweeks']
            
            if not is_season and not is_gameweek:
                if num not in seen_numbers:
                    try:
                        entities['numerical_values'].append(float(num))
                        seen_numbers.add(num)
                    except ValueError:
                        pass
        
        return entities
    
    def _extract_capitalized_names(self, text: str) -> List[str]:
        """Extract potential player names from capitalized words with improved logic"""
        stopwords = {
            'who', 'what', 'find', 'show', 'tell', 'compare', 'the', 'vs', 'v', 'and', 'or',
            'in', 'for', 'of', 'did', 'are', 'have', 'has', 'is', 'how', 'which', 'where',
            'when', 'why', 'do', 'does', 'received', 'earned', 'had', 'players', 'playing',
            'play', 'between', 'against', 'premier', 'league', 'fantasy', 'fpl', 'scored',
            'score', 'gameweek', 'season', 'team', 'teams', 'a', 'an', 'with', 'from', 'to',
            'by', 'their', 'this', 'that', 'these', 'those', 'my', 'your', 'our', 'his', 'her'
        }
        
        words = re.split(r'(\s+|,|;|vs|versus|and|or)', text)
        potential_players = []
        current_name = []
        
        for i, word in enumerate(words):
            clean_word = word.strip().rstrip("'s,?!.:;-")
            
            # Check if word is a capitalized name component
            if (len(clean_word) >= 2 and clean_word[0].isupper() and 
                clean_word.lower() not in stopwords and not clean_word.isdigit()):
                
                # Handle hyphenated names (e.g., Calvert-Lewin)
                if '-' in clean_word:
                    parts = clean_word.split('-')
                    if all(len(p) >= 2 for p in parts):
                        potential_players.append(clean_word)
                        current_name = []
                    continue
                
                current_name.append(clean_word)
                
                # Check if next word is also capitalized (for multi-word names)
                if i + 1 < len(words):
                    next_word = words[i + 1].strip().rstrip("'s,?!.:;-")
                    if (len(next_word) >= 2 and next_word[0].isupper() and 
                        next_word.lower() not in stopwords):
                        continue
                
                # Complete the current name
                if len(current_name) >= 1:
                    player_name = ' '.join(current_name)
                    if len(player_name) >= 3:  # Minimum name length
                        potential_players.append(player_name)
                    current_name = []
        
        # Remove duplicates while preserving order
        seen = set()
        unique_players = []
        for player in potential_players:
            if player.lower() not in seen:
                unique_players.append(player)
                seen.add(player.lower())
        
        return unique_players
    
    def _merge_entities(self, llm_entities: Dict, rule_entities: Dict) -> Dict:
        """Merge LLM and rule-based entities intelligently"""
        merged = {}
        
        for key in rule_entities.keys():
            llm_vals = llm_entities.get(key, [])
            rule_vals = rule_entities.get(key, [])
            
            # Combine and deduplicate
            combined = []
            seen = set()
            
            # Prioritize LLM-extracted players (often more accurate)
            if key == 'players':
                # Add LLM players first
                for val in llm_vals:
                    if isinstance(val, str):
                        val_lower = val.lower()
                        if val_lower not in seen:
                            combined.append(val)
                            seen.add(val_lower)
                
                # Add rule-based players that weren't caught by LLM
                for val in rule_vals:
                    if isinstance(val, str):
                        val_lower = val.lower()
                        if val_lower not in seen:
                            combined.append(val)
                            seen.add(val_lower)
            else:
                # For other entities, combine all
                all_vals = llm_vals + rule_vals
                for val in all_vals:
                    if isinstance(val, (str, int, float)):
                        val_str = str(val).lower()
                        if val_str not in seen:
                            combined.append(val)
                            seen.add(val_str)
            
            merged[key] = combined
        
        return merged
    
    def generate_embedding(self, user_input: str) -> np.ndarray:
        """Generate semantic text embedding vector"""
        try:
            embedding = self.embedder.encode([user_input], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            self.error_log.append({
                'type': 'embedding_error',
                'query': user_input,
                'error': str(e)
            })
            return np.zeros(self.embedder.get_sentence_embedding_dimension())
    
    def entities_to_criteria(self, entities: Dict, position: str = None) -> Dict[str, str]:
        """Convert extracted entities to search criteria"""
        criteria = {'total_points': 'high'}  # Default
        
        # Determine position
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
            criteria['saves'] = 'high'
        
        # Override with explicit metrics from query
        metrics = entities.get('metrics', [])
        
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
            'cards': 'cards',
            'saves': 'saves',
            'yellow_cards': 'cards',
            'red_cards': 'cards',
            'value': 'value',
            'ownership': 'ownership'
        }
        
        for metric in metrics:
            if metric in metric_to_criteria:
                criteria[metric_to_criteria[metric]] = 'high'
        
        return criteria
    
    def generate_numeric_embedding(self, criteria: Dict) -> np.ndarray:
        """Generate a 12-dimensional numeric query embedding"""
        embedding = np.full(12, 0.5, dtype=np.float32)
        
        feature_map = {
            'goals': 0, 
            'assists': 1, 
            'total_points': 2,
            'clean_sheets': 3, 
            'minutes': 4, 
            'bonus': 5,
            'form': 6, 
            'ict_index': 7, 
            'influence': 8, 
            'creativity': 9, 
            'threat': 10, 
            'value': 11
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
        """Complete preprocessing pipeline with enhanced entity extraction"""
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
        
        # Step 1: Classify intent
        intent, confidence = self.classify_intent(user_input)
        
        # Step 2: Extract ALL entities
        entities = self.extract_entities(user_input)
        
        # Step 3: Generate search criteria from entities
        search_criteria = self.entities_to_criteria(entities)
        
        result = {
            'original_query': user_input,
            'intent': intent,
            'intent_confidence': confidence,
            'entities': entities,
            'method': 'llm' if self.use_llm else 'rule-based',
            'search_criteria': search_criteria,
        }
        
        # Step 4: Generate embeddings if requested
        if include_embedding:
            result['text_embedding'] = self.generate_embedding(user_input)
            result['numeric_embedding'] = self.generate_numeric_embedding(search_criteria)
        else:
            result['text_embedding'] = None
            result['numeric_embedding'] = None
        
        result['embedding'] = result['text_embedding']
        
        return result
    
    def _convert_to_season_format(self, year_str: str) -> str:
        """Convert year to KG season format"""
        try:
            year = int(year_str)
            next_year = str(year + 1)[-2:]
            return f"{year}-{next_year}"
        except ValueError:
            return year_str
    
    def get_cypher_params(self, preprocessing_result: Dict) -> Dict[str, any]:
        """Convert entities to Cypher parameters with robust multi-entity support"""
        entities = preprocessing_result['entities']
        
        params = {
            'intent': preprocessing_result['intent'],
            'intent_confidence': preprocessing_result['intent_confidence']
        }
        
        # Handle multiple positions
        if entities.get('positions'):
            params['position'] = entities['positions'][0]
            if len(entities['positions']) > 1:
                params['positions'] = entities['positions']
        
        # Handle multiple teams
        if entities.get('teams'):
            params['team'] = entities['teams'][0]
            if len(entities['teams']) > 1:
                params['team2'] = entities['teams'][1]
                params['teams'] = entities['teams']
        
        # Handle multiple seasons
        if entities.get('seasons'):
            raw_season = entities['seasons'][0]
            params['season'] = self._convert_to_season_format(raw_season)
            if len(entities['seasons']) > 1:
                params['seasons'] = [self._convert_to_season_format(s) for s in entities['seasons']]
        
        # Handle multiple gameweeks
        if entities.get('gameweeks'):
            try:
                params['gameweek'] = int(entities['gameweeks'][0])
                if len(entities['gameweeks']) > 1:
                    params['gameweeks'] = [int(gw) for gw in entities['gameweeks']]
            except ValueError:
                pass
        
        # Handle multiple metrics
        if entities.get('metrics'):
            params['metric'] = entities['metrics'][0]
            if len(entities['metrics']) > 1:
                params['metric2'] = entities['metrics'][1]
                params['metrics'] = entities['metrics']
        
        # Handle multiple players (CRITICAL for comparisons)
        if entities.get('players'):
            params['player_name'] = entities['players'][0]
            if len(entities['players']) > 1:
                params['player_name2'] = entities['players'][1]
                params['player_names'] = entities['players']
        
        # Handle numerical values and comparators
        if entities.get('numerical_values'):
            params['threshold'] = entities['numerical_values'][0]
            if len(entities['numerical_values']) > 1:
                params['thresholds'] = entities['numerical_values']
        
        if entities.get('comparators'):
            params['comparator'] = entities['comparators'][0]
        
        return params
    
    def get_errors(self) -> List[Dict]:
        """Get error log for debugging"""
        return self.error_log
    
    def clear_errors(self):
        """Clear the error log"""
        self.error_log = []
