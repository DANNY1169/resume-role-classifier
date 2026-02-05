"""
Semantic Role Scorer Module

Provides semantic embedding-based scoring for resume classification.
Uses SentenceTransformers for context-aware role matching.
"""

import numpy as np
import re
from typing import Dict, List, Tuple

from .config import (
    ROLE_DEFINITIONS,
    DEFAULT_MODEL,
    ATTENTION_TOP_PERCENT,
    ATTENTION_MID_PERCENT,
    ATTENTION_TOP_WEIGHT,
    ATTENTION_MID_WEIGHT,
    ATTENTION_BOTTOM_WEIGHT,
    SOFTMAX_TEMPERATURE
)


class SemanticRoleScorer:
    """
    Semantic embedding-based scorer - NO manual keywords needed!
    
    Uses SentenceTransformers to understand meaning, not just words.
    Handles paraphrasing, synonyms, and context automatically.
    """
    
    def __init__(self, model_name: str = None, verbose: bool = True):
        """
        Initialize with pre-trained semantic model.
        
        Args:
            model_name: Name of SentenceTransformer model to use
            verbose: Whether to print initialization messages
        """
        self.model_name = model_name or DEFAULT_MODEL
        self.verbose = verbose
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            if self.verbose:
                print(f"✓ Loaded semantic model: {self.model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        
        # Pre-compute role embeddings (one-time cost)
        self.role_embeddings = self._compute_role_embeddings()
        if self.verbose:
            print(f"✓ Computed embeddings for {len(self.role_embeddings)} roles")
    
    def _compute_role_embeddings(self) -> Dict[str, np.ndarray]:
        """Embed role definitions into semantic space"""
        embeddings = {}
        for role, definition in ROLE_DEFINITIONS.items():
            # Embed the definition
            embeddings[role] = self.model.encode(definition, convert_to_numpy=True)
        return embeddings
    
    def score_resume(self, resume_text: str) -> Dict:
        """
        Score resume using semantic similarity.
        
        Args:
            resume_text: Raw resume text to analyze
            
        Returns:
            Dictionary containing:
                - scores: Normalized role scores
                - dominant_role: Role with highest score
                - confidence: Confidence score (0-1)
                - top_sentences: Top evidence sentences
                - sentence_scores: Detailed per-sentence scores
                - embedding_dim: Dimension of embeddings used
        """
        # Extract sentences
        sentences = self._extract_sentences(resume_text)
        
        if len(sentences) < 5:
            return {
                'scores': {role: 0.25 for role in ROLE_DEFINITIONS},
                'dominant_role': 'Unknown',
                'confidence': 0.25,
                'top_sentences': [],
                'total_sentences': len(sentences),
                'error': 'Resume too short (<5 substantive sentences)'
            }
        
        # Embed all sentences at once (batched for speed)
        sentence_embeddings = self.model.encode(sentences, convert_to_numpy=True)
        
        # Calculate similarity to each role
        role_similarities = self._calculate_similarities(sentence_embeddings, sentences)
        
        # Aggregate with attention weighting
        role_scores = self._aggregate_scores(role_similarities)
        
        # Normalize with softmax
        normalized_scores = self._softmax_normalize(role_scores, temperature=SOFTMAX_TEMPERATURE)
        
        # Determine dominant role
        dominant_role = max(normalized_scores.items(), key=lambda x: x[1])[0]
        confidence = normalized_scores[dominant_role]
        
        # Get top evidence sentences
        top_sentences = self._get_top_sentences(role_similarities, dominant_role, sentences)
        
        # Create detailed sentence scores for logging
        sentence_scores = self._create_sentence_scores(role_similarities, sentences)
        
        return {
            'scores': normalized_scores,
            'dominant_role': dominant_role,
            'confidence': confidence,
            'top_sentences': top_sentences[:3],
            'total_sentences': len(sentences),
            'sentences_used': len(sentences),
            'sentence_scores': sentence_scores,
            'raw_similarities': role_similarities,
            'embedding_dim': sentence_embeddings.shape[1] if len(sentence_embeddings) > 0 else 0
        }
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract substantive sentences from resume"""
        lines = text.split('\n')
        sentences = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and headers
            if not line or len(line) < 5:
                continue
            
            # Remove bullet points and dashes
            line = re.sub(r'^[-•*]\s*', '', line)
            line = line.strip()
            
            # Split by sentence endings
            line_sentences = re.split(r'[.!?]+', line)
            for sent in line_sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                # Keep if >= 5 words (more lenient for bullet points)
                words = sent.split()
                if len(words) >= 5:
                    alpha_ratio = sum(c.isalpha() for c in sent) / max(len(sent), 1)
                    if alpha_ratio > 0.4:  # More lenient for technical content
                        # Remove common boilerplate
                        if not any(phrase in sent.lower() for phrase in [
                            'references available', 'upon request', 'email:', 'phone:',
                            'education:', 'skills:', 'experience:'
                        ]):
                            sentences.append(sent)
        
        return sentences
    
    def _calculate_similarities(
        self, 
        sentence_embeddings: np.ndarray, 
        sentences: List[str]
    ) -> Dict[str, List[Tuple[float, str]]]:
        """Calculate cosine similarity between sentences and roles"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        role_sims = {role: [] for role in ROLE_DEFINITIONS}
        
        for sent_emb, sent_text in zip(sentence_embeddings, sentences):
            for role, role_emb in self.role_embeddings.items():
                # Cosine similarity
                sim = cosine_similarity(
                    sent_emb.reshape(1, -1), 
                    role_emb.reshape(1, -1)
                )[0][0]
                role_sims[role].append((sim, sent_text))
        
        return role_sims
    
    def _aggregate_scores(self, role_similarities: Dict[str, List[Tuple[float, str]]]) -> Dict[str, float]:
        """Aggregate with attention weighting - top sentences matter more"""
        scores = {}
        
        for role, sims in role_similarities.items():
            # Sort by similarity
            sorted_sims = sorted([s[0] for s in sims], reverse=True)
            
            if not sorted_sims:
                scores[role] = 0.0
                continue
            
            # Attention weighting: top 30% get 2x, middle 40% get 1x, bottom 30% get 0.5x
            n = len(sorted_sims)
            top_30 = int(n * ATTENTION_TOP_PERCENT)
            mid_70 = int(n * ATTENTION_MID_PERCENT)
            
            top_score = np.mean(sorted_sims[:top_30]) if top_30 > 0 else 0
            mid_score = np.mean(sorted_sims[top_30:mid_70]) if mid_70 > top_30 else 0
            bottom_score = np.mean(sorted_sims[mid_70:]) if len(sorted_sims) > mid_70 else 0
            
            # Weighted combination
            total_weight = ATTENTION_TOP_WEIGHT + ATTENTION_MID_WEIGHT + ATTENTION_BOTTOM_WEIGHT
            scores[role] = (
                ATTENTION_TOP_WEIGHT * top_score + 
                ATTENTION_MID_WEIGHT * mid_score + 
                ATTENTION_BOTTOM_WEIGHT * bottom_score
            ) / total_weight
        
        return scores
    
    def _softmax_normalize(self, scores: Dict[str, float], temperature: float = None) -> Dict[str, float]:
        """Softmax normalization with temperature scaling"""
        if temperature is None:
            temperature = SOFTMAX_TEMPERATURE
            
        score_array = np.array(list(scores.values()))
        
        # Temperature scaling
        score_array = score_array / temperature
        
        # Softmax with numerical stability
        exp_scores = np.exp(score_array - np.max(score_array))
        softmax_scores = exp_scores / exp_scores.sum()
        
        return {role: float(score) for role, score in zip(scores.keys(), softmax_scores)}
    
    def _get_top_sentences(
        self, 
        role_similarities: Dict[str, List[Tuple[float, str]]], 
        role: str,
        sentences: List[str]
    ) -> List[str]:
        """Get sentences most similar to the role"""
        sims = role_similarities[role]
        # Sort by similarity, get top 3
        sorted_sims = sorted(sims, key=lambda x: x[0], reverse=True)
        return [sent for _, sent in sorted_sims[:3]]
    
    def _create_sentence_scores(
        self,
        role_similarities: Dict[str, List[Tuple[float, str]]],
        sentences: List[str]
    ) -> List[Dict]:
        """Create detailed score breakdown for each sentence"""
        sentence_scores = []
        
        # Create a mapping from sentence text to its scores
        for idx, sentence in enumerate(sentences):
            scores_for_sentence = {}
            for role in ROLE_DEFINITIONS.keys():
                # Find this sentence's similarity score for this role
                sims = role_similarities[role]
                if idx < len(sims):
                    similarity, _ = sims[idx]
                    scores_for_sentence[role] = float(similarity)
            
            # Find which role this sentence matches best
            best_role = max(scores_for_sentence.items(), key=lambda x: x[1])[0] if scores_for_sentence else 'Unknown'
            best_score = scores_for_sentence.get(best_role, 0.0)
            
            sentence_scores.append({
                'sentence': sentence,
                'sentence_index': idx,
                'role_scores': scores_for_sentence,
                'best_match_role': best_role,
                'best_match_score': float(best_score)
            })
        
        return sentence_scores
