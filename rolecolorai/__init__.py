"""
RoleColorAI - Automated Resume Analysis & Rewriting Package

A modular NLP system for classifying resumes into RoleColor categories
and generating role-aligned summaries.
"""

__version__ = "1.0.0"
__author__ = "RoleColorAI Team"

from .config import ROLE_DEFINITIONS, FEW_SHOT_EXAMPLES
from .scorer import SemanticRoleScorer
from .generator import SummaryGenerator
from .pipeline import RoleColorPipeline
from .utils import load_resume_from_file, export_sentence_scores

__all__ = [
    'ROLE_DEFINITIONS',
    'FEW_SHOT_EXAMPLES',
    'SemanticRoleScorer',
    'SummaryGenerator',
    'RoleColorPipeline',
    'load_resume_from_file',
    'export_sentence_scores',
]
