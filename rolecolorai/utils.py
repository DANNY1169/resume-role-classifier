"""
Utility Functions Module

Helper functions for file I/O and data export.
"""

import json
import os
from typing import Dict, Optional


def load_resume_from_file(filepath: str) -> str:
    """
    Load resume text from a file.
    
    Args:
        filepath: Path to resume text file
        
    Returns:
        Resume text as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: For other file reading errors
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Resume file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading resume file: {e}")


def export_sentence_scores(result: Dict, output_file: Optional[str] = None) -> Optional[str]:
    """
    Export detailed sentence scores to a separate JSON file.
    
    Args:
        result: Analysis result dictionary containing sentence_scores
        output_file: Optional output file path (defaults to output/sentence_scores.json)
        
    Returns:
        Path to exported file, or None if export failed
    """
    if 'sentence_scores' not in result or not result['sentence_scores']:
        print("⚠ No sentence scores available to export")
        return None
    
    if output_file is None:
        output_file = 'output/sentence_scores.json'
    
    export_data = {
        'resume_analysis': {
            'dominant_role': result.get('dominant_role'),
            'confidence': result.get('confidence'),
            'rolecolor_scores': result.get('rolecolor_scores', {}),
            'total_sentences': len(result['sentence_scores']),
            'embedding_dimension': result.get('embedding_dim', 0)
        },
        'sentence_scores': result['sentence_scores']
    }
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else 'output', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Sentence scores exported to: {output_file}")
    return output_file
