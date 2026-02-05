"""
Pipeline Module

Main orchestration class for end-to-end resume analysis.
"""

from typing import Dict, Optional

from .scorer import SemanticRoleScorer
from .generator import SummaryGenerator


class RoleColorPipeline:
    """Complete end-to-end pipeline"""
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = True):
        """
        Initialize pipeline components.
        
        Args:
            api_key: OpenAI API key (optional)
            verbose: Whether to print initialization messages
        """
        self.verbose = verbose
        if self.verbose:
            print("Initializing RoleColorAI Pipeline...")
        self.scorer = SemanticRoleScorer(verbose=verbose)
        self.generator = SummaryGenerator(api_key, verbose=verbose)
        if self.verbose:
            print("✓ Pipeline ready\n")
    
    def analyze_resume(self, resume_text: str) -> Dict:
        """
        Analyze resume and generate RoleColor-aligned summary.
        
        Args:
            resume_text: Raw resume text to analyze
            
        Returns:
            Dictionary with complete analysis results
        """
        if self.verbose:
            print("=" * 60)
            print("ANALYZING RESUME")
            print("=" * 60)
        
        # Step 1: Score resume
        if self.verbose:
            print("\n[1/2] Scoring resume using semantic embeddings...")
        scoring_result = self.scorer.score_resume(resume_text)
        
        if 'error' in scoring_result:
            return {
                'error': scoring_result['error'],
                'scores': scoring_result['scores']
            }
        
        if self.verbose:
            print(f"✓ Scored {scoring_result['sentences_used']} sentences")
            print(f"✓ Dominant role: {scoring_result['dominant_role']} ({scoring_result['confidence']:.1%} confidence)")
        
        # Step 2: Generate summary
        if self.verbose:
            print("\n[2/2] Generating RoleColor-aligned summary...")
        
        # Extract original summary if exists
        original_summary = self._extract_original_summary(resume_text)
        
        generation_result = self.generator.generate_summary(
            resume_text,
            scoring_result['scores'],
            original_summary
        )
        
        if self.verbose:
            print(f"✓ Summary generated using {generation_result['method']}")
            if generation_result.get('tokens'):
                print(f"  Tokens: {generation_result['tokens']}, Cost: ${generation_result['cost']:.4f}")
        
        # Compile results
        result = {
            'rolecolor_scores': scoring_result['scores'],
            'dominant_role': scoring_result['dominant_role'],
            'confidence': scoring_result['confidence'],
            'rewritten_summary': generation_result['summary'],
            'original_summary': original_summary,
            'generation_method': generation_result['method'],
            'top_evidence': scoring_result['top_sentences'],
            'sentence_scores': scoring_result.get('sentence_scores', []),
            'embedding_dim': scoring_result.get('embedding_dim', 0),
            'metadata': {
                'total_sentences': scoring_result['total_sentences'],
                'generation_tokens': generation_result.get('tokens', 0),
                'generation_cost': generation_result.get('cost', 0.0)
            }
        }
        
        return result
    
    def _extract_original_summary(self, resume_text: str) -> str:
        """Extract original summary section if exists"""
        lines = resume_text.split('\n')
        summary_lines = []
        in_summary = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect summary section header (must be a header, not just contain the word)
            if any(line_lower == h or line_lower.startswith(h + ':') for h in ['summary', 'objective', 'profile', 'about']):
                in_summary = True
                continue
            
            # Detect end of summary - look for section headers, not just words
            if in_summary:
                # Common section headers that indicate end of summary
                section_headers = [
                    'experience', 'professional experience', 'work experience',
                    'education', 'skills', 'technical skills', 'work history',
                    'employment', 'projects', 'certifications', 'awards'
                ]
                
                # Check if this line is a section header
                is_section_header = False
                for header in section_headers:
                    if (line_lower == header or 
                        line_lower.startswith(header + ':') or
                        line_lower.startswith(header + ' ')):
                        is_section_header = True
                        break
                
                if is_section_header:
                    break
            
            if in_summary and line.strip():
                summary_lines.append(line.strip())
        
        return ' '.join(summary_lines) if summary_lines else "No summary found in original resume."
    
    def print_results(self, result: Dict, verbose: bool = False):
        """Pretty print analysis results"""
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        
        if 'error' in result:
            print(f"\n⚠ Error: {result['error']}")
            return
        
        # Scores
        print("\n📊 ROLECOLOR SCORE DISTRIBUTION:")
        print("-" * 60)
        for role, score in sorted(result['rolecolor_scores'].items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(score * 50)
            print(f"  {role:12s}: {score:5.1%} {bar}")
        
        print(f"\n🎯 DOMINANT ROLE: {result['dominant_role']} ({result['confidence']:.1%} confidence)")
        
        # Evidence
        print("\n📝 TOP EVIDENCE SENTENCES:")
        print("-" * 60)
        for i, sent in enumerate(result['top_evidence'], 1):
            print(f"  {i}. {sent[:100]}{'...' if len(sent) > 100 else ''}")
        
        # Sentence scores - always show summary, detailed in verbose
        if 'sentence_scores' in result and result['sentence_scores']:
            if verbose:
                # Detailed view
                print("\n🔍 DETAILED SENTENCE SCORES:")
                print("-" * 60)
                for sent_data in result['sentence_scores'][:15]:  # Show top 15 sentences
                    print(f"\n  Sentence #{sent_data['sentence_index'] + 1}:")
                    print(f"    Text: {sent_data['sentence'][:80]}{'...' if len(sent_data['sentence']) > 80 else ''}")
                    print(f"    Best Match: {sent_data['best_match_role']} ({sent_data['best_match_score']:.3f})")
                    print(f"    All Scores: ", end="")
                    score_str = ", ".join([f"{role}: {score:.3f}" for role, score in sorted(
                        sent_data['role_scores'].items(), key=lambda x: x[1], reverse=True
                    )])
                    print(score_str)
                
                if len(result['sentence_scores']) > 15:
                    print(f"\n    ... and {len(result['sentence_scores']) - 15} more sentences")
                
                # Embedding info
                if 'embedding_dim' in result and result['embedding_dim'] > 0:
                    print(f"\n  📐 Embedding Dimension: {result['embedding_dim']}")
            else:
                # Compact summary view (always shown)
                print("\n📊 SENTENCE SCORES (All Sentences):")
                print("-" * 60)
                print(f"{'#':<4} {'Sentence (truncated)':<50} {'Builder':<8} {'Enabler':<8} {'Thriver':<8} {'Supportee':<8} {'Best':<10}")
                print("-" * 60)
                
                for sent_data in result['sentence_scores']:
                    idx = sent_data['sentence_index'] + 1
                    sent_text = sent_data['sentence'][:47] + '...' if len(sent_data['sentence']) > 50 else sent_data['sentence']
                    scores = sent_data['role_scores']
                    best_role = sent_data['best_match_role']
                    best_score = sent_data['best_match_score']
                    
                    print(f"{idx:<4} {sent_text:<50} "
                          f"{scores.get('Builder', 0):<8.3f} "
                          f"{scores.get('Enabler', 0):<8.3f} "
                          f"{scores.get('Thriver', 0):<8.3f} "
                          f"{scores.get('Supportee', 0):<8.3f} "
                          f"{best_role} ({best_score:.3f})")
                
                # Summary stats
                role_counts = {}
                for sent_data in result['sentence_scores']:
                    role = sent_data['best_match_role']
                    role_counts[role] = role_counts.get(role, 0) + 1
                
                print("-" * 60)
                print(f"  Summary: ", end="")
                summary_parts = [f"{role}: {count} sentences" for role, count in sorted(role_counts.items(), key=lambda x: x[1], reverse=True)]
                print(", ".join(summary_parts))
                
                if 'embedding_dim' in result and result['embedding_dim'] > 0:
                    print(f"  📐 Embedding Dimension: {result['embedding_dim']}")
                
                print(f"\n  💡 Use --verbose or -v flag for detailed view with full sentence text")
        
        # Summaries
        print("\n📄 ORIGINAL SUMMARY:")
        print("-" * 60)
        print(f"  {result['original_summary'][:200]}{'...' if len(result['original_summary']) > 200 else ''}")
        
        print("\n✨ REWRITTEN SUMMARY:")
        print("-" * 60)
        # Print as a flowing paragraph, not bullet points
        summary_text = result['rewritten_summary'].strip()
        # Ensure it ends with a period
        if not summary_text.endswith('.'):
            summary_text += '.'
        print(f"  {summary_text}")
        
        # Metadata
        print("\n⚙️  METADATA:")
        print("-" * 60)
        print(f"  Generation method: {result['generation_method']}")
        print(f"  Sentences analyzed: {result['metadata']['total_sentences']}")
        if result['metadata']['generation_tokens'] > 0:
            print(f"  Tokens used: {result['metadata']['generation_tokens']}")
            print(f"  Cost: ${result['metadata']['generation_cost']:.4f}")
        
        print("\n" + "=" * 60)
