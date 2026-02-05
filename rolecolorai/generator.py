"""
Summary Generator Module

Provides LLM-powered and template-based summary generation
for RoleColor-aligned resume summaries.
"""

import json
import re
from typing import Dict, Optional

from .config import (
    ROLE_DEFINITIONS,
    DEFAULT_LLM_MODEL,
    LLM_MIN_CONFIDENCE,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    RESUME_EXTRACT_MAX_WORDS
)


class SummaryGenerator:
    """
    LLM-powered summary generation with template fallback.
    
    Uses OpenAI GPT for cost-efficiency and speed.
    Falls back to templates if API unavailable.
    """
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = True):
        """
        Initialize with optional API key.
        
        Args:
            api_key: OpenAI API key (optional)
            verbose: Whether to print initialization messages
        """
        self.api_key = api_key
        self.client = None
        self.verbose = verbose
        
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                if self.verbose:
                    print("✓ OpenAI client initialized")
            except ImportError:
                if self.verbose:
                    print("⚠ openai package not installed. Using template-only mode.")
                    print("  Install with: pip install openai")
        else:
            if self.verbose:
                print("ℹ No API key provided. Using template-only mode (free, instant).")
    
    def generate_summary(
        self, 
        resume_text: str, 
        role_scores: Dict[str, float],
        original_summary: str = ""
    ) -> Dict:
        """
        Generate RoleColor-aligned summary.
        
        Uses LLM if available, falls back to template otherwise.
        
        Args:
            resume_text: Full resume text
            role_scores: Dictionary of role scores
            original_summary: Original summary from resume (if any)
            
        Returns:
            Dictionary with summary, method, tokens, cost
        """
        dominant_role = max(role_scores.items(), key=lambda x: x[1])[0]
        confidence = role_scores[dominant_role]
        
        # Extract metadata
        metadata = self._extract_metadata(resume_text)
        
        # Try LLM generation if client available
        if self.client and confidence > LLM_MIN_CONFIDENCE:
            try:
                return self._llm_generation(
                    resume_text, dominant_role, role_scores, metadata, original_summary
                )
            except Exception as e:
                if self.verbose:
                    print(f"⚠ LLM generation failed: {e}. Falling back to template.")
        
        # Template fallback
        return self._template_generation(dominant_role, metadata, confidence)
    
    def _extract_metadata(self, resume_text: str) -> Dict:
        """Extract years, title, skills from resume"""
        metadata = {
            'years': 'experienced',
            'title': 'professional',
            'skills': []
        }
        
        # Extract years
        years_match = re.search(r'(\d+)[\+]?\s*years?', resume_text, re.IGNORECASE)
        if years_match:
            metadata['years'] = years_match.group(1)
        
        # Extract title (heuristic: first capitalized multi-word phrase)
        lines = resume_text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){1,3}$', line):
                metadata['title'] = line
                break
        
        # Extract common skills and capitalize properly
        common_skills = {
            'python', 'java', 'javascript', 'react', 'node', 'aws', 'docker',
            'kubernetes', 'sql', 'machine learning', 'data science', 'api',
            'go', 'postgresql', 'redis', 'kafka', 'graphql', 'microservices'
        }
        text_lower = resume_text.lower()
        found_skills = []
        for skill in common_skills:
            if skill in text_lower:
                # Capitalize properly: "python" -> "Python", "machine learning" -> "Machine Learning"
                formatted_skill = ' '.join(word.capitalize() for word in skill.split())
                found_skills.append(formatted_skill)
        metadata['skills'] = found_skills[:5]
        
        return metadata
    
    def _llm_generation(
        self,
        resume_text: str,
        dominant_role: str,
        role_scores: Dict[str, float],
        metadata: Dict,
        original_summary: str
    ) -> Dict:
        """Generate summary using OpenAI API"""
        
        # Extract key sections (token optimization)
        resume_extract = self._optimize_resume_extract(resume_text, max_words=RESUME_EXTRACT_MAX_WORDS)
        
        # Build prompt - requesting narrative style, not bullet points
        skills_text = ', '.join(metadata['skills'][:5]) if metadata['skills'] else "technical skills"
        role_definition = ROLE_DEFINITIONS.get(dominant_role, f"{dominant_role} role characteristics").strip()
        
        prompt = f"""You are an expert career coach and resume writer. Rewrite the following resume summary for a candidate identified as a "{dominant_role}".

ROLE DEFINITION:
{dominant_role}: {role_definition}

ORIGINAL SUMMARY:
{original_summary[:200] if original_summary else "No original summary provided."}

KEY EXPERIENCE:
{resume_extract}

CANDIDATE METADATA:
- Years of experience: {metadata['years']} years
- Professional title: {metadata['title']}
- Key skills: {skills_text}

Return a JSON object with this exact structure:
{{
  "summary": "A cohesive, flowing paragraph (4-6 sentences) that reads naturally, not like a list",
  "tone": "professional|strategic|dynamic|reliable"
}}

CRITICAL REQUIREMENTS:
1. Write in **FIRST PERSON** (use "I" or action verbs without pronouns) - this is a resume, the candidate wrote it themselves
2. Write a **cohesive, flowing paragraph** - NOT a list of bullet points or separate sentences
3. Use natural transitions between sentences (e.g., "Leveraging...", "Through...", "Additionally...")
4. Integrate skills naturally into sentences rather than listing them separately (e.g., "Building scalable APIs with Python and AWS" NOT "Skills: Python, AWS")
5. Use {dominant_role}-specific language from the role definition above
6. Include 1-2 quantified achievements if available in the experience
7. Professional, confident tone - avoid jargon and phrases like "This professional is recognized for..." (sounds like someone else wrote it)
8. Based ONLY on provided information (no hallucination)
9. The summary should read like the candidate wrote it themselves, not like a third-party description
10. Return valid JSON only"""

        try:
            response = self.client.chat.completions.create(
                model=DEFAULT_LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result_text = response.choices[0].message.content
            
            # Try to extract JSON
            try:
                result = json.loads(result_text)
                summary = result.get('summary', result_text)
            except json.JSONDecodeError:
                # Use raw text if not valid JSON
                summary = result_text.strip()
            
            # Validate length (4-6 sentences)
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            if len(sentences) < 4 or len(sentences) > 6:
                if self.verbose:
                    print(f"⚠ LLM output has {len(sentences)} sentences, expected 4-6. Using anyway.")
            
            # Calculate cost (GPT-4o-mini pricing: $0.15/$0.60 per 1M tokens)
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens / 1_000_000 * 0.15 + output_tokens / 1_000_000 * 0.60)
            
            return {
                'summary': summary,
                'method': 'llm',
                'model': DEFAULT_LLM_MODEL,
                'tokens': input_tokens + output_tokens,
                'cost': cost
            }
            
        except Exception as e:
            if self.verbose:
                print(f"⚠ LLM API call failed: {e}")
            raise
    
    def _template_generation(
        self, 
        dominant_role: str, 
        metadata: Dict,
        confidence: float
    ) -> Dict:
        """Generate summary using templates (free, instant)"""
        
        # Get skills text for natural integration
        skills_text = ', '.join(metadata['skills'][:3]) if metadata['skills'] else "modern technologies"
        years_text = metadata['years']
        # Fix awkward phrasing
        if years_text == 'experienced' or years_text == 'extensive':
            years_text = 'extensive'
            years_prefix = "with extensive experience"
        else:
            years_prefix = f"with {years_text} years of experience"
        title_text = metadata['title']
        
        # Predefined templates - written in first person, natural resume style
        if dominant_role == 'Builder':
            summary = (
                f"Experienced {title_text} {years_prefix} in architecting scalable systems "
                f"and driving technical vision. "
                f"Leverage a strong background in {skills_text} to transform abstract concepts into foundational "
                f"infrastructure that supports organizational growth. "
                f"Have a proven track record of designing long-term solutions and building frameworks that scale "
                f"with evolving business needs, consistently delivering innovative approaches to complex technical challenges."
            )
        elif dominant_role == 'Enabler':
            # Handle case where title might be generic
            title_display = title_text if title_text and title_text != 'professional' else "Professional"
            summary = (
                f"{title_display} {years_prefix} in cross-functional collaboration and "
                f"bridging gaps between technical and business stakeholders. "
                f"Use expertise in {skills_text} to coordinate complex initiatives across multiple teams and "
                f"unblock critical paths, delivering measurable results. "
                f"Facilitate seamless collaboration and enable high-performing teams through effective communication "
                f"and strategic execution."
            )
        elif dominant_role == 'Thriver':
            summary = (
                f"{title_text} {years_prefix} thriving in fast-paced, dynamic environments "
                f"where rapid adaptation is essential. "
                f"Leverage technical expertise in {skills_text} to rapidly iterate and ship high-quality solutions "
                f"under tight deadlines while maintaining delivery standards. "
                f"Deliver exceptional results even under pressure and uncertainty, consistently demonstrating "
                f"the ability to pivot quickly and adapt to changing requirements."
            )
        elif dominant_role == 'Supportee':
            summary = (
                f"{title_text} {years_prefix} focused on reliability and operational excellence "
                f"through rigorous processes and attention to detail. "
                f"Apply deep expertise in {skills_text} to maintain critical systems and ensure consistent quality "
                f"through comprehensive documentation and standardized procedures. "
                f"Committed to operational excellence with a proven track record of implementing standards that "
                f"reduce risk and improve system stability."
            )
        else:
            # Dynamic template generation for new roles - first person resume style
            role_definition = ROLE_DEFINITIONS.get(dominant_role, "")
            summary = (
                f"Experienced {title_text} {years_prefix}, demonstrating strong alignment "
                f"with {dominant_role} principles through professional work. "
                f"Leverage skills in {skills_text} to apply {dominant_role.lower()} approaches and deliver "
                f"consistent value. "
                f"Committed to excellence with a track record of meaningful contributions."
            )
        
        return {
            'summary': summary,
            'method': 'template',
            'model': 'rule-based',
            'tokens': 0,
            'cost': 0.0
        }
    
    def _optimize_resume_extract(self, resume_text: str, max_words: int = None) -> str:
        """Extract most relevant content from resume for LLM"""
        if max_words is None:
            max_words = RESUME_EXTRACT_MAX_WORDS
            
        # Simple extraction: first N words
        words = resume_text.split()
        extract = ' '.join(words[:max_words])
        
        # Try to end on sentence boundary
        last_period = extract.rfind('.')
        if last_period > len(extract) * 0.8:  # If close to end
            extract = extract[:last_period + 1]
        
        return extract
