"""
Command Line Interface Module

Provides CLI entry point for RoleColorAI.
"""

import sys
import os
import json
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .pipeline import RoleColorPipeline
from .utils import load_resume_from_file, export_sentence_scores


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='RoleColorAI - Automated Resume Analysis & Rewriting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sample_resumes/builder_resume.txt
  %(prog)s sample_resumes/builder_resume.txt --verbose
  %(prog)s sample_resumes/builder_resume.txt -v -o custom_output.json
        """
    )
    
    parser.add_argument(
        'resume_file',
        nargs='?',
        help='Path to resume text file (optional, uses default sample if not provided)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed sentence scores and full analysis'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Custom output file name (default: {resume_name}.json)'
    )
    
    parser.add_argument(
        '--api-key',
        help='OpenAI API key (overrides OPENAI_API_KEY environment variable)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress initialization messages'
    )
    
    args = parser.parse_args()
    
    # Load resume
    if args.resume_file:
        if not os.path.exists(args.resume_file):
            print(f"❌ Error: File not found: {args.resume_file}")
            sys.exit(1)
        
        try:
            resume_text = load_resume_from_file(args.resume_file)
            output_name = args.output or os.path.splitext(os.path.basename(args.resume_file))[0]
        except Exception as e:
            print(f"❌ Error loading resume: {e}")
            sys.exit(1)
    else:
        # Default sample resume
        resume_text = """
    Senior Software Engineer
    
    Summary:
    Software engineer with 5 years of experience in backend development and APIs.
    
    Experience:
    
    Senior Software Engineer, TechCorp (2022-Present)
    - Designed and implemented microservices architecture serving 10 million daily users
    - Established technical standards and best practices across engineering organization
    - Built distributed systems from scratch using event-driven patterns and Kafka
    - Led strategic decisions on technology stack evolution and infrastructure roadmap
    - Reduced system latency by 60% through architectural improvements
    
    Software Engineer, StartupXYZ (2020-2022)
    - Developed RESTful APIs handling 1M+ requests per day
    - Collaborated with product and design teams to deliver new features
    - Maintained legacy systems ensuring 99.9% uptime
    - Implemented comprehensive testing and monitoring solutions
    
    Skills:
    Python, Java, AWS, Docker, Kubernetes, PostgreSQL, Redis, Kafka, Microservices
    """
        output_name = args.output or "sample_analysis"
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    
    # Initialize pipeline
    verbose_init = not args.quiet
    if api_key and verbose_init:
        print("ℹ Using OpenAI API for LLM generation\n")
    elif verbose_init:
        print("ℹ No OPENAI_API_KEY found. Using template-based generation (free).\n")
    
    pipeline = RoleColorPipeline(api_key=api_key, verbose=verbose_init)
    
    try:
        # Analyze resume
        result = pipeline.analyze_resume(resume_text)
        
        # Print results
        pipeline.print_results(result, verbose=args.verbose)
        
        # Save to JSON
        os.makedirs('output', exist_ok=True)
        output_file = f'output/{output_name}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Export sentence scores separately if verbose
        if args.verbose and 'sentence_scores' in result and result['sentence_scores']:
            export_sentence_scores(result, f'output/{output_name}_sentence_scores.json')
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
