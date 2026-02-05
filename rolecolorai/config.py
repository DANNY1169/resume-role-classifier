"""
Configuration and constants for RoleColorAI

Contains role definitions and few-shot examples used throughout the system.
"""

ROLE_DEFINITIONS = {
    'Builder': """Creates innovative solutions and drives strategic vision. 
    Architects scalable systems and establishes technical direction. 
    Focuses on long-term product thinking and builds foundational infrastructure.""",
    
    'Enabler': """Facilitates collaboration across cross-functional teams. 
    Coordinates stakeholders and executes complex plans. 
    Bridges technical and business teams, enabling collective success.""",
    
    'Thriver': """Performs exceptionally under tight deadlines and high pressure. 
    Adapts rapidly to changing requirements and priorities. 
    Thrives in fast-paced, dynamic environments with uncertainty.""",
    
    'Supportee': """Ensures reliability and maintains critical systems. 
    Documents processes and establishes quality standards. 
    Provides consistent support and operational excellence."""
}

FEW_SHOT_EXAMPLES = {
    'Builder': [
        "Led design of microservices architecture serving 50M users. Established technical vision and standards. Built distributed systems from scratch.",
        "Architected ML pipeline reducing inference latency by 80%. Created framework adopted across 5 teams. Drove strategic tech stack decisions.",
        "Founded internal platform team. Designed service mesh infrastructure. Pioneered Kubernetes adoption across engineering org."
    ],
    'Enabler': [
        "Coordinated 3 cross-functional teams to deliver Q4 roadmap. Facilitated weekly syncs between engineering, product, design.",
        "Partnered with 5 stakeholder groups to align on API specs. Led sprint planning and retrospectives. Enabled seamless frontend/backend handoffs.",
        "Mentored 8 junior engineers on coding standards. Facilitated knowledge sharing. Bridged communication between remote and onsite teams."
    ],
    'Thriver': [
        "Shipped critical security patch in 48 hours. Adapted to changing requirements mid-sprint. Delivered MVP in 2 weeks with 4 pivots.",
        "Handled production incident affecting 10K users. Quickly diagnosed and deployed hotfix. Thrived during Black Friday 10x traffic spike.",
        "Joined failing project with 1 week to deadline. Rapidly onboarded, identified bottlenecks, shipped on time despite 3 pivots."
    ],
    'Supportee': [
        "Maintained payment processing system with 99.99% uptime. Documented runbooks for 50+ scenarios. Established monitoring best practices.",
        "Implemented comprehensive test suite (95% coverage). Conducted code reviews ensuring quality. Maintained legacy system 3 years, zero critical bugs.",
        "Created technical documentation for 20+ APIs. Standardized deployment procedures reducing errors 60%. Ensured security compliance."
    ]
}

# Model configuration
DEFAULT_MODEL = 'all-mpnet-base-v2'
DEFAULT_LLM_MODEL = 'gpt-4o-mini'

# Scoring configuration
ATTENTION_TOP_PERCENT = 0.3
ATTENTION_MID_PERCENT = 0.7
ATTENTION_TOP_WEIGHT = 2.0
ATTENTION_MID_WEIGHT = 1.0
ATTENTION_BOTTOM_WEIGHT = 0.5
SOFTMAX_TEMPERATURE = 1.2

# LLM configuration
LLM_MIN_CONFIDENCE = 0.3
LLM_MAX_TOKENS = 300
LLM_TEMPERATURE = 0.7
RESUME_EXTRACT_MAX_WORDS = 400
