"""
Generate 500 "Golden Rule" training examples
Creates synthetic training pairs: Vague Prompt -> Ideal Structured Prompt
"""

import json
import os
import sys
from typing import List, Dict
import random

sys.stdout.reconfigure(encoding='utf-8')

# Template for ideal prompts (the "rules" baked into data)
IDEAL_PROMPT_TEMPLATE = """Act as a {persona}.

Context: {context}

Task: {task}

Constraints:
{constraints}

Steps:
{steps}

Goal: {goal}"""

def generate_training_pair(category: str, vague_prompt: str) -> Dict:
    """
    Generate a training pair: Vague -> Ideal
    The "rules" are implicit in the ideal prompt structure
    """
    
    # Define personas, contexts, constraints, and steps based on category
    templates = {
        'code': {
            'persona': 'an expert software engineer',
            'context_template': 'You need to create robust, production-ready code that follows best practices and handles edge cases.',
            'constraints_template': '- Use proper error handling\n- Include type hints/annotations\n- Follow language-specific conventions\n- Add comprehensive documentation\n- Ensure code is testable and maintainable',
            'steps_template': '1. Analyze requirements and identify edge cases\n2. Design the solution architecture\n3. Implement with error handling\n4. Add documentation and type hints\n5. Test with various inputs\n6. Refactor for maintainability',
            'goal_template': 'Deliver production-ready code that is maintainable, well-documented, and follows industry best practices.'
        },
        'web': {
            'persona': 'a senior full-stack web developer',
            'context_template': 'You are building a modern, accessible, and performant web application that will serve real users.',
            'constraints_template': '- Use semantic HTML5\n- Implement responsive design (mobile-first)\n- Follow WCAG 2.1 AA accessibility standards\n- Optimize for SEO\n- Ensure cross-browser compatibility\n- Use modern CSS (Flexbox/Grid)\n- Implement progressive enhancement',
            'steps_template': '1. Plan the information architecture\n2. Design responsive layouts\n3. Implement semantic HTML structure\n4. Add CSS with mobile-first approach\n5. Implement JavaScript functionality\n6. Test accessibility and cross-browser compatibility\n7. Optimize performance and SEO',
            'goal_template': 'Create a production-ready web application that is accessible, performant, and works across all devices and browsers.'
        },
        'data': {
            'persona': 'a data scientist and analyst',
            'context_template': 'You need to extract meaningful insights from data to inform business decisions.',
            'constraints_template': '- Use appropriate statistical methods\n- Create clear visualizations\n- Document data sources and assumptions\n- Handle missing data appropriately\n- Validate findings\n- Present results clearly',
            'steps_template': '1. Understand the business question\n2. Explore and clean the data\n3. Perform exploratory data analysis (EDA)\n4. Apply statistical analysis\n5. Create visualizations\n6. Interpret results and identify patterns\n7. Document findings and recommendations',
            'goal_template': 'Provide actionable insights backed by rigorous data analysis and clear visualizations.'
        },
        'automation': {
            'persona': 'a DevOps and automation engineer',
            'context_template': 'You need to automate workflows to improve efficiency and reduce manual errors.',
            'constraints_template': '- Implement error handling and retry logic\n- Add comprehensive logging\n- Ensure idempotency\n- Handle edge cases gracefully\n- Include monitoring and alerts\n- Document the automation process',
            'steps_template': '1. Analyze the current manual process\n2. Identify automation opportunities\n3. Design the automated workflow\n4. Implement with error handling\n5. Add logging and monitoring\n6. Test with various scenarios\n7. Deploy and document',
            'goal_template': 'Create a reliable, monitored automation system that reduces manual work and improves efficiency.'
        },
        'explanation': {
            'persona': 'an expert educator and technical writer',
            'context_template': 'You need to explain complex concepts in a clear, accessible way that helps others understand.',
            'constraints_template': '- Use clear, simple language\n- Provide concrete examples\n- Include analogies where helpful\n- Structure information logically\n- Address common misconceptions\n- Use visual aids when applicable',
            'steps_template': '1. Identify the core concept to explain\n2. Break it down into fundamental components\n3. Use analogies and examples\n4. Address common questions and misconceptions\n5. Provide step-by-step breakdown\n6. Include practical applications\n7. Summarize key takeaways',
            'goal_template': 'Create a comprehensive explanation that makes complex concepts accessible and actionable.'
        },
        'general': {
            'persona': 'an expert problem solver',
            'context_template': 'You need to solve a problem systematically and deliver high-quality results.',
            'constraints_template': '- Define clear requirements\n- Consider edge cases\n- Ensure quality and reliability\n- Document the solution\n- Make it maintainable\n- Test thoroughly',
            'steps_template': '1. Understand the problem and requirements\n2. Research and plan the approach\n3. Design the solution\n4. Implement with quality checks\n5. Test and validate\n6. Document and deliver',
            'goal_template': 'Deliver a high-quality, well-documented solution that meets all requirements.'
        }
    }
    
    # Get template for category
    template = templates.get(category, templates['general'])
    
    # Generate ideal prompt using template
    ideal_prompt = IDEAL_PROMPT_TEMPLATE.format(
        persona=template['persona'],
        context=template['context_template'],
        task=vague_prompt,
        constraints=template['constraints_template'],
        steps=template['steps_template'],
        goal=template['goal_template']
    )
    
    return {
        'input': vague_prompt,
        'target': ideal_prompt,
        'category': category
    }

def generate_vague_prompts() -> List[Dict]:
    """Generate 500 vague prompts across different categories"""
    
    vague_prompts = {
        'code': [
            'create function', 'fix code', 'write script', 'make program', 'build app',
            'add feature', 'debug error', 'optimize code', 'refactor code', 'test function',
            'implement api', 'create class', 'write test', 'fix bug', 'improve performance',
            'add validation', 'handle errors', 'parse data', 'format output', 'process file',
            'connect database', 'send email', 'fetch data', 'validate input', 'encrypt data',
            'compress file', 'parse json', 'generate report', 'log events', 'cache data',
            'authenticate user', 'authorize access', 'process payment', 'generate token', 'hash password',
            'serialize data', 'deserialize data', 'transform data', 'filter results', 'sort array',
            'search data', 'update record', 'delete record', 'create table', 'query database',
            'export data', 'import data', 'merge files', 'split string', 'format date',
            'calculate sum', 'find maximum', 'get average', 'count items', 'group data'
        ],
        'web': [
            'build website', 'create page', 'make form', 'add button', 'style page',
            'create layout', 'add navigation', 'make responsive', 'add animation', 'create modal',
            'build dashboard', 'create login', 'add search', 'make gallery', 'build blog',
            'create shop', 'add cart', 'make checkout', 'build profile', 'create settings',
            'add comments', 'make rating', 'create feed', 'build chat', 'add notifications',
            'create admin', 'build api', 'add auth', 'make secure', 'optimize speed',
            'add seo', 'make accessible', 'create sitemap', 'add analytics', 'build landing',
            'create portfolio', 'make resume', 'build forum', 'add voting', 'create wiki',
            'make calendar', 'build scheduler', 'create map', 'add location', 'build weather',
            'create news', 'make reader', 'build editor', 'add upload', 'create download'
        ],
        'data': [
            'analyze data', 'process data', 'clean data', 'transform data', 'visualize data',
            'find patterns', 'detect outliers', 'calculate stats', 'create report', 'predict trend',
            'classify data', 'cluster data', 'regress data', 'correlate variables', 'test hypothesis',
            'extract features', 'select features', 'normalize data', 'scale data', 'encode data',
            'split dataset', 'train model', 'validate model', 'test model', 'evaluate performance',
            'compare models', 'tune parameters', 'optimize model', 'deploy model', 'monitor model',
            'analyze sales', 'forecast demand', 'segment customers', 'detect fraud', 'recommend items',
            'analyze sentiment', 'extract text', 'summarize text', 'translate text', 'classify text',
            'process image', 'detect objects', 'recognize faces', 'analyze video', 'transcribe audio',
            'analyze network', 'detect anomalies', 'monitor system', 'predict failure', 'optimize resource'
        ],
        'automation': [
            'automate task', 'schedule job', 'run script', 'process batch', 'sync data',
            'backup files', 'deploy app', 'update system', 'monitor service', 'restart service',
            'send notification', 'generate report', 'process queue', 'handle errors', 'retry failed',
            'clean logs', 'archive data', 'rotate files', 'compress backups', 'verify integrity',
            'update database', 'sync files', 'process emails', 'scrape website', 'crawl pages',
            'extract data', 'transform data', 'load data', 'validate data', 'notify users',
            'process orders', 'update inventory', 'generate invoices', 'send receipts', 'track shipments',
            'monitor servers', 'scale resources', 'balance load', 'failover system', 'recover data',
            'test system', 'validate config', 'update dependencies', 'patch security', 'optimize performance'
        ],
        'explanation': [
            'explain concept', 'describe process', 'teach topic', 'clarify idea', 'define term',
            'show example', 'demonstrate method', 'illustrate point', 'break down problem', 'solve step by step',
            'explain algorithm', 'describe architecture', 'teach language', 'clarify syntax', 'show pattern',
            'explain design', 'describe system', 'teach framework', 'clarify concept', 'show best practice',
            'explain theory', 'describe model', 'teach technique', 'clarify approach', 'show solution',
            'explain method', 'describe tool', 'teach skill', 'clarify usage', 'show workflow',
            'explain principle', 'describe pattern', 'teach concept', 'clarify difference', 'show comparison',
            'explain benefit', 'describe advantage', 'teach strategy', 'clarify tradeoff', 'show alternative',
            'explain reason', 'describe cause', 'teach logic', 'clarify relationship', 'show connection',
            'explain result', 'describe outcome', 'teach lesson', 'clarify meaning', 'show implication',
            'explain purpose', 'describe goal', 'teach objective', 'clarify intent', 'show value'
        ],
        'general': [
            'solve problem', 'improve process', 'optimize workflow', 'enhance quality', 'increase efficiency',
            'reduce cost', 'save time', 'improve accuracy', 'enhance security', 'increase reliability',
            'streamline process', 'simplify system', 'modernize approach', 'upgrade technology', 'innovate solution',
            'create strategy', 'develop plan', 'design system', 'build solution', 'implement change',
            'manage project', 'coordinate team', 'organize work', 'track progress', 'measure success',
            'evaluate performance', 'assess quality', 'review process', 'analyze results', 'identify issues',
            'fix problem', 'resolve conflict', 'address concern', 'handle situation', 'manage risk',
            'plan ahead', 'prepare for change', 'anticipate needs', 'prevent issues', 'mitigate risk',
            'communicate clearly', 'document thoroughly', 'train effectively', 'support users', 'maintain system'
        ]
    }
    
    # Generate 500 pairs (distributed across categories)
    training_pairs = []
    total_needed = 500
    
    # Calculate distribution
    categories = list(vague_prompts.keys())
    per_category = total_needed // len(categories)
    remainder = total_needed % len(categories)
    
    for i, category in enumerate(categories):
        count = per_category + (1 if i < remainder else 0)
        prompts = vague_prompts[category]
        
        # Sample with replacement if needed
        for j in range(count):
            vague = random.choice(prompts)
            pair = generate_training_pair(category, vague)
            training_pairs.append(pair)
    
    return training_pairs

def main():
    """Generate and save 500 training examples"""
    print("Generating 500 'Golden Rule' training examples...")
    print("This embeds the 'rules' into the training data structure...")
    
    training_pairs = generate_vague_prompts()
    
    # Save to JSON
    output_path = './data/lora_training_data.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'training_pairs': training_pairs,
            'count': len(training_pairs),
            'description': '500 synthetic training pairs: Vague Prompt -> Ideal Structured Prompt'
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Generated {len(training_pairs)} training pairs")
    print(f"✓ Saved to {output_path}")
    print("\nSample pairs:")
    for i, pair in enumerate(training_pairs[:3]):
        print(f"\n{i+1}. Input: {pair['input']}")
        print(f"   Target (first 100 chars): {pair['target'][:100]}...")

if __name__ == '__main__':
    main()


