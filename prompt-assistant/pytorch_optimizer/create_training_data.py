"""
Create comprehensive training dataset for prompt optimization
Includes diverse examples from various domains and use cases
"""

import json
import os
from typing import List, Dict

def create_comprehensive_dataset(output_path: str):
    """
    Create a comprehensive training dataset with diverse prompt pairs
    """
    
    triplets = []
    
    # ========== CODE GENERATION ==========
    code_generation_pairs = [
        {
            'anchor': 'Create a function',
            'positive': 'Create a Python function that takes two integers as parameters and returns their sum. Include input validation, type checking, error handling for edge cases (negative numbers, zero), and a comprehensive docstring following Google style.',
            'negative': 'function'
        },
        {
            'anchor': 'Write a class',
            'positive': 'Write a Python class named `User` that represents a user account. Include: (1) constructor with username, email, and password parameters, (2) methods for authentication and password hashing, (3) property decorators for data validation, (4) __str__ and __repr__ methods, and (5) error handling for invalid inputs.',
            'negative': 'class'
        },
        {
            'anchor': 'Make an API endpoint',
            'positive': 'Create a RESTful API endpoint using Flask/FastAPI that handles POST requests to `/api/users`. The endpoint should: (1) validate JSON input (username, email, password), (2) check for duplicate emails, (3) hash passwords using bcrypt, (4) return appropriate HTTP status codes (201 for success, 400 for validation errors, 409 for conflicts), and (5) include proper error messages.',
            'negative': 'api'
        },
        {
            'anchor': 'Fix the bug',
            'positive': 'Debug and fix the null pointer exception occurring in the user authentication module at line 45. Add proper null checks, error handling, and logging. Ensure the fix maintains backward compatibility and doesn\'t break existing functionality. Include unit tests.',
            'negative': 'bug'
        },
        {
            'anchor': 'Optimize this code',
            'positive': 'Refactor and optimize the following code for better performance: (1) analyze time complexity and identify bottlenecks, (2) reduce redundant operations and database queries, (3) improve memory usage with generators where appropriate, (4) maintain code readability and add comments, (5) provide before/after performance comparison.',
            'negative': 'optimize'
        },
        {
            'anchor': 'Add error handling',
            'positive': 'Add comprehensive error handling to the function: (1) validate all input parameters, (2) handle specific exception types (ValueError, TypeError, KeyError), (3) log errors with appropriate severity levels, (4) return meaningful error messages to users, (5) implement retry logic for transient failures.',
            'negative': 'errors'
        },
        {
            'anchor': 'Create a database schema',
            'positive': 'Design a normalized database schema for an e-commerce system. Include tables for: users, products, orders, order_items, and payments. Specify primary keys, foreign keys, indexes, constraints (NOT NULL, UNIQUE, CHECK), and relationships. Include ER diagram notation.',
            'negative': 'database'
        },
        {
            'anchor': 'Write tests',
            'positive': 'Write comprehensive unit tests for the UserService class. Include: (1) test cases for all public methods, (2) edge cases and boundary conditions, (3) mock external dependencies (database, API calls), (4) test error handling scenarios, (5) achieve at least 80% code coverage. Use pytest framework.',
            'negative': 'test'
        }
    ]
    
    # ========== WEB DEVELOPMENT ==========
    web_dev_pairs = [
        {
            'anchor': 'Create a website',
            'positive': 'Create a modern, responsive website using HTML5, CSS3 (flexbox/grid), and vanilla JavaScript. The website should: (1) be mobile-friendly with breakpoints for tablet and desktop, (2) follow WCAG 2.1 AA accessibility standards, (3) implement SEO best practices (meta tags, semantic HTML, structured data), (4) have fast load times (<3 seconds), and (5) work across all major browsers.',
            'negative': 'website'
        },
        {
            'anchor': 'Build a form',
            'positive': 'Build an accessible HTML form with client-side validation. Include: (1) proper label associations and ARIA attributes, (2) real-time validation feedback, (3) error messages that are screen-reader friendly, (4) support for keyboard navigation, (5) submission handling with loading states and success/error messages.',
            'negative': 'form'
        },
        {
            'anchor': 'Make it responsive',
            'positive': 'Make the website fully responsive using CSS media queries. Implement: (1) mobile-first approach with breakpoints at 768px (tablet) and 1024px (desktop), (2) flexible grid layouts using CSS Grid or Flexbox, (3) responsive images with srcset, (4) touch-friendly button sizes (min 44x44px), and (5) test on actual devices.',
            'negative': 'responsive'
        },
        {
            'anchor': 'Add animations',
            'positive': 'Add smooth CSS animations and transitions to improve user experience. Include: (1) page load animations, (2) hover effects on interactive elements, (3) scroll-triggered animations using Intersection Observer API, (4) loading spinners for async operations, (5) ensure animations respect prefers-reduced-motion media query for accessibility.',
            'negative': 'animate'
        }
    ]
    
    # ========== DATA ANALYSIS & ML ==========
    data_ml_pairs = [
        {
            'anchor': 'Analyze this data',
            'positive': 'Perform comprehensive data analysis on the provided dataset. Include: (1) exploratory data analysis (EDA) with summary statistics and visualizations, (2) identify missing values, outliers, and data quality issues, (3) feature engineering and correlation analysis, (4) statistical tests for significance, (5) provide actionable insights and recommendations.',
            'negative': 'analyze'
        },
        {
            'anchor': 'Train a model',
            'positive': 'Train a machine learning model for classification. Include: (1) data preprocessing (scaling, encoding, handling missing values), (2) feature selection and engineering, (3) train/test split with proper stratification, (4) hyperparameter tuning using cross-validation, (5) evaluate with multiple metrics (accuracy, precision, recall, F1, ROC-AUC) and provide confusion matrix.',
            'negative': 'model'
        },
        {
            'anchor': 'Visualize the data',
            'positive': 'Create comprehensive data visualizations using matplotlib/seaborn. Include: (1) distribution plots for numerical features, (2) correlation heatmap, (3) time series plots if applicable, (4) categorical comparisons with bar charts, (5) interactive plots using plotly if needed. Ensure all plots have proper labels, titles, and legends.',
            'negative': 'plot'
        },
        {
            'anchor': 'Clean the dataset',
            'positive': 'Clean and preprocess the dataset: (1) handle missing values (imputation or removal based on analysis), (2) remove duplicates and outliers using IQR method, (3) standardize data types and formats, (4) encode categorical variables appropriately, (5) create a data quality report documenting all changes.',
            'negative': 'clean'
        }
    ]
    
    # ========== EXPLANATION & DOCUMENTATION ==========
    explanation_pairs = [
        {
            'anchor': 'Explain how it works',
            'positive': 'Explain how the authentication system works in detail. Include: (1) overview of the authentication flow, (2) token generation process (JWT structure, signing, expiration), (3) validation steps and security checks, (4) refresh token mechanism, (5) error handling and edge cases. Provide code examples and diagrams where helpful.',
            'negative': 'explain'
        },
        {
            'anchor': 'Document this code',
            'positive': 'Write comprehensive documentation for this codebase. Include: (1) README with setup instructions and dependencies, (2) API documentation with endpoint descriptions, request/response formats, and examples, (3) code comments explaining complex logic, (4) architecture diagrams, (5) troubleshooting guide and FAQ.',
            'negative': 'document'
        },
        {
            'anchor': 'What does this do?',
            'positive': 'Provide a detailed explanation of what this code does. Break down: (1) the main purpose and functionality, (2) input parameters and their types, (3) return values and data structures, (4) side effects or state changes, (5) usage examples with different scenarios.',
            'negative': 'what'
        },
        {
            'anchor': 'How do I use this?',
            'positive': 'Provide a step-by-step guide on how to use this tool/library. Include: (1) installation instructions, (2) basic usage examples, (3) configuration options, (4) common use cases and patterns, (5) troubleshooting tips and known issues.',
            'negative': 'how'
        }
    ]
    
    # ========== AUTOMATION & WORKFLOWS ==========
    automation_pairs = [
        {
            'anchor': 'Automate this process',
            'positive': 'Automate the data processing workflow. Create a script that: (1) reads data from source (CSV/API/database), (2) applies transformations and validations, (3) handles errors gracefully with retry logic, (4) logs all operations, (5) sends notifications on completion or failure. Include configuration file for easy customization.',
            'negative': 'automate'
        },
        {
            'anchor': 'Create a script',
            'positive': 'Create a Python script that processes files in a directory. The script should: (1) accept command-line arguments for input/output paths, (2) validate file formats before processing, (3) process files in parallel for efficiency, (4) provide progress indicators, (5) generate a summary report of processed files.',
            'negative': 'script'
        },
        {
            'anchor': 'Set up CI/CD',
            'positive': 'Set up a CI/CD pipeline using GitHub Actions. Configure: (1) automated testing on pull requests, (2) code quality checks (linting, formatting), (3) build and deployment stages, (4) environment-specific deployments (dev, staging, prod), (5) rollback procedures and notifications.',
            'negative': 'ci/cd'
        }
    ]
    
    # ========== GENERAL IMPROVEMENTS ==========
    general_pairs = [
        {
            'anchor': 'Make it better',
            'positive': 'Improve the code quality and maintainability. Focus on: (1) refactoring for better structure and separation of concerns, (2) improving naming conventions and code readability, (3) adding comprehensive error handling, (4) writing unit tests, (5) updating documentation. Provide before/after comparison.',
            'negative': 'better'
        },
        {
            'anchor': 'Add security',
            'positive': 'Enhance security measures in the application. Implement: (1) input validation and sanitization, (2) authentication and authorization checks, (3) protection against common vulnerabilities (SQL injection, XSS, CSRF), (4) secure password handling and session management, (5) security headers and HTTPS enforcement.',
            'negative': 'security'
        },
        {
            'anchor': 'Improve performance',
            'positive': 'Optimize application performance. Analyze and improve: (1) database query efficiency (indexes, query optimization), (2) caching strategies (Redis, in-memory), (3) code execution time (algorithm complexity, bottlenecks), (4) resource usage (memory, CPU), (5) frontend performance (bundle size, lazy loading, CDN).',
            'negative': 'performance'
        }
    ]
    
    # Combine all pairs
    all_pairs = (
        code_generation_pairs +
        web_dev_pairs +
        data_ml_pairs +
        explanation_pairs +
        automation_pairs +
        general_pairs
    )
    
    # Create triplets with metadata
    for idx, pair in enumerate(all_pairs):
        # Determine category
        if idx < len(code_generation_pairs):
            category = 'code_generation'
        elif idx < len(code_generation_pairs) + len(web_dev_pairs):
            category = 'web_development'
        elif idx < len(code_generation_pairs) + len(web_dev_pairs) + len(data_ml_pairs):
            category = 'data_ml'
        elif idx < len(code_generation_pairs) + len(web_dev_pairs) + len(data_ml_pairs) + len(explanation_pairs):
            category = 'explanation'
        elif idx < len(code_generation_pairs) + len(web_dev_pairs) + len(data_ml_pairs) + len(explanation_pairs) + len(automation_pairs):
            category = 'automation'
        else:
            category = 'general'
        
        triplets.append({
            'anchor': pair['anchor'],
            'positive': pair['positive'],
            'negative': pair['negative'],
            'metadata': {
                'sample_id': idx,
                'category': category,
                'anchor_length': len(pair['anchor']),
                'positive_length': len(pair['positive']),
                'improvement_ratio': len(pair['positive']) / max(len(pair['anchor']), 1)
            }
        })
    
    # Create variations by adding slight modifications
    variations = []
    for triplet in triplets:
        # Add variations with different phrasings
        variations.append({
            'anchor': triplet['anchor'].capitalize(),
            'positive': triplet['positive'],
            'negative': triplet['negative'].lower(),
            'metadata': {**triplet['metadata'], 'variation': 'capitalization'}
        })
        
        # Add question form variations
        if not triplet['anchor'].endswith('?'):
            variations.append({
                'anchor': f"How to {triplet['anchor'].lower()}?",
                'positive': triplet['positive'],
                'negative': triplet['negative'],
                'metadata': {**triplet['metadata'], 'variation': 'question_form'}
            })
    
    # Combine original and variations
    all_triplets = triplets + variations
    
    # Create dataset structure
    dataset = {
        'triplets': all_triplets,
        'metadata': {
            'total_samples': len(all_triplets),
            'categories': {
                'code_generation': len(code_generation_pairs),
                'web_development': len(web_dev_pairs),
                'data_ml': len(data_ml_pairs),
                'explanation': len(explanation_pairs),
                'automation': len(automation_pairs),
                'general': len(general_pairs)
            },
            'description': 'Comprehensive prompt optimization training dataset with diverse examples',
            'version': '1.0'
        }
    }
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Created training dataset with {len(all_triplets)} triplets")
    print(f"   Saved to: {output_path}")
    print(f"   Categories: {dataset['metadata']['categories']}")
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create comprehensive training dataset')
    parser.add_argument('--output', type=str, default='./data/training_data.json',
                       help='Output path for training data JSON file')
    
    args = parser.parse_args()
    create_comprehensive_dataset(args.output)

