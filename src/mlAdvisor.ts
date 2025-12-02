import { PromptAnalyzer, PromptAnalysis } from './promptAnalyzer';

export interface MLRecommendations {
    goal: string;
    suggestedModel: ModelRecommendation;
    temperature: number;
    samplingParams: SamplingParams;
    requiredContext: ContextRequirement[];
    suggestedFormat: FormatRecommendation;
    missingElements: string[];
    filledPrompt: string;
    iterationStrategy: IterationStrategy;
    verificationChecklist: string[];
    confidence: number;
}

export interface ModelRecommendation {
    model: string;
    reason: string;
    alternatives: string[];
    costEstimate: string;
    speedEstimate: string;
}

export interface SamplingParams {
    temperature: number;
    topP?: number;
    topK?: number;
    frequencyPenalty?: number;
    presencePenalty?: number;
}

export interface ContextRequirement {
    type: 'code' | 'error' | 'data' | 'documentation' | 'examples' | 'environment' | 'constraints';
    description: string;
    priority: 'high' | 'medium' | 'low';
    example: string;
}

export interface FormatRecommendation {
    format: 'code' | 'json' | 'markdown' | 'table' | 'list' | 'structured' | 'natural';
    structure: string;
    reason: string;
    example: string;
}

export interface IterationStrategy {
    steps: string[];
    expectedIterations: number;
    refinementPoints: string[];
}

export class MLAdvisor {
    private analyzer: PromptAnalyzer;
    private modelDatabase: Map<string, ModelInfo>;
    private patternMatcher: PatternMatcher;

    constructor() {
        this.analyzer = new PromptAnalyzer();
        this.modelDatabase = this.initializeModelDatabase();
        this.patternMatcher = new PatternMatcher();
    }

    analyzeAndRecommend(prompt: string, availableModels?: string[]): MLRecommendations {
        // Analyze the prompt
        const analysis = this.analyzer.analyze(prompt);
        
        // Determine goal
        const goal = this.determineGoal(prompt, analysis);
        
        // Recommend model
        const suggestedModel = this.recommendModel(prompt, goal, availableModels);
        
        // Suggest parameters
        const temperature = this.recommendTemperature(prompt, goal);
        const samplingParams = this.recommendSamplingParams(prompt, goal, temperature);
        
        // Identify required context
        const requiredContext = this.identifyRequiredContext(prompt, goal, analysis);
        
        // Recommend format
        const suggestedFormat = this.recommendFormat(prompt, goal);
        
        // Find missing elements
        const missingElements = this.findMissingElements(prompt, analysis, requiredContext);
        
        // Fill in missing parts
        const filledPrompt = this.fillMissingParts(prompt, goal, missingElements, suggestedFormat);
        
        // Create iteration strategy
        const iterationStrategy = this.createIterationStrategy(prompt, goal, analysis.score);
        
        // Generate verification checklist
        const verificationChecklist = this.generateVerificationChecklist(prompt, goal, suggestedFormat);
        
        // Calculate confidence
        const confidence = this.calculateConfidence(analysis, missingElements.length, requiredContext.length);
        
        return {
            goal,
            suggestedModel,
            temperature,
            samplingParams,
            requiredContext,
            suggestedFormat,
            missingElements,
            filledPrompt,
            iterationStrategy,
            verificationChecklist,
            confidence
        };
    }

    private determineGoal(prompt: string, analysis: PromptAnalysis): string {
        const lower = prompt.toLowerCase();
        
        // Pattern matching for goal detection
        if (this.patternMatcher.matches(lower, ['create', 'write', 'generate', 'build', 'implement', 'make'])) {
            if (this.patternMatcher.matches(lower, ['function', 'class', 'code', 'script', 'program'])) {
                return 'code_generation';
            } else if (this.patternMatcher.matches(lower, ['test', 'unit test', 'integration'])) {
                return 'test_generation';
            } else {
                return 'content_generation';
            }
        }
        
        if (this.patternMatcher.matches(lower, ['fix', 'debug', 'error', 'bug', 'issue', 'problem'])) {
            return 'debugging';
        }
        
        if (this.patternMatcher.matches(lower, ['review', 'analyze', 'check', 'examine', 'evaluate'])) {
            if (this.patternMatcher.matches(lower, ['performance', 'speed', 'optimize'])) {
                return 'performance_review';
            } else if (this.patternMatcher.matches(lower, ['security', 'vulnerability', 'safe'])) {
                return 'security_review';
            } else {
                return 'code_review';
            }
        }
        
        if (this.patternMatcher.matches(lower, ['explain', 'teach', 'learn', 'understand', 'how', 'what'])) {
            return 'explanation';
        }
        
        if (this.patternMatcher.matches(lower, ['refactor', 'improve', 'optimize', 'enhance'])) {
            return 'refactoring';
        }
        
        if (this.patternMatcher.matches(lower, ['translate', 'convert', 'transform'])) {
            return 'transformation';
        }
        
        if (this.patternMatcher.matches(lower, ['summarize', 'extract', 'find', 'search'])) {
            return 'information_extraction';
        }
        
        // Default
        return 'general_assistance';
    }

    private recommendModel(prompt: string, goal: string, availableModels?: string[]): ModelRecommendation {
        const models = availableModels || Array.from(this.modelDatabase.keys());
        const goalModelMap = this.getGoalModelMapping();
        
        // Get recommended model for goal
        let recommended = goalModelMap[goal] || 'gpt-4';
        
        // Check if recommended model is available
        if (!models.includes(recommended)) {
            // Find closest alternative
            recommended = this.findBestAlternative(recommended, models);
        }
        
        const modelInfo = this.modelDatabase.get(recommended) || this.modelDatabase.get('gpt-4')!;
        
        // Find alternatives
        const alternatives = models
            .filter(m => m !== recommended)
            .slice(0, 2);
        
        return {
            model: recommended,
            reason: this.getModelReason(goal, recommended),
            alternatives,
            costEstimate: modelInfo.costEstimate,
            speedEstimate: modelInfo.speedEstimate
        };
    }

    private recommendTemperature(prompt: string, goal: string): number {
        const goalTempMap: Record<string, number> = {
            'code_generation': 0.2,
            'test_generation': 0.3,
            'debugging': 0.1,
            'code_review': 0.2,
            'performance_review': 0.2,
            'security_review': 0.1,
            'explanation': 0.7,
            'refactoring': 0.3,
            'transformation': 0.2,
            'information_extraction': 0.1,
            'content_generation': 0.8,
            'general_assistance': 0.7
        };
        
        let temp = goalTempMap[goal] || 0.7;
        
        // Adjust based on prompt characteristics
        const lower = prompt.toLowerCase();
        if (this.patternMatcher.matches(lower, ['creative', 'brainstorm', 'ideas', 'suggestions'])) {
            temp = Math.min(temp + 0.3, 1.2);
        } else if (this.patternMatcher.matches(lower, ['exact', 'precise', 'accurate', 'correct'])) {
            temp = Math.max(temp - 0.2, 0.0);
        }
        
        return Math.round(temp * 10) / 10; // Round to 1 decimal
    }

    private recommendSamplingParams(prompt: string, goal: string, temperature: number): SamplingParams {
        const params: SamplingParams = {
            temperature
        };
        
        if (temperature < 0.3) {
            // Low temperature - use top_p for more focused responses
            params.topP = 0.9;
        } else if (temperature > 0.7) {
            // High temperature - use top_k for diversity
            params.topK = 40;
        }
        
        // Adjust for specific goals
        if (goal === 'code_generation' || goal === 'debugging') {
            params.frequencyPenalty = 0.1; // Reduce repetition
        }
        
        return params;
    }

    private identifyRequiredContext(prompt: string, goal: string, analysis: PromptAnalysis): ContextRequirement[] {
        const requirements: ContextRequirement[] = [];
        const lower = prompt.toLowerCase();
        
        // Code-related context
        if (goal.includes('code') || goal === 'debugging' || goal === 'refactoring') {
            if (!this.patternMatcher.matches(lower, ['code', 'function', 'class', 'file'])) {
                requirements.push({
                    type: 'code',
                    description: 'Include the relevant code file or function',
                    priority: 'high',
                    example: 'Paste the code you want to work with'
                });
            }
        }
        
        // Error context
        if (goal === 'debugging') {
            if (!this.patternMatcher.matches(lower, ['error', 'exception', 'traceback'])) {
                requirements.push({
                    type: 'error',
                    description: 'Include the full error message and stack trace',
                    priority: 'high',
                    example: 'Copy the complete error message'
                });
            }
        }
        
        // Environment context
        if (goal === 'code_generation' || goal === 'debugging') {
            requirements.push({
                type: 'environment',
                description: 'Specify language version, framework, and dependencies',
                priority: 'medium',
                example: 'Python 3.10, Django 4.2, PostgreSQL 14'
            });
        }
        
        // Constraints
        if (goal === 'code_generation' || goal === 'refactoring') {
            if (!this.patternMatcher.matches(lower, ['constraint', 'requirement', 'must', 'should'])) {
                requirements.push({
                    type: 'constraints',
                    description: 'Specify any constraints (performance, style, dependencies)',
                    priority: 'medium',
                    example: 'Must work with Python 3.8+, no external dependencies'
                });
            }
        }
        
        // Examples
        if (analysis.score < 60 && !this.patternMatcher.matches(lower, ['example', 'sample'])) {
            requirements.push({
                type: 'examples',
                description: 'Provide examples of desired input/output',
                priority: 'low',
                example: 'Show what good output looks like'
            });
        }
        
        // Documentation
        if (goal === 'explanation' && !this.patternMatcher.matches(lower, ['documentation', 'docs', 'reference'])) {
            requirements.push({
                type: 'documentation',
                description: 'Reference relevant documentation or standards',
                priority: 'low',
                example: 'Link to API docs or style guides'
            });
        }
        
        return requirements;
    }

    private recommendFormat(prompt: string, goal: string): FormatRecommendation {
        const formatMap: Record<string, FormatRecommendation> = {
            'code_generation': {
                format: 'code',
                structure: 'Production-ready code with comments and docstrings',
                reason: 'Code generation requires executable, well-documented code',
                example: '```python\ndef function():\n    """Docstring"""\n    # Implementation\n```'
            },
            'test_generation': {
                format: 'code',
                structure: 'Unit tests with test cases and assertions',
                reason: 'Tests need to be runnable and comprehensive',
                example: '```python\nimport unittest\nclass TestClass(unittest.TestCase):\n    ...'
            },
            'debugging': {
                format: 'structured',
                structure: 'Problem analysis, root cause, solution, and fixed code',
                reason: 'Debugging requires clear explanation and solution',
                example: 'Problem: ...\nRoot Cause: ...\nSolution: ...\nFixed Code: ...'
            },
            'code_review': {
                format: 'structured',
                structure: 'Issues found, suggestions, and refactored code',
                reason: 'Reviews need clear structure for actionable feedback',
                example: 'Issues:\n1. ...\nSuggestions:\n1. ...\nRefactored Code: ...'
            },
            'explanation': {
                format: 'markdown',
                structure: 'Markdown with headers, examples, and code blocks',
                reason: 'Explanations benefit from rich formatting',
                example: '## Concept\n### How it works\n- Point 1\n- Point 2\n```code```'
            },
            'information_extraction': {
                format: 'json',
                structure: 'Structured JSON with relevant fields',
                reason: 'Extracted information needs structured format',
                example: '{"key1": "value1", "key2": "value2"}'
            },
            'general_assistance': {
                format: 'natural',
                structure: 'Natural language response',
                reason: 'General queries work best with natural responses',
                example: 'A clear, well-structured answer'
            }
        };
        
        return formatMap[goal] || formatMap['general_assistance'];
    }

    private findMissingElements(prompt: string, analysis: PromptAnalysis, context: ContextRequirement[]): string[] {
        const missing: string[] = [];
        
        // Check principles
        analysis.principleChecks.forEach(check => {
            if (check.status === 'fail') {
                missing.push(check.principle);
            }
        });
        
        // Check context requirements
        context.filter(c => c.priority === 'high').forEach(req => {
            missing.push(`${req.type}: ${req.description}`);
        });
        
        // Check for vague language
        if (analysis.score < 40) {
            missing.push('Specific details and examples');
        }
        
        return missing;
    }

    private fillMissingParts(prompt: string, goal: string, missing: string[], format: FormatRecommendation): string {
        let filled = prompt;
        
        // If prompt is too vague, add structure
        if (missing.some(m => m.includes('Structure'))) {
            filled = this.applyFrameworkStructure(filled, goal);
        }
        
        // Add context placeholders
        if (missing.some(m => m.includes('code:'))) {
            filled += '\n\n[Code to be provided]';
        }
        
        if (missing.some(m => m.includes('error:'))) {
            filled += '\n\n[Error message to be provided]';
        }
        
        // Add format specification if missing
        if (!prompt.toLowerCase().includes('format') && !prompt.toLowerCase().includes('output')) {
            filled += `\n\n[Format]: ${format.structure}`;
        }
        
        // Add role if missing
        if (!prompt.toLowerCase().includes('role') && !prompt.toLowerCase().includes('you are')) {
            const role = this.getRoleForGoal(goal);
            filled = `[Role]: ${role}\n\n${filled}`;
        }
        
        return filled.trim();
    }

    private createIterationStrategy(prompt: string, goal: string, score: number): IterationStrategy {
        const steps: string[] = [];
        let expectedIterations = 1;
        
        if (score < 60) {
            steps.push('1. Start with initial prompt');
            steps.push('2. Review AI response for gaps');
            steps.push('3. Refine prompt with specific details');
            steps.push('4. Request improvements on weak areas');
            expectedIterations = 2;
        } else if (score < 80) {
            steps.push('1. Use initial prompt');
            steps.push('2. Request clarification if needed');
            expectedIterations = 1;
        } else {
            steps.push('1. Use prompt as-is');
            expectedIterations = 1;
        }
        
        const refinementPoints: string[] = [];
        if (goal === 'code_generation') {
            refinementPoints.push('Add error handling');
            refinementPoints.push('Include edge cases');
            refinementPoints.push('Add documentation');
        } else if (goal === 'debugging') {
            refinementPoints.push('Provide more error context');
            refinementPoints.push('Test edge cases');
        }
        
        return {
            steps,
            expectedIterations,
            refinementPoints
        };
    }

    private generateVerificationChecklist(prompt: string, goal: string, format: FormatRecommendation): string[] {
        const checklist: string[] = [];
        
        checklist.push('Verify the response addresses all requirements');
        
        if (goal.includes('code')) {
            checklist.push('Test the code with sample inputs');
            checklist.push('Check for syntax errors');
            checklist.push('Verify it follows best practices');
        }
        
        if (goal === 'debugging') {
            checklist.push('Confirm the root cause is identified');
            checklist.push('Test that the fix resolves the issue');
            checklist.push('Check for regressions');
        }
        
        if (goal === 'code_review') {
            checklist.push('Review all identified issues');
            checklist.push('Verify suggestions are applicable');
            checklist.push('Test refactored code');
        }
        
        if (format.format === 'json') {
            checklist.push('Validate JSON structure');
            checklist.push('Check all required fields are present');
        }
        
        checklist.push('Verify response matches requested format');
        checklist.push('Check for hallucinations or incorrect information');
        
        return checklist;
    }

    private calculateConfidence(analysis: PromptAnalysis, missingCount: number, contextCount: number): number {
        let confidence = analysis.score;
        
        // Reduce confidence for missing elements
        confidence -= missingCount * 5;
        
        // Reduce confidence if context is needed
        if (contextCount > 0) {
            confidence -= contextCount * 3;
        }
        
        // Boost confidence for high scores
        if (analysis.score >= 80) {
            confidence += 10;
        }
        
        return Math.max(0, Math.min(100, Math.round(confidence)));
    }

    // Helper methods
    private initializeModelDatabase(): Map<string, ModelInfo> {
        const db = new Map<string, ModelInfo>();
        
        db.set('gpt-4', {
            costEstimate: '$$$',
            speedEstimate: 'Medium',
            strengths: ['complex reasoning', 'code generation', 'analysis'],
            contextWindow: 128000
        });
        
        db.set('gpt-4-turbo', {
            costEstimate: '$$',
            speedEstimate: 'Fast',
            strengths: ['code generation', 'speed', 'cost-effective'],
            contextWindow: 128000
        });
        
        db.set('gpt-3.5-turbo', {
            costEstimate: '$',
            speedEstimate: 'Very Fast',
            strengths: ['simple tasks', 'speed', 'low cost'],
            contextWindow: 16000
        });
        
        db.set('claude-3-opus', {
            costEstimate: '$$$',
            speedEstimate: 'Medium',
            strengths: ['long context', 'nuanced understanding', 'safety'],
            contextWindow: 200000
        });
        
        db.set('claude-3-sonnet', {
            costEstimate: '$$',
            speedEstimate: 'Fast',
            strengths: ['balanced performance', 'code', 'analysis'],
            contextWindow: 200000
        });
        
        db.set('claude-3-haiku', {
            costEstimate: '$',
            speedEstimate: 'Very Fast',
            strengths: ['speed', 'simple tasks', 'low cost'],
            contextWindow: 200000
        });
        
        return db;
    }

    private getGoalModelMapping(): Record<string, string> {
        return {
            'code_generation': 'gpt-4-turbo',
            'test_generation': 'gpt-4-turbo',
            'debugging': 'gpt-4',
            'code_review': 'claude-3-sonnet',
            'performance_review': 'gpt-4',
            'security_review': 'claude-3-opus',
            'explanation': 'claude-3-sonnet',
            'refactoring': 'gpt-4-turbo',
            'transformation': 'gpt-3.5-turbo',
            'information_extraction': 'gpt-3.5-turbo',
            'content_generation': 'claude-3-sonnet',
            'general_assistance': 'gpt-3.5-turbo'
        };
    }

    private findBestAlternative(preferred: string, available: string[]): string {
        // Simple fallback logic
        if (available.includes('gpt-4-turbo')) return 'gpt-4-turbo';
        if (available.includes('gpt-4')) return 'gpt-4';
        if (available.includes('claude-3-sonnet')) return 'claude-3-sonnet';
        if (available.includes('gpt-3.5-turbo')) return 'gpt-3.5-turbo';
        return available[0] || 'gpt-4';
    }

    private getModelReason(goal: string, model: string): string {
        const reasons: Record<string, Record<string, string>> = {
            'code_generation': {
                'gpt-4-turbo': 'Excellent for code generation with good speed and cost balance',
                'gpt-4': 'Best for complex code generation requiring deep reasoning',
                'claude-3-sonnet': 'Great for code with long context needs'
            },
            'debugging': {
                'gpt-4': 'Superior reasoning for complex debugging scenarios',
                'claude-3-opus': 'Excellent for nuanced problem analysis'
            },
            'code_review': {
                'claude-3-sonnet': 'Balanced analysis with good code understanding',
                'gpt-4': 'Thorough review capabilities'
            }
        };
        
        return reasons[goal]?.[model] || `Good choice for ${goal.replace('_', ' ')}`;
    }

    private applyFrameworkStructure(prompt: string, goal: string): string {
        if (goal === 'code_generation' || goal === 'test_generation') {
            return `[Role]: You are a [language] developer
[Context]: ${prompt}
[Task]: [Specific action]
[Format]: Production-ready code`;
        }
        
        if (goal === 'debugging') {
            return `[Situation]: ${prompt}
[Task]: Identify and fix the issue
[Action]: Analyze error, find root cause, provide solution
[Result]: Working code with explanation`;
        }
        
        return prompt;
    }

    private getRoleForGoal(goal: string): string {
        const roles: Record<string, string> = {
            'code_generation': 'senior software developer',
            'debugging': 'experienced software engineer',
            'code_review': 'senior code reviewer',
            'explanation': 'technical educator',
            'refactoring': 'senior software architect'
        };
        
        return roles[goal] || 'AI assistant';
    }
}

interface ModelInfo {
    costEstimate: string;
    speedEstimate: string;
    strengths: string[];
    contextWindow: number;
}

class PatternMatcher {
    matches(text: string, patterns: string[]): boolean {
        return patterns.some(pattern => 
            text.includes(pattern.toLowerCase())
        );
    }
}



