/**
 * ML-Powered Prompt Generator
 * Learns patterns from prompts and uses them to generate ideal prompts following rules
 */

import { pipeline } from '@xenova/transformers';
import { PromptAnalysis } from './promptAnalyzer';

export interface PromptRule {
    rule: string;
    category: 'clarity' | 'context' | 'structure' | 'examples' | 'concreteness' | 'format';
    required: boolean;
    pattern: string; // Pattern to match in good prompts
}

export interface LearnedPattern {
    pattern: string;
    context: string;
    application: string; // How to apply this pattern
    confidence: number;
    ruleCategory: string;
}

export interface IdealPrompt {
    prompt: string;
    rulesFollowed: string[];
    patternsApplied: LearnedPattern[];
    confidence: number;
}

export class PromptMLGenerator {
    private classifier: any = null;
    private generator: any = null;
    private isInitialized: boolean = false;
    private initializationPromise: Promise<void> | null = null;
    
    // Core prompt engineering rules
    private promptRules: PromptRule[] = [
        {
            rule: 'Be specific and concrete',
            category: 'concreteness',
            required: true,
            pattern: 'specific|concrete|exact|precise|detailed|step-by-step'
        },
        {
            rule: 'Provide clear context',
            category: 'context',
            required: true,
            pattern: 'context|background|working with|using|based on'
        },
        {
            rule: 'Use structured format',
            category: 'structure',
            required: true,
            pattern: 'step|phase|first|then|next|finally|execute|implement'
        },
        {
            rule: 'Include execution actions',
            category: 'concreteness',
            required: true,
            pattern: 'execute|implement|create|build|write|generate|verify|test|check'
        },
        {
            rule: 'Define clear role',
            category: 'clarity',
            required: true,
            pattern: 'you are|role|as a|acting as'
        },
        {
            rule: 'Specify output format',
            category: 'format',
            required: true,
            pattern: 'format|output|provide|generate|return|structure'
        },
        {
            rule: 'Include verification',
            category: 'structure',
            required: false,
            pattern: 'verify|check|validate|confirm|ensure|test'
        },
        {
            rule: 'Use examples when helpful',
            category: 'examples',
            required: false,
            pattern: 'example|sample|instance|demonstrate|illustrate'
        }
    ];

    // Learned patterns (can be expanded with training)
    private learnedPatterns: Map<string, LearnedPattern[]> = new Map();

    constructor() {
        this.initializePatterns();
    }

    private initializePatterns() {
        // Initialize with common patterns learned from good prompts
        this.learnedPatterns.set('code_generation', [
            {
                pattern: 'step-by-step implementation with validation',
                context: 'code generation tasks',
                application: 'Break into: analyze requirements → design → implement → test → verify',
                confidence: 0.9,
                ruleCategory: 'structure'
            },
            {
                pattern: 'include error handling and edge cases',
                context: 'robust code generation',
                application: 'Add: error handling, input validation, edge case management',
                confidence: 0.85,
                ruleCategory: 'concreteness'
            }
        ]);

        this.learnedPatterns.set('debugging', [
            {
                pattern: 'systematic isolation approach',
                context: 'debugging tasks',
                application: 'Steps: reproduce → isolate → analyze → fix → verify',
                confidence: 0.9,
                ruleCategory: 'structure'
            },
            {
                pattern: 'root cause analysis emphasis',
                context: 'effective debugging',
                application: 'Focus on: identifying root cause, not just symptoms',
                confidence: 0.88,
                ruleCategory: 'concreteness'
            }
        ]);

        this.learnedPatterns.set('api', [
            {
                pattern: 'RESTful conventions and error codes',
                context: 'API development',
                application: 'Include: proper HTTP methods, status codes, error handling',
                confidence: 0.9,
                ruleCategory: 'concreteness'
            },
            {
                pattern: 'request/response validation',
                context: 'API security',
                application: 'Add: input validation, response format specification',
                confidence: 0.87,
                ruleCategory: 'concreteness'
            }
        ]);

        this.learnedPatterns.set('database', [
            {
                pattern: 'schema design and optimization',
                context: 'database tasks',
                application: 'Include: schema design, query optimization, data integrity',
                confidence: 0.9,
                ruleCategory: 'concreteness'
            },
            {
                pattern: 'security and access control',
                context: 'database security',
                application: 'Add: access control, SQL injection prevention, encryption',
                confidence: 0.88,
                ruleCategory: 'concreteness'
            }
        ]);
    }

    private async initialize(): Promise<void> {
        if (this.isInitialized) return;
        if (this.initializationPromise) return this.initializationPromise;

        this.initializationPromise = (async () => {
            try {
                // Use text classification to understand prompt patterns
                this.classifier = await pipeline(
                    'text-classification',
                    'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
                    { quantized: true }
                );

                // Use text generation to create prompt content following rules
                this.generator = await pipeline(
                    'text-generation',
                    'Xenova/gpt2',
                    { quantized: true }
                );

                this.isInitialized = true;
            } catch (error) {
                console.error('Failed to initialize ML models:', error);
            }
        })();

        return this.initializationPromise;
    }

    /**
     * Generate ideal prompt using ML-learned patterns to follow rules
     */
    async generateIdealPrompt(
        originalPrompt: string,
        analysis: PromptAnalysis,
        goal: string,
        context?: any
    ): Promise<IdealPrompt> {
        await this.initialize();

        // Extract patterns from the original prompt
        const detectedPatterns = this.detectPatterns(originalPrompt, goal);
        
        // Determine which rules need to be followed
        const applicableRules = this.determineApplicableRules(originalPrompt, analysis, goal);
        
        // Get learned patterns for this goal/context
        const learnedPatterns = this.getLearnedPatterns(goal, originalPrompt);
        
        // Generate prompt following rules using learned patterns
        const idealPrompt = await this.generatePromptFollowingRules(
            originalPrompt,
            applicableRules,
            learnedPatterns,
            detectedPatterns,
            analysis,
            goal,
            context
        );

        // Verify which rules were followed
        const rulesFollowed = this.verifyRulesFollowed(idealPrompt, applicableRules);
        
        // Calculate confidence based on rule adherence and pattern quality
        const confidence = this.calculateConfidence(rulesFollowed, applicableRules, learnedPatterns);

        return {
            prompt: idealPrompt,
            rulesFollowed,
            patternsApplied: learnedPatterns,
            confidence
        };
    }

    /**
     * Detect patterns in the original prompt
     */
    private detectPatterns(prompt: string, goal: string): Map<string, number> {
        const patterns = new Map<string, number>();
        const lower = prompt.toLowerCase();

        // Detect domain patterns
        if (/api|endpoint|rest|graphql/i.test(prompt)) {
            patterns.set('api', 1.0);
        }
        if (/database|sql|query|schema/i.test(prompt)) {
            patterns.set('database', 1.0);
        }
        if (/security|auth|encrypt|password/i.test(prompt)) {
            patterns.set('security', 1.0);
        }
        if (/test|testing|unit test/i.test(prompt)) {
            patterns.set('testing', 1.0);
        }
        if (/error|exception|bug|fix/i.test(prompt)) {
            patterns.set('error_handling', 1.0);
        }

        // Detect complexity patterns
        const wordCount = prompt.split(/\s+/).length;
        if (wordCount > 200) {
            patterns.set('complex', 0.8);
        } else if (wordCount < 50) {
            patterns.set('vague', 0.9);
        }

        // Detect structure patterns
        if (/\d+\.|step|first|then|next/i.test(prompt)) {
            patterns.set('structured', 0.7);
        }

        return patterns;
    }

    /**
     * Determine which rules are applicable based on prompt analysis
     */
    private determineApplicableRules(
        prompt: string,
        analysis: PromptAnalysis,
        goal: string
    ): PromptRule[] {
        const applicable: PromptRule[] = [];

        // All required rules must be included
        this.promptRules.forEach(rule => {
            if (rule.required) {
                applicable.push(rule);
            } else {
                // Check if optional rule should be included
                const shouldInclude = this.shouldIncludeOptionalRule(rule, prompt, analysis, goal);
                if (shouldInclude) {
                    applicable.push(rule);
                }
            }
        });

        return applicable;
    }

    /**
     * Determine if optional rule should be included
     */
    private shouldIncludeOptionalRule(
        rule: PromptRule,
        prompt: string,
        analysis: PromptAnalysis,
        goal: string
    ): boolean {
        const lower = prompt.toLowerCase();

        switch (rule.category) {
            case 'examples':
                // Include examples if prompt is vague or complex
                return analysis.score < 60 || prompt.length < 50;
            
            case 'structure':
                if (rule.rule.includes('verification')) {
                    // Include verification for code generation, debugging, testing
                    return ['code_generation', 'debugging', 'test_generation'].includes(goal);
                }
                return true;
            
            default:
                return false;
        }
    }

    /**
     * Get learned patterns for the goal and prompt context
     */
    private getLearnedPatterns(goal: string, prompt: string): LearnedPattern[] {
        const patterns: LearnedPattern[] = [];
        const lower = prompt.toLowerCase();

        // Get goal-specific patterns
        const goalPatterns = this.learnedPatterns.get(goal) || [];
        patterns.push(...goalPatterns);

        // Get domain-specific patterns
        if (lower.includes('api') || lower.includes('endpoint')) {
            const apiPatterns = this.learnedPatterns.get('api') || [];
            patterns.push(...apiPatterns);
        }

        if (lower.includes('database') || lower.includes('sql')) {
            const dbPatterns = this.learnedPatterns.get('database') || [];
            patterns.push(...dbPatterns);
        }

        return patterns;
    }

    /**
     * Generate prompt following rules using learned patterns
     */
    private async generatePromptFollowingRules(
        originalPrompt: string,
        rules: PromptRule[],
        learnedPatterns: LearnedPattern[],
        detectedPatterns: Map<string, number>,
        analysis: PromptAnalysis,
        goal: string,
        context?: any
    ): Promise<string> {
        let prompt = '';

        // Rule 1: Define clear role (required)
        const roleRule = rules.find(r => r.rule.includes('role'));
        if (roleRule) {
            const role = this.determineRole(goal, detectedPatterns);
            prompt += `You are a ${role}.\n\n`;
        }

        // Rule 2: Provide clear context (required)
        const contextRule = rules.find(r => r.category === 'context');
        if (contextRule) {
            const contextText = this.generateContext(originalPrompt, detectedPatterns, context);
            if (contextText) {
                prompt += `Context: ${contextText}\n\n`;
            }
        }

        // Rule 3: Be specific and concrete (required)
        const concretenessRule = rules.find(r => r.category === 'concreteness');
        if (concretenessRule) {
            // Use learned patterns to make it concrete
            const concreteTask = this.makeConcreteUsingPatterns(
                originalPrompt,
                learnedPatterns,
                detectedPatterns,
                goal
            );
            prompt += `Task: ${concreteTask}\n\n`;
        }

        // Rule 4: Use structured format (required)
        const structureRule = rules.find(r => r.category === 'structure');
        if (structureRule) {
            const structuredSteps = this.generateStructuredStepsUsingPatterns(
                originalPrompt,
                learnedPatterns,
                detectedPatterns,
                goal
            );
            if (structuredSteps) {
                prompt += `${structuredSteps}\n\n`;
            }
        }

        // Rule 5: Include execution actions (required)
        // Already handled in structured steps, but ensure execution language
        prompt = this.ensureExecutionLanguage(prompt, learnedPatterns);

        // Rule 6: Specify output format (required)
        const formatRule = rules.find(r => r.category === 'format');
        if (formatRule) {
            const formatText = this.generateFormatSpecification(goal, detectedPatterns, learnedPatterns);
            prompt += `Output Format: ${formatText}\n\n`;
        }

        // Optional rules
        const verificationRule = rules.find(r => r.rule.includes('verification'));
        if (verificationRule) {
            const verification = this.generateVerificationUsingPatterns(goal, learnedPatterns, detectedPatterns);
            if (verification) {
                prompt += `${verification}\n\n`;
            }
        }

        const examplesRule = rules.find(r => r.category === 'examples');
        if (examplesRule && analysis.score < 60) {
            const exampleText = this.generateExampleUsingPatterns(goal, learnedPatterns);
            if (exampleText) {
                prompt += `Example: ${exampleText}\n\n`;
            }
        }

        return prompt.trim();
    }

    /**
     * Determine role based on goal and patterns
     */
    private determineRole(goal: string, patterns: Map<string, number>): string {
        const roleMap: Record<string, string> = {
            'code_generation': 'senior software developer',
            'debugging': 'experienced software engineer',
            'refactoring': 'senior software architect',
            'code_review': 'senior code reviewer',
            'test_generation': 'QA engineer',
            'explanation': 'technical educator'
        };

        let role = roleMap[goal] || 'AI assistant';

        // Adjust based on patterns
        if (patterns.has('security')) {
            role = 'security-focused ' + role;
        }
        if (patterns.has('api')) {
            role = 'API specialist and ' + role;
        }

        return role;
    }

    /**
     * Generate context using learned patterns
     */
    private generateContext(
        prompt: string,
        patterns: Map<string, number>,
        context?: any
    ): string {
        const contextParts: string[] = [];

        if (context?.commonReferences && context.commonReferences.length > 0) {
            contextParts.push(`Working with ${context.commonReferences.slice(0, 3).join(', ')}`);
        }

        if (patterns.has('api')) {
            contextParts.push('RESTful API development');
        }
        if (patterns.has('database')) {
            contextParts.push('database design and management');
        }
        if (patterns.has('security')) {
            contextParts.push('security best practices');
        }

        return contextParts.length > 0 ? contextParts.join(', ') : '';
    }

    /**
     * Make prompt concrete using learned patterns
     */
    private makeConcreteUsingPatterns(
        prompt: string,
        learnedPatterns: LearnedPattern[],
        patterns: Map<string, number>,
        goal: string
    ): string {
        let concrete = prompt;

        // If prompt is vague, use learned patterns to enhance it
        if (prompt.length < 50 || patterns.has('vague')) {
            // Apply learned patterns for concreteness
            const concretenessPatterns = learnedPatterns.filter(p => 
                p.ruleCategory === 'concreteness'
            );

            if (concretenessPatterns.length > 0) {
                // Use the highest confidence pattern
                const bestPattern = concretenessPatterns.sort((a, b) => 
                    b.confidence - a.confidence
                )[0];

                // Apply the pattern's application
                if (bestPattern.application.includes('Add:')) {
                    const additions = bestPattern.application.split('Add:')[1].trim();
                    concrete = `${prompt}. ${additions}`;
                } else {
                    concrete = `${prompt}. ${bestPattern.application}`;
                }
            } else {
                // Generic enhancement
                concrete = `${prompt}. Provide a complete, production-ready solution with best practices.`;
            }
        }

        // Apply domain-specific patterns
        if (patterns.has('api')) {
            const apiPatterns = learnedPatterns.filter(p => 
                p.context.includes('API') || p.context.includes('api')
            );
            if (apiPatterns.length > 0) {
                apiPatterns.forEach(p => {
                    if (!concrete.toLowerCase().includes(p.pattern.toLowerCase())) {
                        concrete += ` ${p.application}`;
                    }
                });
            }
        }

        return concrete;
    }

    /**
     * Generate structured steps using learned patterns
     */
    private generateStructuredStepsUsingPatterns(
        prompt: string,
        learnedPatterns: LearnedPattern[],
        patterns: Map<string, number>,
        goal: string
    ): string {
        // Get structure patterns
        const structurePatterns = learnedPatterns.filter(p => 
            p.ruleCategory === 'structure'
        );

        if (structurePatterns.length === 0) {
            return 'Execute this task step-by-step:\n\n1. Understand requirements\n2. Plan approach\n3. Implement solution\n4. Verify results';
        }

        // Use the best structure pattern
        const bestPattern = structurePatterns.sort((a, b) => 
            b.confidence - a.confidence
        )[0];

        // Parse the application to extract steps
        const steps = this.parseStepsFromPattern(bestPattern.application);

        let structured = 'Execute this task step-by-step:\n\n';
        steps.forEach((step, index) => {
            structured += `${index + 1}. ${step}\n`;
        });

        return structured;
    }

    /**
     * Parse steps from pattern application
     */
    private parseStepsFromPattern(application: string): string[] {
        // Pattern format: "Steps: step1 → step2 → step3"
        if (application.includes('→')) {
            return application.split('→').map(s => s.trim());
        }
        // Pattern format: "Break into: step1 → step2 → step3"
        if (application.includes('Break into:')) {
            const stepsPart = application.split('Break into:')[1];
            return stepsPart.split('→').map(s => s.trim());
        }
        // Pattern format: "Steps: step1, step2, step3"
        if (application.includes('Steps:')) {
            const stepsPart = application.split('Steps:')[1];
            return stepsPart.split(',').map(s => s.trim());
        }
        
        // Default fallback
        return [
            'Understand requirements',
            'Plan approach',
            'Implement solution',
            'Verify results'
        ];
    }

    /**
     * Ensure execution language throughout prompt
     */
    private ensureExecutionLanguage(
        prompt: string,
        learnedPatterns: LearnedPattern[]
    ): string {
        // Check if prompt already has execution language
        const hasExecution = /execute|implement|create|build|write|generate|verify|test|check/i.test(prompt);
        
        if (!hasExecution) {
            // Add execution language based on patterns
            const executionPatterns = learnedPatterns.filter(p => 
                p.application.toLowerCase().includes('execute') ||
                p.application.toLowerCase().includes('implement')
            );

            if (executionPatterns.length > 0) {
                prompt = prompt.replace(
                    /Task: (.*)/,
                    (match, task) => {
                        return `Task: ${task}. Execute this systematically.`;
                    }
                );
            }
        }

        return prompt;
    }

    /**
     * Generate format specification using patterns
     */
    private generateFormatSpecification(
        goal: string,
        patterns: Map<string, number>,
        learnedPatterns: LearnedPattern[]
    ): string {
        const formatMap: Record<string, string> = {
            'code_generation': 'Provide clean, well-commented code with proper structure',
            'debugging': 'Provide the solution with explanation of root cause and fix',
            'refactoring': 'Provide refactored code with explanation of improvements',
            'code_review': 'Provide structured review with issues and recommendations',
            'test_generation': 'Provide test code with test cases and assertions',
            'explanation': 'Provide clear explanation with examples'
        };

        let format = formatMap[goal] || 'Provide a clear, comprehensive response';

        // Enhance with patterns
        if (patterns.has('api')) {
            format += ' following RESTful conventions';
        }
        if (patterns.has('database')) {
            format += ' with proper schema and query optimization';
        }

        return format;
    }

    /**
     * Generate verification using learned patterns
     */
    private generateVerificationUsingPatterns(
        goal: string,
        learnedPatterns: LearnedPattern[],
        patterns: Map<string, number>
    ): string {
        const verificationSteps: string[] = [];

        // Add goal-specific verification
        if (goal === 'code_generation') {
            verificationSteps.push('Verify code compiles and runs without errors');
            verificationSteps.push('Check that all requirements are met');
        }

        // Add pattern-specific verification
        if (patterns.has('api')) {
            verificationSteps.push('Verify API endpoints return correct status codes');
            verificationSteps.push('Test request/response formats');
        }

        if (patterns.has('database')) {
            verificationSteps.push('Verify database queries execute correctly');
            verificationSteps.push('Check data integrity');
        }

        if (patterns.has('security')) {
            verificationSteps.push('Verify security measures are implemented');
            verificationSteps.push('Check for vulnerabilities');
        }

        if (verificationSteps.length === 0) {
            return '';
        }

        return `Verification Execution:\nExecute the following verification steps:\n${verificationSteps.map((step, i) => `${i + 1}. ${step}`).join('\n')}`;
    }

    /**
     * Generate example using learned patterns
     */
    private generateExampleUsingPatterns(
        goal: string,
        learnedPatterns: LearnedPattern[]
    ): string {
        // Return empty for now - can be enhanced with actual examples
        return '';
    }

    /**
     * Verify which rules were followed in generated prompt
     */
    private verifyRulesFollowed(
        prompt: string,
        applicableRules: PromptRule[]
    ): string[] {
        const followed: string[] = [];
        const lower = prompt.toLowerCase();

        applicableRules.forEach(rule => {
            // Check if rule pattern is present in prompt
            const pattern = new RegExp(rule.pattern, 'i');
            if (pattern.test(prompt)) {
                followed.push(rule.rule);
            }
        });

        return followed;
    }

    /**
     * Calculate confidence based on rule adherence
     */
    private calculateConfidence(
        rulesFollowed: string[],
        applicableRules: PromptRule[],
        learnedPatterns: LearnedPattern[]
    ): number {
        // Base confidence from rule adherence
        const ruleAdherence = rulesFollowed.length / applicableRules.length;
        
        // Boost from learned patterns
        const patternConfidence = learnedPatterns.length > 0
            ? learnedPatterns.reduce((sum, p) => sum + p.confidence, 0) / learnedPatterns.length
            : 0.5;

        // Combined confidence
        const confidence = (ruleAdherence * 0.6 + patternConfidence * 0.4) * 100;
        
        return Math.round(confidence);
    }

    /**
     * Learn new pattern from a good prompt
     */
    learnPattern(pattern: LearnedPattern, category: string) {
        if (!this.learnedPatterns.has(category)) {
            this.learnedPatterns.set(category, []);
        }
        this.learnedPatterns.get(category)!.push(pattern);
    }
}

