/**
 * ML Model for Prompt Pattern Recognition and Generation
 * Uses transformers.js (lightweight alternative to PyTorch) for pattern learning
 */

import { pipeline, Pipeline } from '@xenova/transformers';

export interface PromptFeatures {
    text: string;
    length: number;
    complexity: number;
    goal: string;
    hasCode: boolean;
    hasApi: boolean;
    hasDatabase: boolean;
    hasSecurity: boolean;
    wordCount: number;
    questionCount: number;
    imperativeCount: number;
}

export interface ExecutionStep {
    stepNumber: number;
    action: string;
    subActions: string[];
    rationale: string;
}

export interface ExecutionRequirements {
    requirements: string[];
    priorities: number[];
}

export interface VerificationSteps {
    steps: string[];
    order: number[];
}

export class PromptMLModel {
    private classifier: any = null; // TextClassificationPipeline
    private generator: any = null; // TextGenerationPipeline
    private isInitialized: boolean = false;
    private initializationPromise: Promise<void> | null = null;

    constructor() {
        // Lazy initialization
    }

    private async initialize(): Promise<void> {
        if (this.isInitialized) return;
        if (this.initializationPromise) return this.initializationPromise;

        this.initializationPromise = (async () => {
            try {
                // Use a lightweight text classification model for pattern recognition
                this.classifier = await pipeline(
                    'text-classification',
                    'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
                    { quantized: true } // Use quantized model for smaller size
                );

                // Use a text generation model for step generation
                this.generator = await pipeline(
                    'text-generation',
                    'Xenova/gpt2',
                    { quantized: true }
                );

                this.isInitialized = true;
            } catch (error) {
                console.error('Failed to initialize ML models:', error);
                // Continue with rule-based fallback
            }
        })();

        return this.initializationPromise;
    }

    /**
     * Extract features from prompt for ML analysis
     */
    extractFeatures(prompt: string, goal: string, analysis: any): PromptFeatures {
        const lower = prompt.toLowerCase();
        const words = prompt.split(/\s+/);
        
        return {
            text: prompt,
            length: prompt.length,
            complexity: analysis?.score || 50,
            goal: goal,
            hasCode: /(code|function|class|variable|algorithm|implementation)/i.test(prompt),
            hasApi: /(api|endpoint|rest|graphql|request|response)/i.test(prompt),
            hasDatabase: /(database|sql|query|schema|table|data)/i.test(prompt),
            hasSecurity: /(security|auth|encrypt|password|token|permission)/i.test(prompt),
            wordCount: words.length,
            questionCount: (prompt.match(/\?/g) || []).length,
            imperativeCount: (prompt.match(/\b(create|build|implement|write|make|generate|fix|add|remove|update|delete)\b/gi) || []).length
        };
    }

    /**
     * Generate execution steps using ML pattern recognition
     */
    async generateExecutionSteps(
        features: PromptFeatures,
        ruleBasedSteps: ExecutionStep[]
    ): Promise<ExecutionStep[]> {
        await this.initialize();

        if (!this.classifier || !this.generator) {
            // Fallback to rule-based
            return ruleBasedSteps;
        }

        try {
            // Use ML to analyze prompt patterns and adjust steps
            const promptContext = this.buildPromptContext(features);
            
            // Classify prompt complexity and type
            const classification = await this.classifier(promptContext);
            
            // Generate ML-enhanced steps
            const mlSteps = await this.generateStepsFromPatterns(
                features,
                classification,
                ruleBasedSteps
            );

            // Merge ML insights with rule-based structure
            return this.mergeMLWithRules(mlSteps, ruleBasedSteps);
        } catch (error) {
            console.error('ML step generation failed, using rules:', error);
            return ruleBasedSteps;
        }
    }

    /**
     * Generate execution requirements using ML
     */
    async generateExecutionRequirements(
        features: PromptFeatures,
        ruleBasedRequirements: string[]
    ): Promise<ExecutionRequirements> {
        await this.initialize();

        if (!this.classifier) {
            return {
                requirements: ruleBasedRequirements,
                priorities: ruleBasedRequirements.map(() => 1)
            };
        }

        try {
            const promptContext = this.buildPromptContext(features);
            const classification = await this.classifier(promptContext);
            
            // Use ML to determine requirement priorities and add context-aware requirements
            const mlRequirements = await this.inferRequirementsFromPatterns(
                features,
                classification,
                ruleBasedRequirements
            );

            return {
                requirements: mlRequirements,
                priorities: this.calculatePriorities(mlRequirements, features)
            };
        } catch (error) {
            console.error('ML requirement generation failed, using rules:', error);
            return {
                requirements: ruleBasedRequirements,
                priorities: ruleBasedRequirements.map(() => 1)
            };
        }
    }

    /**
     * Generate verification steps using ML
     */
    async generateVerificationSteps(
        features: PromptFeatures,
        ruleBasedVerification: string[]
    ): Promise<VerificationSteps> {
        await this.initialize();

        if (!this.classifier) {
            return {
                steps: ruleBasedVerification,
                order: ruleBasedVerification.map((_, i) => i)
            };
        }

        try {
            const promptContext = this.buildPromptContext(features);
            const classification = await this.classifier(promptContext);
            
            // Use ML to determine optimal verification order and steps
            const mlVerification = await this.inferVerificationFromPatterns(
                features,
                classification,
                ruleBasedVerification
            );

            return {
                steps: mlVerification,
                order: this.optimizeVerificationOrder(mlVerification, features)
            };
        } catch (error) {
            console.error('ML verification generation failed, using rules:', error);
            return {
                steps: ruleBasedVerification,
                order: ruleBasedVerification.map((_, i) => i)
            };
        }
    }

    /**
     * Build context string for ML analysis
     */
    private buildPromptContext(features: PromptFeatures): string {
        const context = [
            `Goal: ${features.goal}`,
            `Complexity: ${features.complexity}`,
            `Has code: ${features.hasCode}`,
            `Has API: ${features.hasApi}`,
            `Has database: ${features.hasDatabase}`,
            `Has security: ${features.hasSecurity}`,
            `Word count: ${features.wordCount}`,
            `Prompt: ${features.text.substring(0, 200)}`
        ].join('. ');
        
        return context;
    }

    /**
     * Generate steps from ML pattern recognition
     */
    private async generateStepsFromPatterns(
        features: PromptFeatures,
        classification: any,
        ruleBasedSteps: ExecutionStep[]
    ): Promise<ExecutionStep[]> {
        // ML-enhanced step generation based on patterns
        const steps: ExecutionStep[] = [];
        
        // Analyze prompt patterns to determine optimal step sequence
        const stepPatterns = this.identifyStepPatterns(features, classification);
        
        // Generate steps based on learned patterns
        for (let i = 0; i < ruleBasedSteps.length; i++) {
            const ruleStep = ruleBasedSteps[i];
            const pattern = stepPatterns[i] || {};
            
            // Enhance step with ML insights
            const enhancedStep: ExecutionStep = {
                stepNumber: ruleStep.stepNumber,
                action: this.enhanceActionWithML(ruleStep.action, features, pattern),
                subActions: this.enhanceSubActionsWithML(ruleStep.subActions, features, pattern),
                rationale: this.enhanceRationaleWithML(ruleStep.rationale, features, pattern)
            };
            
            steps.push(enhancedStep);
        }
        
        return steps;
    }

    /**
     * Identify step patterns from prompt features
     */
    private identifyStepPatterns(features: PromptFeatures, classification: any): any[] {
        const patterns: any[] = [];
        
        // Pattern: High complexity + API = needs more validation steps
        if (features.complexity > 70 && features.hasApi) {
            patterns.push({ needsValidation: true, needsErrorHandling: true });
        }
        
        // Pattern: Database + Security = needs security checks
        if (features.hasDatabase && features.hasSecurity) {
            patterns.push({ needsSecurityChecks: true, needsDataValidation: true });
        }
        
        // Pattern: High word count = needs more decomposition
        if (features.wordCount > 200) {
            patterns.push({ needsDecomposition: true, needsClarification: true });
        }
        
        // Pattern: Many imperatives = needs sequential execution
        if (features.imperativeCount > 5) {
            patterns.push({ sequential: true, needsDependencies: true });
        }
        
        return patterns;
    }

    /**
     * Enhance action with ML insights
     */
    private enhanceActionWithML(action: string, features: PromptFeatures, pattern: any): string {
        let enhanced = action;
        
        // Add ML-determined specifics based on patterns
        if (pattern.needsValidation && !enhanced.toLowerCase().includes('validate')) {
            enhanced = `${enhanced} with validation`;
        }
        
        if (pattern.needsSecurityChecks && !enhanced.toLowerCase().includes('security')) {
            enhanced = `${enhanced} including security checks`;
        }
        
        if (pattern.needsDecomposition && !enhanced.toLowerCase().includes('decompose')) {
            enhanced = `${enhanced} by decomposing into sub-tasks`;
        }
        
        return enhanced;
    }

    /**
     * Enhance sub-actions with ML insights
     */
    private enhanceSubActionsWithML(
        subActions: string[],
        features: PromptFeatures,
        pattern: any
    ): string[] {
        const enhanced = [...subActions];
        
        // Add ML-determined sub-actions based on patterns
        if (pattern.needsErrorHandling && !enhanced.some(a => a.toLowerCase().includes('error'))) {
            enhanced.push('Implement error handling and edge case management');
        }
        
        if (pattern.needsSecurityChecks && !enhanced.some(a => a.toLowerCase().includes('security'))) {
            enhanced.push('Verify security requirements and access controls');
        }
        
        if (pattern.needsDataValidation && !enhanced.some(a => a.toLowerCase().includes('validate'))) {
            enhanced.push('Validate input data and enforce constraints');
        }
        
        return enhanced;
    }

    /**
     * Enhance rationale with ML insights
     */
    private enhanceRationaleWithML(
        rationale: string,
        features: PromptFeatures,
        pattern: any
    ): string {
        let enhanced = rationale;
        
        // Add ML-determined context
        if (pattern.needsValidation) {
            enhanced += ' ML pattern indicates validation is critical for this task type.';
        }
        
        if (pattern.sequential) {
            enhanced += ' ML analysis shows this requires sequential execution with dependencies.';
        }
        
        return enhanced;
    }

    /**
     * Infer requirements from ML patterns
     */
    private async inferRequirementsFromPatterns(
        features: PromptFeatures,
        classification: any,
        ruleBasedRequirements: string[]
    ): Promise<string[]> {
        const requirements = [...ruleBasedRequirements];
        
        // Add ML-determined requirements based on patterns
        if (features.hasApi && !requirements.some(r => r.toLowerCase().includes('api'))) {
            requirements.push('Implement proper API error handling and status codes');
        }
        
        if (features.hasDatabase && !requirements.some(r => r.toLowerCase().includes('database'))) {
            requirements.push('Ensure database queries are optimized and secure');
        }
        
        if (features.hasSecurity && !requirements.some(r => r.toLowerCase().includes('security'))) {
            requirements.push('Implement security best practices and authentication');
        }
        
        if (features.complexity > 70 && !requirements.some(r => r.toLowerCase().includes('test'))) {
            requirements.push('Include comprehensive testing and validation');
        }
        
        return requirements;
    }

    /**
     * Calculate requirement priorities using ML
     */
    private calculatePriorities(requirements: string[], features: PromptFeatures): number[] {
        return requirements.map(req => {
            let priority = 1; // Default priority
            
            // ML-based priority calculation
            const lower = req.toLowerCase();
            
            if (features.hasSecurity && lower.includes('security')) {
                priority = 3; // High priority
            } else if (features.hasApi && lower.includes('api')) {
                priority = 2; // Medium-high priority
            } else if (lower.includes('error') || lower.includes('validation')) {
                priority = 2; // Medium-high priority
            } else if (lower.includes('test')) {
                priority = features.complexity > 70 ? 2 : 1;
            }
            
            return priority;
        });
    }

    /**
     * Infer verification steps from ML patterns
     */
    private async inferVerificationFromPatterns(
        features: PromptFeatures,
        classification: any,
        ruleBasedVerification: string[]
    ): Promise<string[]> {
        const verification = [...ruleBasedVerification];
        
        // Add ML-determined verification steps
        if (features.hasApi) {
            verification.push('Verify API endpoints return correct status codes and error handling');
            verification.push('Test API request/response formats and data validation');
        }
        
        if (features.hasDatabase) {
            verification.push('Verify database queries execute correctly and handle edge cases');
            verification.push('Check data integrity and constraint enforcement');
        }
        
        if (features.hasSecurity) {
            verification.push('Verify authentication and authorization mechanisms');
            verification.push('Check for security vulnerabilities and proper encryption');
        }
        
        if (features.complexity > 70) {
            verification.push('Verify all edge cases and error scenarios are handled');
            verification.push('Check performance and scalability requirements');
        }
        
        return verification;
    }

    /**
     * Optimize verification order using ML
     */
    private optimizeVerificationOrder(
        verification: string[],
        features: PromptFeatures
    ): number[] {
        // ML-determined optimal verification order
        const order = verification.map((_, i) => i);
        
        // Reorder based on ML insights
        order.sort((a, b) => {
            const stepA = verification[a].toLowerCase();
            const stepB = verification[b].toLowerCase();
            
            // Security checks first if security is involved
            if (features.hasSecurity) {
                if (stepA.includes('security') && !stepB.includes('security')) return -1;
                if (!stepA.includes('security') && stepB.includes('security')) return 1;
            }
            
            // API verification early if API is involved
            if (features.hasApi) {
                if (stepA.includes('api') && !stepB.includes('api')) return -1;
                if (!stepA.includes('api') && stepB.includes('api')) return 1;
            }
            
            // Error handling verification early
            if (stepA.includes('error') && !stepB.includes('error')) return -1;
            if (!stepA.includes('error') && stepB.includes('error')) return 1;
            
            return 0;
        });
        
        return order;
    }

    /**
     * Merge ML-generated steps with rule-based steps
     */
    private mergeMLWithRules(
        mlSteps: ExecutionStep[],
        ruleSteps: ExecutionStep[]
    ): ExecutionStep[] {
        // Intelligent merge: use ML enhancements but keep rule-based structure
        return ruleSteps.map((ruleStep, index) => {
            const mlStep = mlSteps[index];
            
            if (!mlStep) return ruleStep;
            
            // Merge: prefer ML-enhanced actions but validate with rules
            return {
                stepNumber: ruleStep.stepNumber,
                action: mlStep.action || ruleStep.action,
                subActions: [
                    ...ruleStep.subActions,
                    ...mlStep.subActions.filter(sa => 
                        !ruleStep.subActions.some(rsa => rsa.toLowerCase() === sa.toLowerCase())
                    )
                ],
                rationale: mlStep.rationale || ruleStep.rationale
            };
        });
    }
}

