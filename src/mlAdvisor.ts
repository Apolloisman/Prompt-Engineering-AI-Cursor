import { PromptAnalyzer, PromptAnalysis } from './promptAnalyzer';
import { ChatHistoryContext } from './chatHistoryManager';
import { PromptAgent, AgenticPrompt } from './promptAgent';

export interface TaskActionStep {
    stepNumber: number;
    action: string;
    description: string;
    why: string;
    goal: string; // Specific goal for this step
    subGoals?: string[]; // Sub-goals to achieve this step's goal
    estimatedTime?: string;
    dependencies?: string[];
    requiredContext?: string[];
    successCriteria?: string[]; // How to know this step is complete
}

export interface TaskActionPlan {
    goal: string;
    overallGoal: string; // Clear statement of what we're trying to achieve
    planningStrategy: string; // The agent's chosen approach/strategy
    steps: TaskActionStep[];
    totalEstimatedTime?: string;
    complexity: 'simple' | 'medium' | 'complex';
    prerequisites: string[];
    basedOnHistory: boolean;
    historyPattern?: string;
    goalHierarchy: { // How goals relate to each other
        overall: string;
        phaseGoals: { phase: string; goal: string; steps: number[] }[];
    };
}

export interface MLRecommendations {
    goal: string;
    suggestedModel: ModelRecommendation;
    cursorAutoModelSufficient: boolean;
    cursorAutoModelReason: string;
    frontierModelRecommendation?: FrontierModelRecommendation;
    gemini3FreeCompatible: GeminiCompatibility;
    temperature: number;
    samplingParams: SamplingParams;
    requiredContext: ContextRequirement[];
    requiredReferences: ReferenceRequirement[];
    referenceAssessment: ReferenceAssessment;
    suggestedFormat: FormatRecommendation;
    suggestedLanguages: LanguageRecommendation[];
    missingElements: string[];
    filledPrompt: string;
    agenticPrompt?: AgenticPrompt;
    iterationStrategy: IterationStrategy;
    taskActionPlan: TaskActionPlan;
    verificationChecklist: string[];
    confidence: number;
}

export interface ModelRecommendation {
    model: string;
    reason: string;
    alternatives: string[];
    costEstimate: string;
    speedEstimate: string;
    isFrontier: boolean;
}

export interface FrontierModelRecommendation {
    recommended: string;
    reason: string;
    alternatives: string[];
    whenToUse: string;
}

export interface GeminiCompatibility {
    compatible: boolean;
    reason: string;
    estimatedTokens: number;
    limitations: string[];
    recommendations: string[];
}

export interface ReferenceRequirement {
    type: 'documentation' | 'api' | 'tutorial' | 'example' | 'specification' | 'standard' | 'library';
    description: string;
    priority: 'high' | 'medium' | 'low';
    examples: string[];
    whereToFind: string;
    criticalForDecision: boolean;
    whyNeeded: string;
}

export interface ReferenceAssessment {
    criticalReferencesNeeded: ReferenceRequirement[];
    referencesImpactDecision: boolean;
    decisionConfidenceWithoutRefs: number;
    decisionConfidenceWithRefs: number;
    processStepsRequiringRefs: string[];
    recommendation: string;
}

export interface LanguageRecommendation {
    language: string;
    reason: string;
    complexity: 'simple' | 'medium' | 'complex';
    suitability: number; // 0-100
    pros: string[];
    cons: string[];
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
    private promptAgent: PromptAgent;

    constructor() {
        this.analyzer = new PromptAnalyzer();
        this.modelDatabase = this.initializeModelDatabase();
        this.patternMatcher = new PatternMatcher();
        this.promptAgent = new PromptAgent();
    }

    async analyzeAndRecommend(prompt: string, availableModels?: string[], chatHistory?: ChatHistoryContext): Promise<MLRecommendations> {
        // Analyze the prompt
        const analysis = this.analyzer.analyze(prompt);
        
        // Determine goal (considering chat history patterns)
        const goal = this.determineGoal(prompt, analysis, chatHistory);
        
        // Check if Cursor's auto model is sufficient
        const cursorAutoModelCheck = this.checkCursorAutoModelSufficiency(prompt, goal, analysis);
        
        // Recommend model
        const suggestedModel = this.recommendModel(prompt, goal, availableModels);
        
        // Recommend frontier models if needed
        const frontierModel = cursorAutoModelCheck.sufficient ? undefined : this.recommendFrontierModel(prompt, goal);
        
        // Check Gemini 3 free mode compatibility
        const gemini3Compatibility = this.checkGemini3FreeCompatibility(prompt, goal, analysis);
        
        // Suggest parameters
        const temperature = this.recommendTemperature(prompt, goal);
        const samplingParams = this.recommendSamplingParams(prompt, goal, temperature);
        
        // Identify required context (considering what was already provided)
        const requiredContext = this.identifyRequiredContext(prompt, goal, analysis, chatHistory);
        
        // Identify required references (filtering out already mentioned ones)
        const requiredReferences = this.identifyRequiredReferences(prompt, goal, chatHistory);
        
        // Assess if references are critical for decision-making
        const referenceAssessment = this.assessReferenceNeeds(prompt, goal, requiredReferences, analysis, chatHistory);
        
        // Adjust model recommendation based on reference needs
        const adjustedModel = this.adjustModelForReferences(suggestedModel, referenceAssessment);
        
        // Recommend programming languages (considering reference availability)
        const suggestedLanguages = this.recommendLanguages(prompt, goal, referenceAssessment);
        
        // Recommend format
        const suggestedFormat = this.recommendFormat(prompt, goal);
        
        // Check if prompt needs more detail/breakdown
        const detailRecommendation = this.assessDetailLevel(prompt, goal, analysis);
        
        // Find missing elements (including reference needs)
        const missingElements = this.findMissingElements(prompt, analysis, requiredContext, referenceAssessment);
        
        // Add detail recommendations to missing elements
        if (detailRecommendation.needsMoreDetail || detailRecommendation.needsBreakdown) {
            missingElements.push(...detailRecommendation.recommendations);
        }
        
        // Generate concrete task action plan (using ML/history) - needed for enhanced prompt
        const taskActionPlan = this.generateTaskActionPlan(prompt, goal, analysis, chatHistory, referenceAssessment);
        
        // Calculate confidence first (needed for agentic prompt)
        const confidence = this.calculateConfidence(analysis, missingElements.length, requiredContext.length, referenceAssessment);
        
        // Generate verification checklist (needed for agentic prompt)
        const verificationChecklist = this.generateVerificationChecklist(prompt, goal, suggestedFormat);
        
        // Generate iteration strategy (needed for agentic prompt)
        const iterationStrategy = this.createIterationStrategy(prompt, goal, analysis.score, referenceAssessment, detailRecommendation);
        
        // Generate full ML recommendations to pass to agentic system
        const fullMLRecommendations: MLRecommendations = {
            goal,
            suggestedModel,
            cursorAutoModelSufficient: cursorAutoModelCheck.sufficient,
            cursorAutoModelReason: cursorAutoModelCheck.reason,
            frontierModelRecommendation: frontierModel,
            gemini3FreeCompatible: gemini3Compatibility,
            temperature,
            samplingParams,
            requiredContext,
            requiredReferences,
            referenceAssessment,
            suggestedFormat,
            suggestedLanguages,
            missingElements,
            filledPrompt: '', // Will be set below
            iterationStrategy,
            taskActionPlan,
            verificationChecklist,
            confidence
        };
        
        // Generate agentic prompt with full ML recommendations for structural decisions
        // Now async to support ML model inference
        const finalAgenticPrompt = await this.promptAgent.generateAgenticPrompt(
            prompt,
            analysis,
            chatHistory,
            referenceAssessment,
            fullMLRecommendations
        );
        
        // Use agentic prompt as the filled prompt (it's more comprehensive and structurally optimized)
        const filledPrompt = finalAgenticPrompt.optimizedPrompt;
        
        // Note: confidence, verificationChecklist, and iterationStrategy are now calculated above
        
        return {
            goal,
            suggestedModel,
            cursorAutoModelSufficient: cursorAutoModelCheck.sufficient,
            cursorAutoModelReason: cursorAutoModelCheck.reason,
            frontierModelRecommendation: frontierModel,
            gemini3FreeCompatible: gemini3Compatibility,
            temperature,
            samplingParams,
            requiredContext,
            requiredReferences,
            referenceAssessment,
            suggestedFormat,
            suggestedLanguages,
            missingElements,
            filledPrompt,
            agenticPrompt: finalAgenticPrompt,
            iterationStrategy,
            taskActionPlan,
            verificationChecklist,
            confidence
        };
    }

    private determineGoal(prompt: string, analysis: PromptAnalysis, chatHistory?: ChatHistoryContext): string {
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
        
        // Check if it's a frontier model
        const frontierModels = ['gpt-4', 'gpt-4-turbo', 'claude-3-opus', 'claude-3-sonnet'];
        const isFrontier = frontierModels.includes(recommended);
        
        return {
            model: recommended,
            reason: this.getModelReason(goal, recommended),
            alternatives,
            costEstimate: modelInfo.costEstimate,
            speedEstimate: modelInfo.speedEstimate,
            isFrontier
        };
    }

    private checkCursorAutoModelSufficiency(prompt: string, goal: string, analysis: PromptAnalysis): { sufficient: boolean; reason: string } {
        const complexity = this.assessComplexity(prompt, goal, analysis);
        
        // Simple tasks that Cursor's auto model handles well
        const simpleTasks = [
            'code_generation', // Simple code generation
            'transformation', // Code transformation
            'information_extraction' // Simple extraction
        ];
        
        // Tasks that need frontier models
        const complexTasks = [
            'debugging', // Complex debugging often needs better reasoning
            'code_review', // Deep analysis
            'performance_review', // Requires advanced understanding
            'security_review' // Critical analysis
        ];
        
        if (complexity === 'low' && simpleTasks.includes(goal)) {
            return {
                sufficient: true,
                reason: 'Cursor\'s auto model selection is sufficient for this simple task. The auto model will choose an appropriate model based on complexity.'
            };
        }
        
        if (complexity === 'high' || complexTasks.includes(goal) || analysis.score < 50) {
            return {
                sufficient: false,
                reason: 'This task requires advanced reasoning or has high complexity. Consider using a frontier model for better results.'
            };
        }
        
        return {
            sufficient: true,
            reason: 'Cursor\'s auto model should work, but a frontier model may provide better quality for this task.'
        };
    }

    private recommendFrontierModel(prompt: string, goal: string): FrontierModelRecommendation {
        const frontierModels: Record<string, FrontierModelRecommendation> = {
            'code_generation': {
                recommended: 'GPT-4 Turbo',
                reason: 'Excellent code generation with good speed and cost balance',
                alternatives: ['Claude 3 Sonnet', 'GPT-4'],
                whenToUse: 'For complex code generation, refactoring, or when you need production-ready code'
            },
            'debugging': {
                recommended: 'GPT-4',
                reason: 'Superior reasoning capabilities for complex debugging scenarios',
                alternatives: ['Claude 3 Opus', 'GPT-4 Turbo'],
                whenToUse: 'For difficult bugs, complex error analysis, or when standard debugging fails'
            },
            'code_review': {
                recommended: 'Claude 3 Sonnet',
                reason: 'Balanced analysis with excellent code understanding and safety',
                alternatives: ['GPT-4', 'Claude 3 Opus'],
                whenToUse: 'For thorough code reviews, security analysis, or architectural feedback'
            },
            'performance_review': {
                recommended: 'GPT-4',
                reason: 'Strong analytical capabilities for performance optimization',
                alternatives: ['Claude 3 Sonnet', 'GPT-4 Turbo'],
                whenToUse: 'For performance analysis, optimization suggestions, or bottleneck identification'
            },
            'security_review': {
                recommended: 'Claude 3 Opus',
                reason: 'Excellent safety focus and nuanced security analysis',
                alternatives: ['GPT-4', 'Claude 3 Sonnet'],
                whenToUse: 'For security audits, vulnerability assessment, or security-critical code'
            }
        };
        
        return frontierModels[goal] || {
            recommended: 'GPT-4 Turbo',
            reason: 'Good general-purpose frontier model',
            alternatives: ['Claude 3 Sonnet', 'GPT-4'],
            whenToUse: 'For tasks requiring advanced reasoning or high-quality output'
        };
    }

    private checkGemini3FreeCompatibility(prompt: string, goal: string, analysis: PromptAnalysis): GeminiCompatibility {
        const estimatedTokens = this.estimateTokenCount(prompt);
        const maxTokens = 100000; // Gemini 3 free mode limit
        
        // Check if task is suitable for Gemini
        const geminiCompatibleGoals = [
            'code_generation',
            'explanation',
            'transformation',
            'information_extraction',
            'content_generation'
        ];
        
        const incompatibleGoals = [
            'security_review', // May have safety restrictions
            'performance_review' // May need more advanced reasoning
        ];
        
        const compatible = 
            estimatedTokens < maxTokens * 0.8 && // Use 80% of limit for safety
            geminiCompatibleGoals.includes(goal) &&
            !incompatibleGoals.includes(goal) &&
            analysis.score >= 50; // Need decent prompt quality
        
        const limitations: string[] = [];
        const recommendations: string[] = [];
        
        if (estimatedTokens > maxTokens * 0.8) {
            limitations.push(`Estimated tokens (${estimatedTokens.toLocaleString()}) may exceed free tier limit`);
            recommendations.push('Consider reducing context or splitting the task');
        }
        
        if (!geminiCompatibleGoals.includes(goal)) {
            limitations.push(`Task type (${goal}) may not be optimal for Gemini`);
            recommendations.push('Consider using GPT-4 or Claude for this task type');
        }
        
        if (compatible) {
            recommendations.push('Gemini 3 free mode is suitable for this task');
            recommendations.push('Great cost-effective option (free up to 100k tokens)');
        }
        
        return {
            compatible,
            reason: compatible 
                ? `Task is compatible with Gemini 3 free mode. Estimated ${estimatedTokens.toLocaleString()} tokens (well under 100k limit).`
                : `Task may not be optimal for Gemini 3 free mode. ${limitations.join('; ')}`,
            estimatedTokens,
            limitations: limitations.length > 0 ? limitations : ['None'],
            recommendations
        };
    }

    private identifyRequiredReferences(prompt: string, goal: string, chatHistory?: ChatHistoryContext): ReferenceRequirement[] {
        const requirements: ReferenceRequirement[] = [];
        const lower = prompt.toLowerCase();
        
        // API documentation - CRITICAL for decision-making
        if (goal === 'code_generation' && this.patternMatcher.matches(lower, ['api', 'library', 'framework', 'sdk'])) {
            requirements.push({
                type: 'api',
                description: 'API documentation for the libraries/frameworks you\'re using',
                priority: 'high',
                examples: ['Official API docs', 'Library documentation', 'SDK reference'],
                whereToFind: 'Check official documentation sites (e.g., docs.python.org, developer.mozilla.org)',
                criticalForDecision: true,
                whyNeeded: 'Without API docs, the AI cannot know correct function signatures, parameters, or return types. This leads to incorrect code generation.'
            });
        }
        
        // Language documentation - HELPFUL but not always critical
        if (goal.includes('code')) {
            requirements.push({
                type: 'documentation',
                description: 'Language-specific documentation and style guides',
                priority: 'medium',
                examples: ['Python PEP 8', 'JavaScript MDN', 'TypeScript Handbook'],
                whereToFind: 'Official language documentation and style guides',
                criticalForDecision: false,
                whyNeeded: 'Helps ensure code follows best practices and language conventions, improving quality and maintainability.'
            });
        }
        
        // Tutorials/examples - HELPFUL for learning
        if (this.patternMatcher.matches(lower, ['learn', 'how', 'tutorial'])) {
            requirements.push({
                type: 'tutorial',
                description: 'Tutorials or examples showing similar implementations',
                priority: 'low',
                examples: ['Stack Overflow examples', 'GitHub code samples', 'Official tutorials'],
                whereToFind: 'Search GitHub, Stack Overflow, or official tutorial sites',
                criticalForDecision: false,
                whyNeeded: 'Provides examples of similar implementations to guide the solution approach.'
            });
        }
        
        // Specifications - CRITICAL for reviews
        if (goal === 'code_review' || goal === 'security_review') {
            requirements.push({
                type: 'specification',
                description: 'Relevant specifications or standards',
                priority: 'high',
                examples: ['Security best practices', 'Performance benchmarks', 'Code standards'],
                whereToFind: 'Industry standards, OWASP guidelines, language-specific best practices',
                criticalForDecision: true,
                whyNeeded: 'Reviews require standards and best practices to evaluate code quality, security, and performance properly.'
            });
        }
        
        // Library documentation - CRITICAL if library is mentioned
        if (this.patternMatcher.matches(lower, ['react', 'django', 'express', 'flask', 'library'])) {
            const libRef: ReferenceRequirement = {
                type: 'library',
                description: 'Documentation for specific libraries/frameworks mentioned',
                priority: 'high',
                examples: ['React docs', 'Django documentation', 'Express.js guide'],
                whereToFind: 'Official library documentation websites',
                criticalForDecision: true,
                whyNeeded: 'Library-specific APIs, patterns, and conventions are essential for correct implementation.'
            };
            
            // Check if already mentioned in chat history
            if (!chatHistory || this.shouldSuggestReference(libRef.description, chatHistory)) {
                requirements.push(libRef);
            }
        }
        
        // Filter out references already mentioned in chat history
        if (chatHistory && chatHistory.alreadyMentionedReferences.length > 0) {
            return requirements.filter(req => {
                return !chatHistory.alreadyMentionedReferences.some(mentioned => 
                    req.description.toLowerCase().includes(mentioned.toLowerCase()) ||
                    mentioned.toLowerCase().includes(req.type.toLowerCase())
                );
            });
        }
        
        return requirements;
    }

    private shouldSuggestReference(refDescription: string, chatHistory: ChatHistoryContext): boolean {
        return !chatHistory.alreadyMentionedReferences.some(mentioned => 
            refDescription.toLowerCase().includes(mentioned.toLowerCase()) ||
            mentioned.toLowerCase().includes(refDescription.toLowerCase())
        );
    }

    private assessReferenceNeeds(prompt: string, goal: string, references: ReferenceRequirement[], analysis: PromptAnalysis, chatHistory?: ChatHistoryContext): ReferenceAssessment {
        const criticalRefs = references.filter(r => r.criticalForDecision);
        const processSteps: string[] = [];
        
        // Determine which decision steps need references
        if (goal === 'code_generation') {
            if (criticalRefs.some(r => r.type === 'api' || r.type === 'library')) {
                processSteps.push('Determining correct API usage and function signatures');
                processSteps.push('Selecting appropriate library methods');
                processSteps.push('Ensuring compatibility with library versions');
            }
        }
        
        if (goal === 'code_review' || goal === 'security_review') {
            if (criticalRefs.some(r => r.type === 'specification')) {
                processSteps.push('Evaluating against security standards');
                processSteps.push('Checking compliance with best practices');
                processSteps.push('Assessing performance against benchmarks');
            }
        }
        
        // Calculate confidence impact
        let confidenceWithoutRefs = analysis.score;
        let confidenceWithRefs = analysis.score;
        
        if (criticalRefs.length > 0) {
            // Each critical reference missing reduces confidence
            confidenceWithoutRefs = Math.max(0, analysis.score - (criticalRefs.length * 15));
            confidenceWithRefs = Math.min(100, analysis.score + (criticalRefs.length * 10));
        }
        
        const referencesImpactDecision = criticalRefs.length > 0 || processSteps.length > 0;
        
        let recommendation = '';
        if (criticalRefs.length > 0) {
            const alreadyMentioned = chatHistory?.alreadyMentionedReferences.length || 0;
            if (alreadyMentioned > 0) {
                recommendation = `⚠️ CRITICAL: ${criticalRefs.length} critical reference(s) still needed. ${alreadyMentioned} reference(s) already mentioned in chat history, but additional ones are required for proper decision-making.`;
            } else {
                recommendation = `⚠️ CRITICAL: ${criticalRefs.length} critical reference(s) needed for proper decision-making. Without these, the AI may generate incorrect or suboptimal solutions.`;
            }
        } else if (references.length > 0) {
            if (chatHistory && chatHistory.alreadyMentionedReferences.length > 0) {
                recommendation = `Some references already mentioned in chat history. Additional references will improve code quality and accuracy.`;
            } else {
                recommendation = `References are helpful but not critical. Having them will improve code quality and accuracy.`;
            }
        } else {
            if (chatHistory && chatHistory.alreadyMentionedReferences.length > 0) {
                recommendation = `Good! Required references have been mentioned in previous messages. The prompt contains sufficient information for decision-making.`;
            } else {
                recommendation = `No specific references required. The prompt contains sufficient information for decision-making.`;
            }
        }
        
        // Add chat history context if available
        if (chatHistory && chatHistory.patterns.length > 0) {
            recommendation += `\n\n📊 Chat patterns detected: ${chatHistory.patterns.join(', ')}`;
        }
        
        return {
            criticalReferencesNeeded: criticalRefs,
            referencesImpactDecision: referencesImpactDecision,
            decisionConfidenceWithoutRefs: confidenceWithoutRefs,
            decisionConfidenceWithRefs: confidenceWithRefs,
            processStepsRequiringRefs: processSteps,
            recommendation
        };
    }

    private adjustModelForReferences(model: ModelRecommendation, assessment: ReferenceAssessment): ModelRecommendation {
        // If references are critical, prefer models with better web access or knowledge
        if (assessment.criticalReferencesNeeded.length > 0) {
            // Models with better knowledge/context handling
            const betterKnowledgeModels = ['gpt-4', 'claude-3-opus', 'claude-3-sonnet'];
            
            if (!betterKnowledgeModels.includes(model.model)) {
                return {
                    ...model,
                    reason: model.reason + ' Note: With critical references needed, consider a model with better knowledge access.',
                    alternatives: [...betterKnowledgeModels.filter(m => m !== model.model).slice(0, 2), ...model.alternatives]
                };
            }
        }
        
        return model;
    }

    private recommendLanguages(prompt: string, goal: string, referenceAssessment?: ReferenceAssessment): LanguageRecommendation[] {
        const lower = prompt.toLowerCase();
        const recommendations: LanguageRecommendation[] = [];
        
        // Detect mentioned languages
        const mentionedLanguages = this.detectMentionedLanguages(lower);
        
        // Language suitability for different goals
        const languageSuitability: Record<string, Record<string, { suitability: number; pros: string[]; cons: string[] }>> = {
            'code_generation': {
                'python': { suitability: 95, pros: ['Simple syntax', 'Great AI support', 'Rich libraries'], cons: ['Slower than compiled'] },
                'javascript': { suitability: 90, pros: ['Versatile', 'Great for web', 'Large ecosystem'], cons: ['Type safety issues'] },
                'typescript': { suitability: 95, pros: ['Type safety', 'Great tooling', 'Modern'], cons: ['Requires compilation'] },
                'rust': { suitability: 70, pros: ['Fast', 'Memory safe'], cons: ['Steep learning curve', 'Complex'] },
                'go': { suitability: 80, pros: ['Simple', 'Fast', 'Concurrent'], cons: ['Less expressive'] },
                'java': { suitability: 75, pros: ['Enterprise-ready', 'Strong typing'], cons: ['Verbose', 'Complex'] }
            },
            'debugging': {
                'python': { suitability: 90, pros: ['Clear error messages', 'Easy to debug'], cons: [] },
                'javascript': { suitability: 85, pros: ['Good debugging tools'], cons: ['Type errors'] },
                'typescript': { suitability: 90, pros: ['Type checking helps', 'Good tooling'], cons: [] }
            }
        };
        
        // If languages mentioned, prioritize those
        if (mentionedLanguages.length > 0) {
            mentionedLanguages.forEach(lang => {
                const suitability = languageSuitability[goal]?.[lang] || { suitability: 70, pros: [], cons: [] };
                recommendations.push({
                    language: lang.charAt(0).toUpperCase() + lang.slice(1),
                    reason: `Mentioned in prompt - ${this.getLanguageReason(lang, goal)}`,
                    complexity: this.getLanguageComplexity(lang),
                    suitability: suitability.suitability,
                    pros: suitability.pros,
                    cons: suitability.cons
                });
            });
        }
        
        // Add general recommendations if none mentioned
        if (recommendations.length === 0) {
            const generalRecs = this.getGeneralLanguageRecommendations(goal);
            recommendations.push(...generalRecs);
        }
        
        // Adjust suitability based on reference availability
        if (referenceAssessment && referenceAssessment.criticalReferencesNeeded.length > 0) {
            // Languages with better documentation availability get a boost
            recommendations.forEach(rec => {
                if (['Python', 'JavaScript', 'TypeScript'].includes(rec.language)) {
                    rec.suitability = Math.min(100, rec.suitability + 5);
                    rec.reason += ' (Excellent documentation available)';
                }
            });
        }
        
        // Sort by suitability
        return recommendations.sort((a, b) => b.suitability - a.suitability);
    }

    private detectMentionedLanguages(text: string): string[] {
        const languages: string[] = [];
        const langPatterns: Record<string, string[]> = {
            'python': ['python', 'py', 'django', 'flask', 'pandas', 'numpy'],
            'javascript': ['javascript', 'js', 'node', 'react', 'vue', 'angular'],
            'typescript': ['typescript', 'ts', 'tsx'],
            'rust': ['rust', 'cargo'],
            'go': ['go', 'golang'],
            'java': ['java', 'spring', 'maven'],
            'csharp': ['c#', 'csharp', '.net', 'asp.net'],
            'cpp': ['c++', 'cpp', 'cplusplus'],
            'c': [' c ', ' c,', ' c.', ' c\n']
        };
        
        for (const [lang, patterns] of Object.entries(langPatterns)) {
            if (patterns.some(p => text.includes(p.toLowerCase()))) {
                languages.push(lang);
            }
        }
        
        return languages;
    }

    private getGeneralLanguageRecommendations(goal: string): LanguageRecommendation[] {
        const recs: LanguageRecommendation[] = [];
        
        if (goal === 'code_generation' || goal === 'debugging') {
            recs.push({
                language: 'Python',
                reason: 'Excellent for AI-assisted development, simple syntax, great libraries',
                complexity: 'simple',
                suitability: 95,
                pros: ['Simple syntax', 'Great AI support', 'Rich ecosystem', 'Easy to learn'],
                cons: ['Slower execution']
            });
            
            recs.push({
                language: 'TypeScript',
                reason: 'Type safety helps AI generate better code, modern tooling',
                complexity: 'medium',
                suitability: 90,
                pros: ['Type safety', 'Great tooling', 'Modern features'],
                cons: ['Requires compilation', 'More verbose than JavaScript']
            });
            
            recs.push({
                language: 'JavaScript',
                reason: 'Versatile, great for web development, large ecosystem',
                complexity: 'simple',
                suitability: 85,
                pros: ['Versatile', 'No compilation', 'Huge ecosystem'],
                cons: ['No type safety', 'Can be error-prone']
            });
        }
        
        return recs;
    }

    private getLanguageReason(lang: string, goal: string): string {
        const reasons: Record<string, Record<string, string>> = {
            'python': {
                'code_generation': 'Simple syntax makes it easy for AI to generate correct code',
                'debugging': 'Clear error messages and simple structure aid debugging'
            },
            'typescript': {
                'code_generation': 'Type safety helps AI generate more accurate code',
                'debugging': 'Type checking catches errors early'
            }
        };
        
        return reasons[lang]?.[goal] || 'Good choice for this task';
    }

    private getLanguageComplexity(lang: string): 'simple' | 'medium' | 'complex' {
        const simple = ['python', 'javascript', 'go'];
        const medium = ['typescript', 'java', 'csharp'];
        const complex = ['rust', 'cpp', 'c'];
        
        if (simple.includes(lang)) return 'simple';
        if (medium.includes(lang)) return 'medium';
        if (complex.includes(lang)) return 'complex';
        return 'medium';
    }

    private assessComplexity(prompt: string, goal: string, analysis: PromptAnalysis): 'low' | 'medium' | 'high' {
        let score = 0;
        
        // Length indicates complexity
        if (prompt.length > 500) score += 1;
        if (prompt.length > 1000) score += 1;
        
        // Low analysis score indicates complexity
        if (analysis.score < 50) score += 2;
        if (analysis.score < 30) score += 1;
        
        // Goal complexity
        const complexGoals = ['debugging', 'code_review', 'performance_review', 'security_review'];
        if (complexGoals.includes(goal)) score += 2;
        
        // Multiple requirements
        if (analysis.principleChecks.filter(c => c.status === 'fail').length > 3) score += 1;
        
        if (score >= 4) return 'high';
        if (score >= 2) return 'medium';
        return 'low';
    }

    private estimateTokenCount(prompt: string): number {
        // Rough estimation: ~4 characters per token for English
        // More accurate: count words and multiply by 1.3
        const words = prompt.split(/\s+/).length;
        return Math.ceil(words * 1.3);
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

    private identifyRequiredContext(prompt: string, goal: string, analysis: PromptAnalysis, chatHistory?: ChatHistoryContext): ContextRequirement[] {
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
        
        // Environment context (skip if already provided)
        if (goal === 'code_generation' || goal === 'debugging') {
            const envReq: ContextRequirement = {
                type: 'environment',
                description: 'Specify language version, framework, and dependencies',
                priority: 'medium',
                example: 'Python 3.10, Django 4.2, PostgreSQL 14'
            };
            
            // Check if environment was already mentioned
            if (!chatHistory || !chatHistory.alreadyProvidedContext.some(ctx => 
                ctx.toLowerCase().includes('environment') || 
                ctx.toLowerCase().includes('version') ||
                ctx.toLowerCase().includes('framework')
            )) {
                requirements.push(envReq);
            }
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

    private findMissingElements(prompt: string, analysis: PromptAnalysis, context: ContextRequirement[], referenceAssessment?: ReferenceAssessment): string[] {
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
        
        // Check for critical references
        if (referenceAssessment && referenceAssessment.criticalReferencesNeeded.length > 0) {
            referenceAssessment.criticalReferencesNeeded.forEach(ref => {
                missing.push(`CRITICAL REFERENCE: ${ref.type} - ${ref.description}`);
            });
        }
        
        // Check for vague language
        if (analysis.score < 40) {
            missing.push('Specific details and examples');
        }
        
        return missing;
    }

    private assessDetailLevel(prompt: string, goal: string, analysis: PromptAnalysis): { needsMoreDetail: boolean; needsBreakdown: boolean; recommendations: string[] } {
        const wordCount = prompt.split(/\s+/).length;
        const hasSteps = /(step|first|then|next|finally|\d+\.)/i.test(prompt);
        const hasDetails = wordCount > 100 || 
                          /(specific|exact|precise|detailed|concrete|particular|example)/i.test(prompt) ||
                          /\d+/.test(prompt) ||
                          /['"]([^'"]{10,})['"]/.test(prompt);
        
        const needsMoreDetail = wordCount < 50 || (!hasDetails && analysis.score < 70);
        const needsBreakdown = !hasSteps && (wordCount > 30 || analysis.score < 60);
        
        const recommendations: string[] = [];
        
        if (needsMoreDetail) {
            recommendations.push('Add more specific details about what you want');
            recommendations.push('Include examples or specific requirements');
            recommendations.push('Specify: What exactly? Where? How? Why?');
            recommendations.push('Add concrete examples of desired output');
        }
        
        if (needsBreakdown) {
            recommendations.push('Break the task into numbered steps (1, 2, 3...)');
            recommendations.push('Break down complex parts into sub-steps');
            recommendations.push('Use action verbs for each step: "add", "remove", "extract"');
            recommendations.push('Make each step independently actionable');
        }
        
        return { needsMoreDetail, needsBreakdown, recommendations };
    }

    private fillMissingParts(prompt: string, goal: string, missing: string[], format: FormatRecommendation, referenceAssessment?: ReferenceAssessment, detailRecommendation?: any, taskActionPlan?: TaskActionPlan, analysis?: PromptAnalysis): string {
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
        
        // Add reference placeholders if critical references needed
        if (referenceAssessment && referenceAssessment.criticalReferencesNeeded.length > 0) {
            filled += '\n\n[CRITICAL REFERENCES NEEDED]:';
            referenceAssessment.criticalReferencesNeeded.forEach(ref => {
                filled += `\n- ${ref.type}: ${ref.description} (${ref.whereToFind})`;
            });
            filled += '\n\nPlease reference these when generating the solution to ensure accuracy.';
        }
        
        // AGENTIC: Add optimized prompt structure with goals and planning
        if (taskActionPlan) {
            filled += '\n\n[AGENT-GENERATED OPTIMIZED PLAN]:';
            filled += `\n\nOverall Goal: ${taskActionPlan.overallGoal}`;
            filled += `\nPlanning Strategy: ${taskActionPlan.planningStrategy}`;
            
            if (taskActionPlan.goalHierarchy.phaseGoals.length > 0) {
                filled += '\n\nGoal Hierarchy:';
                taskActionPlan.goalHierarchy.phaseGoals.forEach(phase => {
                    filled += `\n- ${phase.phase}: ${phase.goal} (Steps ${phase.steps.join(', ')})`;
                });
            }
        }
        
        // Add detail/breakdown recommendations if needed - fill with actual steps from action plan
        if (detailRecommendation && (detailRecommendation.needsMoreDetail || detailRecommendation.needsBreakdown)) {
            filled += '\n\n[RECOMMENDED: Add More Detail and Break Down]:';
            if (detailRecommendation.needsBreakdown && taskActionPlan && taskActionPlan.steps.length > 0) {
                filled += '\n\nBreak this task into detailed steps (agent-generated based on ML analysis and chat history):';
                taskActionPlan.steps.forEach((step, index) => {
                    filled += `\n\n${step.stepNumber}. ${step.action}`;
                    filled += `\n   Goal: ${step.goal}`;
                    if (step.subGoals && step.subGoals.length > 0) {
                        filled += `\n   Sub-goals:`;
                        step.subGoals.forEach(sg => filled += `\n     - ${sg}`);
                    }
                    filled += `\n   Description: ${step.description}`;
                    filled += `\n   Why: ${step.why}`;
                    if (step.successCriteria && step.successCriteria.length > 0) {
                        filled += `\n   Success Criteria:`;
                        step.successCriteria.forEach(sc => filled += `\n     ✓ ${sc}`);
                    }
                    if (step.estimatedTime) {
                        filled += `\n   Estimated Time: ${step.estimatedTime}`;
                    }
                    if (step.requiredContext && step.requiredContext.length > 0) {
                        filled += `\n   Required Context: ${step.requiredContext.join(', ')}`;
                    }
                    if (step.dependencies && step.dependencies.length > 0) {
                        filled += `\n   Dependencies: ${step.dependencies.join(', ')}`;
                    }
                });
                
                // Add history context if available
                if (taskActionPlan.basedOnHistory && taskActionPlan.historyPattern) {
                    filled += `\n\nNote: ${taskActionPlan.historyPattern}`;
                }
                
                // Add reference-based steps if applicable
                if (referenceAssessment && referenceAssessment.processStepsRequiringRefs.length > 0) {
                    filled += '\n\nAdditional steps requiring references:';
                    referenceAssessment.processStepsRequiringRefs.forEach((refStep, index) => {
                        filled += `\n- ${refStep}`;
                    });
                }
            } else if (detailRecommendation.needsBreakdown) {
                // Fallback to template if no action plan available
                filled += '\n\nBreak this task into detailed steps:';
                filled += '\n1. [Step 1 - What exactly needs to happen?]';
                filled += '\n2. [Step 2 - What exactly needs to happen?]';
                filled += '\n3. [Step 3 - What exactly needs to happen?]';
                filled += '\n\nFor each step, specify:';
                filled += '\n- What: The specific action or change';
                filled += '\n- Where: The location or component';
                filled += '\n- How: The method or approach';
                filled += '\n- Why: The reason or benefit';
            }
            if (detailRecommendation.needsMoreDetail) {
                filled += '\n\nAdd more specific details:';
                
                // Use action plan steps to suggest specific details
                if (taskActionPlan && taskActionPlan.steps.length > 0) {
                    filled += '\n\nBased on ML analysis, consider adding:';
                    taskActionPlan.steps.forEach(step => {
                        if (step.requiredContext && step.requiredContext.length > 0) {
                            filled += `\n- For "${step.action}": ${step.requiredContext.join(', ')}`;
                        }
                    });
                    
                    // Add prerequisites as details to include
                    if (taskActionPlan.prerequisites.length > 0) {
                        filled += '\n\nPrerequisites to have ready:';
                        taskActionPlan.prerequisites.forEach(prereq => {
                            filled += `\n- ${prereq}`;
                        });
                    }
                } else {
                    // Fallback template
                    filled += '\n- Include exact names, numbers, or values';
                    filled += '\n- Provide concrete examples';
                    filled += '\n- Specify constraints or requirements';
                    filled += '\n- Describe expected outcomes';
                }
            }
        } else if (taskActionPlan && taskActionPlan.steps.length > 0 && (prompt.length < 100 || (analysis && analysis.score < 70))) {
            // Even if not flagged for breakdown, add steps if prompt is short or low quality
            filled += '\n\n[SUGGESTED TASK BREAKDOWN - Based on ML Analysis]:';
            filled += `\nComplexity: ${taskActionPlan.complexity.toUpperCase()}`;
            if (taskActionPlan.totalEstimatedTime) {
                filled += ` | Estimated Time: ${taskActionPlan.totalEstimatedTime}`;
            }
            if (taskActionPlan.basedOnHistory && taskActionPlan.historyPattern) {
                filled += `\n${taskActionPlan.historyPattern}`;
            }
            filled += '\n\nRecommended steps:';
            taskActionPlan.steps.forEach(step => {
                filled += `\n${step.stepNumber}. ${step.action}: ${step.description}`;
            });
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

    private createIterationStrategy(prompt: string, goal: string, score: number, referenceAssessment?: ReferenceAssessment, detailRecommendation?: any): IterationStrategy {
        const steps: string[] = [];
        let expectedIterations = 1;
        
        // If critical references are needed, add steps to gather them
        if (referenceAssessment && referenceAssessment.criticalReferencesNeeded.length > 0) {
            steps.push('1. Gather required references (CRITICAL for accuracy)');
            steps.push('2. Review references to understand requirements');
            steps.push('3. Start with initial prompt + references');
            expectedIterations += 1;
        } else {
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
        }
        
        const refinementPoints: string[] = [];
        
        // Add detail/breakdown refinement points
        if (detailRecommendation) {
            if (detailRecommendation.needsBreakdown) {
                refinementPoints.push('Break down complex parts into smaller steps');
                refinementPoints.push('Add more detail to each step');
            }
            if (detailRecommendation.needsMoreDetail) {
                refinementPoints.push('Add more specific details and examples');
                refinementPoints.push('Specify exact requirements and constraints');
            }
        }
        
        if (referenceAssessment && referenceAssessment.criticalReferencesNeeded.length > 0) {
            refinementPoints.push('Verify solution matches reference specifications');
            refinementPoints.push('Cross-check API usage with documentation');
        }
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

    private generateTaskActionPlan(prompt: string, goal: string, analysis: PromptAnalysis, chatHistory?: ChatHistoryContext, referenceAssessment?: ReferenceAssessment): TaskActionPlan {
        const lower = prompt.toLowerCase();
        const steps: TaskActionStep[] = [];
        const prerequisites: string[] = [];
        let complexity: 'simple' | 'medium' | 'complex' = 'medium';
        let basedOnHistory = false;
        let historyPattern: string | undefined;
        
        // Define overall goal and planning strategy
        const overallGoal = `${goal.replace(/_/g, ' ')}: ${prompt.substring(0, 100)}${prompt.length > 100 ? '...' : ''}`;
        const planningStrategy = `Sequential execution with goal decomposition for ${goal.replace(/_/g, ' ')}`;
        
        // Check chat history for similar tasks
        if (chatHistory && chatHistory.recentMessages.length > 0) {
            const similarGoals = chatHistory.recentMessages.filter(msg => 
                msg.goal === goal && msg.prompt.toLowerCase().includes(lower.split(' ').slice(0, 3).join(' '))
            );
            
            if (similarGoals.length > 0) {
                basedOnHistory = true;
                historyPattern = `Based on ${similarGoals.length} similar task(s) in chat history`;
            }
        }
        
        // Generate steps based on goal type
        switch (goal) {
            case 'code_generation':
                complexity = this.assessTaskComplexity(prompt, analysis);
                steps.push(...this.generateCodeGenerationSteps(prompt, lower, analysis, referenceAssessment, overallGoal));
                prerequisites.push('Code editor/IDE ready', 'Required dependencies installed');
                if (referenceAssessment && referenceAssessment.criticalReferencesNeeded.length > 0) {
                    prerequisites.push('API documentation available', 'Library references ready');
                }
                break;
                
            case 'debugging':
                complexity = 'medium';
                steps.push(...this.generateDebuggingSteps(prompt, lower, analysis, chatHistory, overallGoal));
                prerequisites.push('Error reproduction steps', 'Access to logs/debugger');
                break;
                
            case 'refactoring':
                complexity = this.assessTaskComplexity(prompt, analysis);
                steps.push(...this.generateRefactoringSteps(prompt, lower, analysis, overallGoal));
                prerequisites.push('Original code available', 'Test suite ready');
                break;
                
            case 'code_review':
                complexity = 'medium';
                steps.push(...this.generateCodeReviewSteps(prompt, lower, analysis, overallGoal));
                prerequisites.push('Code to review', 'Coding standards/guidelines');
                break;
                
            case 'test_generation':
                complexity = 'medium';
                steps.push(...this.generateTestGenerationSteps(prompt, lower, analysis, overallGoal));
                prerequisites.push('Code to test', 'Testing framework installed');
                break;
                
            case 'explanation':
                complexity = 'simple';
                steps.push(...this.generateExplanationSteps(prompt, lower, analysis, overallGoal));
                prerequisites.push('Topic/concept identified', 'Target audience level known');
                break;
                
            case 'transformation':
                complexity = 'medium';
                steps.push(...this.generateTransformationSteps(prompt, lower, analysis, overallGoal));
                prerequisites.push('Source code/data available', 'Target format specified');
                break;
                
            default:
                steps.push(...this.generateGenericSteps(prompt, goal, analysis, overallGoal));
                complexity = 'medium';
        }
        
        // Estimate total time based on complexity and step count
        const totalTime = this.estimateTotalTime(complexity, steps.length);
        
        // AGENTIC: Build goal hierarchy showing how goals relate
        const goalHierarchy = this.buildGoalHierarchy(overallGoal, steps);
        
        return {
            goal,
            overallGoal,
            planningStrategy,
            steps,
            totalEstimatedTime: totalTime,
            complexity,
            prerequisites,
            basedOnHistory,
            historyPattern,
            goalHierarchy
        };
    }
    
    private determineOverallGoal(prompt: string, goal: string, analysis: PromptAnalysis): string {
        // Agentic reasoning: Extract the true objective from the prompt
        const lower = prompt.toLowerCase();
        
        // Look for explicit goal statements
        const goalPatterns = [
            /(?:i want to|i need to|goal is|objective is|trying to|aiming to|purpose is)\s+(.+?)(?:\.|$)/i,
            /(?:create|build|make|develop|implement|design)\s+(.+?)(?:that|which|with|for)/i,
            /(?:help me|assist with|need help|want)\s+(.+?)(?:\.|$)/i
        ];
        
        for (const pattern of goalPatterns) {
            const match = prompt.match(pattern);
            if (match && match[1]) {
                return match[1].trim();
            }
        }
        
        // Fallback: Generate goal from task type and key phrases
        const keyPhrases = prompt.split(/[.!?]/).slice(0, 2).join(' ').trim();
        if (keyPhrases.length > 10 && keyPhrases.length < 200) {
            return keyPhrases;
        }
        
        // Default based on goal type
        const goalDescriptions: Record<string, string> = {
            'code_generation': 'Generate functional, well-structured code that meets the specified requirements',
            'debugging': 'Identify and fix the root cause of the issue',
            'refactoring': 'Improve code quality while maintaining functionality',
            'code_review': 'Review code for quality, security, and best practices',
            'test_generation': 'Create comprehensive tests that verify correctness',
            'explanation': 'Provide clear, understandable explanation of the concept',
            'transformation': 'Transform code/data from source format to target format accurately'
        };
        
        return goalDescriptions[goal] || 'Complete the requested task effectively';
    }
    
    private planStrategy(prompt: string, goal: string, analysis: PromptAnalysis, chatHistory?: ChatHistoryContext, referenceAssessment?: ReferenceAssessment): string {
        // Agentic planning: Determine the best approach based on multiple factors
        const factors: string[] = [];
        
        // Factor 1: Complexity
        const complexity = this.assessTaskComplexity(prompt, analysis);
        if (complexity === 'complex') {
            factors.push('incremental development with frequent validation');
        } else if (complexity === 'simple') {
            factors.push('direct implementation approach');
        } else {
            factors.push('structured approach with clear phases');
        }
        
        // Factor 2: Reference requirements
        if (referenceAssessment && referenceAssessment.criticalReferencesNeeded.length > 0) {
            factors.push('reference-driven development');
            factors.push('verify against documentation at each step');
        }
        
        // Factor 3: Chat history patterns
        if (chatHistory && chatHistory.patterns.length > 0) {
            if (chatHistory.patterns.some(p => p.includes('Frequent goal'))) {
                factors.push('leverage patterns from previous similar tasks');
            }
        }
        
        // Factor 4: Prompt quality
        if (analysis.score < 60) {
            factors.push('iterative refinement with clarification');
        } else if (analysis.score >= 80) {
            factors.push('confident execution based on clear requirements');
        }
        
        // Factor 5: Goal-specific strategies
        if (goal === 'code_generation') {
            factors.push('test-driven development if tests mentioned');
        } else if (goal === 'debugging') {
            factors.push('systematic isolation and root cause analysis');
        } else if (goal === 'refactoring') {
            factors.push('incremental changes with test verification');
        }
        
        return factors.length > 0 
            ? `Strategy: ${factors.join(', ')}.`
            : 'Strategy: Standard approach with clear milestones.';
    }
    
    private buildGoalHierarchy(overallGoal: string, steps: TaskActionStep[]): { overall: string; phaseGoals: { phase: string; goal: string; steps: number[] }[] } {
        // Organize steps into phases with phase-level goals
        const phaseGoals: { phase: string; goal: string; steps: number[] }[] = [];
        
        // Group steps into logical phases
        if (steps.length >= 3) {
            // Phase 1: Planning/Preparation (first 1-2 steps)
            const planningSteps = steps.slice(0, Math.min(2, Math.floor(steps.length / 3)));
            if (planningSteps.length > 0) {
                phaseGoals.push({
                    phase: 'Planning & Preparation',
                    goal: 'Understand requirements and prepare necessary resources',
                    steps: planningSteps.map(s => s.stepNumber)
                });
            }
            
            // Phase 2: Implementation (middle steps)
            const midStart = planningSteps.length;
            const midEnd = Math.min(midStart + Math.floor(steps.length / 2), steps.length - 1);
            if (midEnd > midStart) {
                const implementationSteps = steps.slice(midStart, midEnd);
                phaseGoals.push({
                    phase: 'Implementation',
                    goal: 'Execute the core work to achieve the objective',
                    steps: implementationSteps.map(s => s.stepNumber)
                });
            }
            
            // Phase 3: Validation/Refinement (last steps)
            if (midEnd < steps.length) {
                const validationSteps = steps.slice(midEnd);
                phaseGoals.push({
                    phase: 'Validation & Refinement',
                    goal: 'Verify correctness and improve quality',
                    steps: validationSteps.map(s => s.stepNumber)
                });
            }
        }
        
        return {
            overall: overallGoal,
            phaseGoals
        };
    }
    
    private assessTaskComplexity(prompt: string, analysis: PromptAnalysis): 'simple' | 'medium' | 'complex' {
        const wordCount = prompt.split(/\s+/).length;
        const hasMultipleComponents = /(and|also|plus|additionally|multiple|several)/i.test(prompt);
        const hasComplexLogic = /(algorithm|optimize|performance|scalability|distributed|concurrent|async)/i.test(prompt);
        
        if (wordCount > 200 || hasMultipleComponents && hasComplexLogic) {
            return 'complex';
        } else if (wordCount > 100 || hasMultipleComponents || hasComplexLogic) {
            return 'medium';
        }
        return 'simple';
    }
    
    private generateStepGoal(action: string, stepNum: number, totalSteps: number, overallGoal: string, goal: string): string {
        // Agentic goal generation for each step
        const goalTemplates: Record<string, string[]> = {
            'Analyze Requirements': [
                'Establish clear understanding of what needs to be built and all constraints',
                'Document all requirements, inputs, outputs, and edge cases',
                'Create a complete specification for the task'
            ],
            'Design Architecture': [
                'Create a scalable, maintainable system design',
                'Plan component structure and interactions',
                'Design the system architecture that supports requirements'
            ],
            'Gather Required References': [
                'Collect all necessary documentation and references',
                'Ensure access to correct API specifications and examples',
                'Prepare reference materials for accurate implementation'
            ],
            'Implement Core Logic': [
                'Implement working solution that meets core requirements',
                'Write the main functionality that achieves the primary goal',
                'Create the core implementation that solves the problem'
            ],
            'Add Error Handling': [
                'Make the solution robust and production-ready',
                'Ensure graceful handling of edge cases and errors',
                'Add comprehensive error handling for reliability'
            ],
            'Write Unit Tests': [
                'Verify correctness and prevent regressions',
                'Ensure the solution works as expected',
                'Create tests that validate functionality'
            ],
            'Review and Refine': [
                'Ensure code quality and best practices',
                'Improve the solution to production quality',
                'Polish and optimize the final implementation'
            ]
        };
        
        return goalTemplates[action]?.[0] || `Complete ${action.toLowerCase()} to progress toward: ${overallGoal}`;
    }
    
    private generateCodeGenerationSteps(prompt: string, lower: string, analysis: PromptAnalysis, referenceAssessment?: ReferenceAssessment, overallGoal?: string): TaskActionStep[] {
        const steps: TaskActionStep[] = [];
        let stepNum = 1;
        // Calculate total steps dynamically
        let estimatedTotalSteps = 4; // Base: Analyze, Design (if complex), Implement, Review
        if (this.assessTaskComplexity(prompt, analysis) !== 'simple') estimatedTotalSteps++;
        if (lower.includes('error') || lower.includes('exception') || lower.includes('handle')) estimatedTotalSteps++;
        if (lower.includes('test') || this.assessTaskComplexity(prompt, analysis) === 'complex') estimatedTotalSteps++;
        if (referenceAssessment && referenceAssessment.criticalReferencesNeeded.length > 0) estimatedTotalSteps++;
        const totalSteps = estimatedTotalSteps;
        
        const stepGoal1 = this.generateStepGoal('Analyze Requirements', 1, totalSteps, overallGoal || 'Complete the task', 'code_generation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Analyze Requirements',
            description: 'Break down the prompt to identify: what needs to be created, inputs/outputs, constraints, and edge cases',
            why: 'Understanding requirements prevents scope creep and ensures all needs are met',
            goal: stepGoal1,
            estimatedTime: '5-10 min',
            requiredContext: ['Full prompt text', 'Any examples provided']
        });
        
        if (this.assessTaskComplexity(prompt, analysis) !== 'simple') {
            const stepGoal2 = this.generateStepGoal('Design Architecture', stepNum, totalSteps, overallGoal || 'Complete the task', 'code_generation');
            steps.push({
                stepNumber: stepNum++,
                action: 'Design Architecture',
                description: 'Plan the structure: classes, functions, data flow, and relationships between components',
                why: 'Good architecture makes code maintainable and easier to implement',
                goal: stepGoal2,
                subGoals: [
                    'Define component structure',
                    'Establish data flow patterns',
                    'Plan component interactions',
                    'Ensure separation of concerns'
                ],
                estimatedTime: '10-15 min',
                dependencies: ['Step 1 completed'],
                successCriteria: [
                    'Architecture diagram or structure is defined',
                    'Component responsibilities are clear',
                    'Data flow is mapped',
                    'Design supports requirements'
                ]
            });
        }
        
        if (referenceAssessment && referenceAssessment.criticalReferencesNeeded.length > 0) {
            const stepGoalRef = this.generateStepGoal('Gather Required References', stepNum, totalSteps, overallGoal || 'Complete the task', 'code_generation');
            steps.push({
                stepNumber: stepNum++,
                action: 'Gather Required References',
                goal: stepGoalRef,
                description: `Collect: ${referenceAssessment.criticalReferencesNeeded.map(r => r.type).join(', ')}`,
                why: 'References ensure correct API usage, patterns, and best practices',
                estimatedTime: '5-10 min',
                requiredContext: referenceAssessment.criticalReferencesNeeded.map(r => r.description)
            });
        }
        
        const stepGoal4 = this.generateStepGoal('Implement Core Logic', stepNum, totalSteps, overallGoal || 'Complete the task', 'code_generation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Implement Core Logic',
            goal: stepGoal4,
            description: 'Write the main functionality: data structures, algorithms, and business logic',
            why: 'Core logic is the foundation - get it right before adding features',
            estimatedTime: this.assessTaskComplexity(prompt, analysis) === 'complex' ? '30-60 min' : '15-30 min',
            dependencies: referenceAssessment && referenceAssessment.criticalReferencesNeeded.length > 0 
                ? [`Step ${stepNum - 1} completed`] 
                : ['Step 1 completed']
        });
        
        if (lower.includes('error') || lower.includes('exception') || lower.includes('handle')) {
            const stepGoal5 = this.generateStepGoal('Add Error Handling', stepNum, totalSteps, overallGoal || 'Complete the task', 'code_generation');
            steps.push({
                stepNumber: stepNum++,
                action: 'Add Error Handling',
                goal: stepGoal5,
                description: 'Implement try-catch blocks, validation, and graceful error messages',
                why: 'Robust error handling prevents crashes and improves user experience',
                estimatedTime: '10-15 min',
                dependencies: [`Step ${stepNum - 1} completed`]
            });
        }
        
        if (lower.includes('test') || this.assessTaskComplexity(prompt, analysis) === 'complex') {
            const stepGoal6 = this.generateStepGoal('Write Unit Tests', stepNum, totalSteps, overallGoal || 'Complete the task', 'code_generation');
            steps.push({
                stepNumber: stepNum++,
                action: 'Write Unit Tests',
                goal: stepGoal6,
                description: 'Create tests for core functionality, edge cases, and error scenarios',
                why: 'Tests verify correctness and prevent regressions',
                estimatedTime: '15-20 min',
                dependencies: [`Step ${stepNum - 2} completed`]
            });
        }
        
        const stepGoalFinal = this.generateStepGoal('Review and Refine', stepNum, totalSteps, overallGoal || 'Complete the task', 'code_generation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Review and Refine',
            goal: stepGoalFinal,
            description: 'Check code quality: readability, performance, documentation, and adherence to best practices',
            why: 'Review catches issues early and improves code quality',
            estimatedTime: '10-15 min',
            dependencies: [`Step ${stepNum - 1} completed`]
        });
        
        return steps;
    }
    
    private generateDebuggingSteps(prompt: string, lower: string, analysis: PromptAnalysis, chatHistory?: ChatHistoryContext, overallGoal?: string): TaskActionStep[] {
        const steps: TaskActionStep[] = [];
        let stepNum = 1;
        const totalSteps = 6;
        
        const stepGoal1 = this.generateStepGoal('Reproduce the Error', 1, totalSteps, overallGoal || 'Fix the issue', 'debugging');
        steps.push({
            stepNumber: stepNum++,
            action: 'Reproduce the Error',
            goal: stepGoal1,
            description: 'Run the code and confirm you can consistently reproduce the issue',
            why: 'Reproducibility is essential for effective debugging',
            estimatedTime: '5-10 min',
            requiredContext: ['Error message', 'Steps to reproduce']
        });
        
        const stepGoal2 = this.generateStepGoal('Gather Error Details', 2, 6, overallGoal || 'Fix the issue', 'debugging');
        steps.push({
            stepNumber: stepNum++,
            action: 'Gather Error Details',
            description: 'Collect: full error message, stack trace, input data, and environment details',
            why: 'Complete error information helps identify root cause faster',
            goal: stepGoal2,
            estimatedTime: '5 min',
            dependencies: ['Step 1 completed']
        });
        
        const stepGoal3 = this.generateStepGoal('Isolate the Problem', 3, 6, overallGoal || 'Fix the issue', 'debugging');
        steps.push({
            stepNumber: stepNum++,
            action: 'Isolate the Problem',
            description: 'Narrow down: comment out code sections, add print statements, or use debugger to find the failing line',
            why: 'Isolation helps focus on the actual problem area',
            goal: stepGoal3,
            estimatedTime: '10-20 min',
            dependencies: ['Step 2 completed']
        });
        
        const stepGoal4 = this.generateStepGoal('Analyze Root Cause', 4, 6, overallGoal || 'Fix the issue', 'debugging');
        steps.push({
            stepNumber: stepNum++,
            action: 'Analyze Root Cause',
            description: 'Understand why it fails: check variable values, data types, logic flow, and edge cases',
            why: 'Understanding root cause leads to proper fix, not just symptom treatment',
            goal: stepGoal4,
            estimatedTime: '10-15 min',
            dependencies: ['Step 3 completed']
        });
        
        const stepGoal5 = this.generateStepGoal('Implement Fix', 5, 6, overallGoal || 'Fix the issue', 'debugging');
        steps.push({
            stepNumber: stepNum++,
            action: 'Implement Fix',
            description: 'Apply the solution: fix the bug, handle edge cases, and ensure logic is correct',
            why: 'The fix should address root cause, not just symptoms',
            goal: stepGoal5,
            estimatedTime: '10-20 min',
            dependencies: ['Step 4 completed']
        });
        
        const stepGoal6 = this.generateStepGoal('Verify Fix', 6, 6, overallGoal || 'Fix the issue', 'debugging');
        steps.push({
            stepNumber: stepNum++,
            action: 'Verify Fix',
            description: 'Test: reproduce original error (should be fixed), test edge cases, check for regressions',
            why: 'Verification ensures fix works and doesn\'t break other functionality',
            goal: stepGoal6,
            estimatedTime: '10 min',
            dependencies: ['Step 5 completed']
        });
        
        return steps;
    }
    
    private generateRefactoringSteps(prompt: string, lower: string, analysis: PromptAnalysis, overallGoal?: string): TaskActionStep[] {
        const steps: TaskActionStep[] = [];
        let stepNum = 1;
        const totalSteps = 5;
        
        const stepGoal1 = this.generateStepGoal('Understand Current Code', 1, totalSteps, overallGoal || 'Refactor the code', 'refactoring');
        steps.push({
            stepNumber: stepNum++,
            action: 'Understand Current Code',
            goal: stepGoal1,
            description: 'Read and analyze existing code: structure, dependencies, and current behavior',
            why: 'Understanding prevents breaking changes during refactoring',
            estimatedTime: '10-15 min',
            requiredContext: ['Original code']
        });
        
        const stepGoal2 = this.generateStepGoal('Run Existing Tests', 2, totalSteps, overallGoal || 'Refactor the code', 'refactoring');
        steps.push({
            stepNumber: stepNum++,
            action: 'Run Existing Tests',
            goal: stepGoal2,
            description: 'Ensure all tests pass before refactoring to establish baseline',
            why: 'Tests verify behavior hasn\'t changed after refactoring',
            estimatedTime: '5 min',
            dependencies: ['Step 1 completed']
        });
        
        const stepGoal3 = this.generateStepGoal('Identify Refactoring Targets', 3, totalSteps, overallGoal || 'Refactor the code', 'refactoring');
        steps.push({
            stepNumber: stepNum++,
            action: 'Identify Refactoring Targets',
            goal: stepGoal3,
            description: 'Find: duplicated code, long functions, complex logic, or design issues',
            why: 'Targeted refactoring is more effective than random changes',
            estimatedTime: '10 min',
            dependencies: ['Step 2 completed']
        });
        
        const stepGoal4 = this.generateStepGoal('Apply Refactoring', 4, totalSteps, overallGoal || 'Refactor the code', 'refactoring');
        steps.push({
            stepNumber: stepNum++,
            action: 'Apply Refactoring',
            goal: stepGoal4,
            description: 'Make changes incrementally: extract functions, rename variables, simplify logic',
            why: 'Incremental changes are safer and easier to verify',
            estimatedTime: '20-30 min',
            dependencies: ['Step 3 completed']
        });
        
        const stepGoal5 = this.generateStepGoal('Verify Tests Still Pass', 5, totalSteps, overallGoal || 'Refactor the code', 'refactoring');
        steps.push({
            stepNumber: stepNum++,
            action: 'Verify Tests Still Pass',
            goal: stepGoal5,
            description: 'Run tests after each refactoring step to ensure behavior is preserved',
            why: 'Continuous verification catches issues early',
            estimatedTime: '5 min',
            dependencies: ['Step 4 completed']
        });
        
        return steps;
    }
    
    private generateCodeReviewSteps(prompt: string, lower: string, analysis: PromptAnalysis, overallGoal?: string): TaskActionStep[] {
        const steps: TaskActionStep[] = [];
        let stepNum = 1;
        const totalSteps = 5;
        
        const stepGoal1 = this.generateStepGoal('Initial Read-Through', 1, totalSteps, overallGoal || 'Review the code', 'code_review');
        steps.push({
            stepNumber: stepNum++,
            action: 'Initial Read-Through',
            goal: stepGoal1,
            description: 'Read the code once to understand what it does and its overall structure',
            why: 'First pass gives context for detailed review',
            estimatedTime: '5-10 min',
            requiredContext: ['Code to review']
        });
        
        const stepGoal2 = this.generateStepGoal('Check Functionality', 2, totalSteps, overallGoal || 'Review the code', 'code_review');
        steps.push({
            stepNumber: stepNum++,
            action: 'Check Functionality',
            goal: stepGoal2,
            description: 'Verify logic correctness: algorithms, edge cases, and expected behavior',
            why: 'Functional correctness is the primary concern',
            estimatedTime: '10-15 min',
            dependencies: ['Step 1 completed']
        });
        
        const stepGoal3 = this.generateStepGoal('Review Code Quality', 3, totalSteps, overallGoal || 'Review the code', 'code_review');
        steps.push({
            stepNumber: stepNum++,
            action: 'Review Code Quality',
            goal: stepGoal3,
            description: 'Check: readability, naming, structure, and adherence to coding standards',
            why: 'Code quality affects maintainability',
            estimatedTime: '10 min',
            dependencies: ['Step 2 completed']
        });
        
        const stepGoal4 = this.generateStepGoal('Check Security & Performance', 4, totalSteps, overallGoal || 'Review the code', 'code_review');
        steps.push({
            stepNumber: stepNum++,
            action: 'Check Security & Performance',
            goal: stepGoal4,
            description: 'Look for: security vulnerabilities, performance issues, and resource leaks',
            why: 'Security and performance are critical for production code',
            estimatedTime: '10-15 min',
            dependencies: ['Step 3 completed']
        });
        
        const stepGoal5 = this.generateStepGoal('Provide Feedback', 5, totalSteps, overallGoal || 'Review the code', 'code_review');
        steps.push({
            stepNumber: stepNum++,
            action: 'Provide Feedback',
            goal: stepGoal5,
            description: 'Document: issues found, suggestions for improvement, and positive aspects',
            why: 'Constructive feedback helps improve code quality',
            estimatedTime: '10 min',
            dependencies: ['Step 4 completed']
        });
        
        return steps;
    }
    
    private generateTestGenerationSteps(prompt: string, lower: string, analysis: PromptAnalysis, overallGoal?: string): TaskActionStep[] {
        const steps: TaskActionStep[] = [];
        let stepNum = 1;
        const totalSteps = 4;
        
        const stepGoal1 = this.generateStepGoal('Understand Code Under Test', 1, totalSteps, overallGoal || 'Generate tests', 'test_generation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Understand Code Under Test',
            goal: stepGoal1,
            description: 'Analyze the code: inputs, outputs, edge cases, and expected behavior',
            why: 'Understanding ensures comprehensive test coverage',
            estimatedTime: '10 min',
            requiredContext: ['Code to test']
        });
        
        const stepGoal2 = this.generateStepGoal('Identify Test Cases', 2, totalSteps, overallGoal || 'Generate tests', 'test_generation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Identify Test Cases',
            goal: stepGoal2,
            description: 'List: happy path, edge cases, error scenarios, and boundary conditions',
            why: 'Comprehensive test cases ensure robust testing',
            estimatedTime: '10 min',
            dependencies: ['Step 1 completed']
        });
        
        const stepGoal3 = this.generateStepGoal('Write Unit Tests', 3, totalSteps, overallGoal || 'Generate tests', 'test_generation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Write Unit Tests',
            goal: stepGoal3,
            description: 'Implement tests: arrange-act-assert pattern, mock dependencies, test each function',
            why: 'Unit tests verify individual components work correctly',
            estimatedTime: '20-30 min',
            dependencies: ['Step 2 completed']
        });
        
        const stepGoal4 = this.generateStepGoal('Run Tests', 4, totalSteps, overallGoal || 'Generate tests', 'test_generation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Run Tests',
            goal: stepGoal4,
            description: 'Execute tests and verify they pass, fix any issues',
            why: 'Running tests validates they work correctly',
            estimatedTime: '5 min',
            dependencies: ['Step 3 completed']
        });
        
        return steps;
    }
    
    private generateExplanationSteps(prompt: string, lower: string, analysis: PromptAnalysis, overallGoal?: string): TaskActionStep[] {
        const steps: TaskActionStep[] = [];
        let stepNum = 1;
        const totalSteps = 3;
        
        const stepGoal1 = this.generateStepGoal('Identify Core Concepts', 1, totalSteps, overallGoal || 'Explain the concept', 'explanation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Identify Core Concepts',
            goal: stepGoal1,
            description: 'Extract key concepts, terminology, and fundamental ideas to explain',
            why: 'Core concepts form the foundation of understanding',
            estimatedTime: '5 min',
            requiredContext: ['Topic/concept to explain']
        });
        
        const stepGoal2 = this.generateStepGoal('Structure Explanation', 2, totalSteps, overallGoal || 'Explain the concept', 'explanation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Structure Explanation',
            goal: stepGoal2,
            description: 'Organize: introduction, main concepts, examples, and summary',
            why: 'Good structure improves comprehension',
            estimatedTime: '5 min',
            dependencies: ['Step 1 completed']
        });
        
        const stepGoal3 = this.generateStepGoal('Add Examples', 3, totalSteps, overallGoal || 'Explain the concept', 'explanation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Add Examples',
            goal: stepGoal3,
            description: 'Include: concrete examples, analogies, and practical use cases',
            why: 'Examples make abstract concepts concrete and understandable',
            estimatedTime: '10 min',
            dependencies: ['Step 2 completed']
        });
        
        return steps;
    }
    
    private generateTransformationSteps(prompt: string, lower: string, analysis: PromptAnalysis, overallGoal?: string): TaskActionStep[] {
        const steps: TaskActionStep[] = [];
        let stepNum = 1;
        const totalSteps = 5;
        
        const stepGoal1 = this.generateStepGoal('Analyze Source Format', 1, totalSteps, overallGoal || 'Transform the data', 'transformation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Analyze Source Format',
            goal: stepGoal1,
            description: 'Understand: structure, data types, and patterns in source code/data',
            why: 'Understanding source format is essential for accurate transformation',
            estimatedTime: '10 min',
            requiredContext: ['Source code/data']
        });
        
        const stepGoal2 = this.generateStepGoal('Define Target Format', 2, totalSteps, overallGoal || 'Transform the data', 'transformation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Define Target Format',
            goal: stepGoal2,
            description: 'Specify: target structure, required fields, and transformation rules',
            why: 'Clear target specification guides transformation',
            estimatedTime: '5 min',
            dependencies: ['Step 1 completed']
        });
        
        const stepGoal3 = this.generateStepGoal('Map Transformation Logic', 3, totalSteps, overallGoal || 'Transform the data', 'transformation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Map Transformation Logic',
            goal: stepGoal3,
            description: 'Create mapping: source fields to target fields, data conversions, and validations',
            why: 'Mapping ensures all data is correctly transformed',
            estimatedTime: '15 min',
            dependencies: ['Step 2 completed']
        });
        
        const stepGoal4 = this.generateStepGoal('Implement Transformation', 4, totalSteps, overallGoal || 'Transform the data', 'transformation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Implement Transformation',
            goal: stepGoal4,
            description: 'Write code to perform the transformation with error handling',
            why: 'Implementation executes the transformation logic',
            estimatedTime: '20-30 min',
            dependencies: ['Step 3 completed']
        });
        
        const stepGoal5 = this.generateStepGoal('Verify Output', 5, totalSteps, overallGoal || 'Transform the data', 'transformation');
        steps.push({
            stepNumber: stepNum++,
            action: 'Verify Output',
            goal: stepGoal5,
            description: 'Test transformation: validate output format, check data accuracy, test edge cases',
            why: 'Verification ensures transformation works correctly',
            estimatedTime: '10 min',
            dependencies: ['Step 4 completed']
        });
        
        return steps;
    }
    
    private generateGenericSteps(prompt: string, goal: string, analysis: PromptAnalysis, overallGoal?: string): TaskActionStep[] {
        const totalSteps = 4;
        return [
            {
                stepNumber: 1,
                action: 'Understand Requirements',
                goal: this.generateStepGoal('Understand Requirements', 1, totalSteps, overallGoal || 'Complete the task', goal),
                description: 'Analyze what needs to be accomplished',
                why: 'Clear understanding prevents mistakes',
                estimatedTime: '10 min'
            },
            {
                stepNumber: 2,
                action: 'Plan Approach',
                goal: this.generateStepGoal('Plan Approach', 2, totalSteps, overallGoal || 'Complete the task', goal),
                description: 'Determine the best method to achieve the goal',
                why: 'Planning improves efficiency',
                estimatedTime: '10 min',
                dependencies: ['Step 1 completed']
            },
            {
                stepNumber: 3,
                action: 'Execute',
                goal: this.generateStepGoal('Execute', 3, totalSteps, overallGoal || 'Complete the task', goal),
                description: 'Implement the planned approach',
                why: 'Execution achieves the goal',
                estimatedTime: '20-30 min',
                dependencies: ['Step 2 completed']
            },
            {
                stepNumber: 4,
                action: 'Verify Results',
                goal: this.generateStepGoal('Verify Results', 4, totalSteps, overallGoal || 'Complete the task', goal),
                description: 'Check that the outcome meets requirements',
                why: 'Verification ensures quality',
                estimatedTime: '10 min',
                dependencies: ['Step 3 completed']
            }
        ];
    }
    
    private estimateTotalTime(complexity: 'simple' | 'medium' | 'complex', stepCount: number): string {
        const baseTimePerStep = complexity === 'simple' ? 10 : complexity === 'medium' ? 15 : 25;
        const totalMinutes = stepCount * baseTimePerStep;
        
        if (totalMinutes < 60) {
            return `${totalMinutes} minutes`;
        } else {
            const hours = Math.floor(totalMinutes / 60);
            const minutes = totalMinutes % 60;
            return minutes > 0 ? `${hours}h ${minutes}m` : `${hours} hour${hours > 1 ? 's' : ''}`;
        }
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

    private calculateConfidence(analysis: PromptAnalysis, missingCount: number, contextCount: number, referenceAssessment?: ReferenceAssessment): number {
        let confidence = analysis.score;
        
        // Use reference assessment confidence if available
        if (referenceAssessment) {
            confidence = referenceAssessment.decisionConfidenceWithoutRefs;
        } else {
            // Fallback to old calculation
            // Reduce confidence for missing elements
            confidence -= missingCount * 5;
            
            // Reduce confidence if context is needed
            if (contextCount > 0) {
                confidence -= contextCount * 3;
            }
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



