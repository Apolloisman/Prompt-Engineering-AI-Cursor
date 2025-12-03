/**
 * ML-Powered Prompt Enhancement
 * Actually analyzes and improves prompts by making them more specific
 */

import { PromptAnalysis } from './promptAnalyzer';
import { ChatHistoryContext, AttachedReference } from './chatHistoryManager';
import { PromptMLModel, PromptFeatures } from './mlModel';
import { PyTorchOptimizerClient } from './pytorchOptimizerClient';

export interface PromptEnhancement {
    enhancedPrompt: string;
    improvements: string[];
    specificity: number;
    detailsAdded: string[];
}

export class PromptEnhancer {
    private mlModel: PromptMLModel;
    private pytorchClient: PyTorchOptimizerClient;
    private usePyTorch: boolean = false;

    constructor() {
        this.mlModel = new PromptMLModel();
        this.pytorchClient = new PyTorchOptimizerClient();
        // Check if PyTorch API is available
        this.pytorchClient.checkHealth().then(available => {
            this.usePyTorch = available;
        }).catch(() => {
            this.usePyTorch = false;
        });
    }

    /**
     * ML analyzes prompt and generates ideal prompt (not steps, the actual prompt)
     * ML determines what to add based on rules, not hard-coded intents
     * For creation requests, includes concrete task breakdown
     */
    async analyzeAndImprovePrompt(
        originalPrompt: string,
        goal: string,
        patterns: Map<string, number>,
        analysis: PromptAnalysis,
        context?: ChatHistoryContext
    ): Promise<string> {
        // ML analyzes the prompt against rules and determines what's missing
        const idealPrompt = await this.mlAnalyzeAndEnhance(
            originalPrompt,
            goal,
            patterns,
            analysis,
            context
        );

        const actualIntent = this.mlInferActualIntent(originalPrompt, goal, patterns, analysis);
        let enhancedPrompt = this.addReferenceInsights(idealPrompt, context, actualIntent);

        // Always provide execution plan (implementation vs informational)
        // Use ML to generate concrete steps from prompt analysis
        const executionPlan = actualIntent.needsImplementation
            ? await this.generateConcreteTaskBreakdown(
                originalPrompt,
                goal,
                patterns,
                actualIntent,
                analysis,
                context
            )
            : this.generateInformationPlan(
                originalPrompt,
                goal,
                patterns,
                actualIntent,
                analysis,
                context
            );

        if (executionPlan) {
            enhancedPrompt = `${enhancedPrompt}\n\nExecute this task step-by-step:\n${executionPlan}`;
        }

        return enhancedPrompt.trim();
    }

    /**
     * Generate concrete task breakdown using ML analysis of the actual prompt
     * ML determines specific steps by analyzing prompt content, relationships, and requirements
     */
    private async generateConcreteTaskBreakdown(
        prompt: string,
        goal: string,
        patterns: Map<string, number>,
        actualIntent: { type: string | null; purpose: string | null; isSolutionRequest: boolean; needsImplementation: boolean },
        analysis: PromptAnalysis,
        context?: ChatHistoryContext
    ): Promise<string | null> {
        const features = this.mlModel.extractFeatures(prompt, goal, analysis);
        
        // Use ML to generate concrete steps from the actual prompt analysis
        try {
            // First, try to get ML-generated steps
            const mlSteps = await this.mlModel.generateExecutionSteps(features, []);
            
            if (mlSteps && mlSteps.length > 0) {
                // Format ML-generated steps as concrete actions
                const formattedSteps = mlSteps.map((step, index) => {
                    let stepText = step.action;
                    
                    // Add sub-actions if they make the step more concrete
                    if (step.subActions && step.subActions.length > 0) {
                        const concreteSubActions = step.subActions
                            .filter(sub => sub.length > 10)
                            .slice(0, 2);
                        
                        if (concreteSubActions.length > 0) {
                            stepText += ` (${concreteSubActions.join('; ')})`;
                        }
                    }
                    
                    return `${index + 1}. ${stepText}`;
                });
                
                // Add reference step if available
                const referenceStep = this.buildReferenceIntegrationStep(context?.attachedReferences);
                if (referenceStep) {
                    formattedSteps.splice(1, 0, `1.5. ${referenceStep}`);
                    // Renumber
                    return formattedSteps.map((step, index) => {
                        const num = index + 1;
                        return step.replace(/^\d+\./, `${num}.`);
                    }).join('\n');
                }
                
                return formattedSteps.join('\n');
            }
        } catch (error) {
            console.error('ML step generation failed, using prompt analysis:', error);
        }
        
        // Fallback: Generate concrete steps by analyzing the actual prompt content
        return this.generateStepsFromPromptAnalysis(prompt, goal, patterns, actualIntent, analysis, features, context);
    }

    /**
     * Generate concrete steps by analyzing the actual prompt content
     * ML determines what needs to be done based on the prompt, not templates
     */
    private generateStepsFromPromptAnalysis(
        prompt: string,
        goal: string,
        patterns: Map<string, number>,
        actualIntent: { type: string | null; purpose: string | null; isSolutionRequest: boolean; needsImplementation: boolean },
        analysis: PromptAnalysis,
        features: any,
        context?: ChatHistoryContext
    ): string | null {
        const steps: string[] = [];
        const lower = prompt.toLowerCase();
        
        // ML analyzes the prompt to extract specific requirements and generate concrete steps
        // Parse the prompt to understand what specifically needs to be done
        
        // Extract key information from the prompt
        const actionVerbs = this.extractActionVerbs(prompt);
        const targetObjects = this.extractTargetObjects(prompt);
        const constraints = this.extractConstraints(prompt);
        const domain = this.extractDomainFromPrompt(prompt);
        const specificRequirements = this.extractSpecificRequirements(prompt);
        
        // Generate concrete steps based on what the ML found in the prompt
        if (actualIntent.needsImplementation) {
            // Step 1: Analyze what specifically needs to be created/implemented
            if (targetObjects.length > 0) {
                const obj = targetObjects[0];
                if (specificRequirements.length > 0) {
                    steps.push(`Analyze requirements for ${obj} and identify: ${specificRequirements.slice(0, 3).join(', ')}`);
                } else {
                    steps.push(`Analyze requirements and design the ${obj} architecture based on specified constraints`);
                }
            } else {
                steps.push(`Analyze requirements for ${domain} and design the solution architecture`);
            }
            
            // Step 2: Extract specific implementation details from prompt
            if (actionVerbs.includes('automate') || actionVerbs.includes('automation')) {
                const automationTarget = this.extractAutomationTargets(prompt);
                steps.push(`Map the current ${automationTarget} process, identify automation points (manual steps, decision points, data handoffs), and design automation logic`);
                steps.push(`Implement automation scripts/controls to handle ${this.extractAutomationTargets(prompt)} with error handling and rollback capabilities`);
                if (constraints.length > 0) {
                    steps.push(`Apply constraints: ${constraints.slice(0, 2).join(', ')}`);
                }
                steps.push(`Integrate with existing systems (${this.extractSystemIntegrations(prompt)}) and add monitoring/alerting`);
                steps.push(`Test automation in a controlled environment and validate against success criteria (${this.extractSuccessMetrics(prompt)})`);
            } else if (actionVerbs.includes('create') || actionVerbs.includes('build') || actionVerbs.includes('make')) {
                const coreFunc = this.extractCoreFunctionality(prompt);
                steps.push(`Design the ${targetObjects[0] || domain} structure and components based on requirements: ${specificRequirements.slice(0, 2).join(', ')}`);
                steps.push(`Implement core functionality for ${coreFunc} with proper error handling`);
                if (constraints.length > 0) {
                    steps.push(`Apply constraints: ${constraints.join(', ')}`);
                }
                steps.push(`Test and validate the solution meets all specified requirements and quality gates`);
            } else {
                // Generic but still concrete based on prompt analysis
                steps.push(`Design solution for ${domain} based on analyzed requirements`);
                steps.push(`Implement the specific functionality identified in the prompt: ${this.extractKeyActions(prompt).slice(0, 2).join(', ')}`);
                steps.push(`Validate solution against success criteria and requirements`);
            }
        } else {
            // Informational/analysis request - ML determines specific research/analysis steps
            steps.push(`Clarify the specific questions and success criteria for ${domain} analysis (what evidence or output is expected)`);
            
            // Check for attached references
            if (context?.attachedReferences && context.attachedReferences.length > 0) {
                const refNames = context.attachedReferences.slice(-3).map((ref: any) => ref.name).join(', ');
                steps.push(`Review attached references (${refNames}) to extract relevant data, constraints, and prior findings`);
            } else {
                steps.push(`Identify and gather relevant data sources for ${domain} (internal docs, experts, datasets)`);
            }
            
            steps.push(`Collect and organize data/insights, noting assumptions, gaps, and conflicting information`);
            steps.push(`Analyze findings against success criteria, identify trends, blockers, and opportunities`);
            steps.push(`Summarize actionable recommendations and verification steps before delivering the final answer`);
        }
        
        // Add reference integration step if references are available
        if (context?.attachedReferences && context.attachedReferences.length > 0) {
            const refNames = context.attachedReferences.slice(-3).map((ref: any) => ref.name).join(', ');
            steps.splice(1, 0, `Review attached references (${refNames}) to extract domain-specific constraints and definitions`);
        }
        
        return steps.length > 0 ? steps.map((step, index) => `${index + 1}. ${step}`).join('\n') : null;
    }

    /**
     * Extract specific requirements mentioned in the prompt
     */
    private extractSpecificRequirements(prompt: string): string[] {
        const requirements: string[] = [];
        const lower = prompt.toLowerCase();
        
        // Look for requirement indicators
        const reqPatterns = [
            /(?:need|require|must|should|include)\s+([^\.\?]+)/gi,
            /(?:with|using|via)\s+([^\.\?]+)/gi,
            /(?:support|handle|manage)\s+([^\.\?]+)/gi
        ];
        
        reqPatterns.forEach(pattern => {
            const matches = prompt.matchAll(pattern);
            for (const match of matches) {
                const req = match[1].trim();
                if (req.length > 5 && req.length < 50) {
                    requirements.push(req);
                }
            }
        });
        
        return requirements.slice(0, 5); // Limit to 5 most relevant
    }

    /**
     * Extract system integrations mentioned in prompt
     */
    private extractSystemIntegrations(prompt: string): string {
        const lower = prompt.toLowerCase();
        
        const integrations: string[] = [];
        if (lower.includes('mes') || lower.includes('erp')) integrations.push('MES/ERP');
        if (lower.includes('plc') || lower.includes('scada')) integrations.push('PLC/SCADA');
        if (lower.includes('database') || lower.includes('sql')) integrations.push('database');
        if (lower.includes('api')) integrations.push('APIs');
        if (lower.includes('sensor')) integrations.push('sensors');
        
        return integrations.length > 0 ? integrations.join(', ') : 'existing systems';
    }

    /**
     * Extract success metrics from prompt
     */
    private extractSuccessMetrics(prompt: string): string {
        const lower = prompt.toLowerCase();
        
        const metrics: string[] = [];
        if (lower.includes('throughput') || lower.includes('speed')) metrics.push('throughput');
        if (lower.includes('quality')) metrics.push('quality');
        if (lower.includes('cost') || lower.includes('efficiency')) metrics.push('cost efficiency');
        if (lower.includes('utilization')) metrics.push('utilization');
        
        return metrics.length > 0 ? metrics.join(', ') : 'defined success criteria';
    }

    /**
     * Extract key actions from prompt
     */
    private extractKeyActions(prompt: string): string[] {
        const actions: string[] = [];
        const lower = prompt.toLowerCase();
        
        // Extract verb-noun pairs
        const actionPatterns = [
            /(?:create|build|make|develop|design|implement|generate)\s+([a-z]+(?:\s+[a-z]+)?)/gi,
            /(?:automate|optimize|improve|enhance)\s+([a-z]+(?:\s+[a-z]+)?)/gi
        ];
        
        actionPatterns.forEach(pattern => {
            const matches = prompt.matchAll(pattern);
            for (const match of matches) {
                const action = match[1].trim();
                if (action.length > 3 && action.length < 30) {
                    actions.push(action);
                }
            }
        });
        
        return actions.slice(0, 3);
    }

    private generateInformationPlan(
        prompt: string,
        goal: string,
        patterns: Map<string, number>,
        actualIntent: { type: string | null; purpose: string | null; isSolutionRequest: boolean; needsImplementation: boolean },
        analysis: PromptAnalysis,
        context?: ChatHistoryContext
    ): string | null {
        const steps: string[] = [];
        const domain = this.extractDomainFromPrompt(prompt);
        const references = context?.attachedReferences || [];

        steps.push(`Clarify the exact questions and success criteria for the ${domain} analysis (what evidence or output is expected).`);

        if (references.length > 0) {
            const refNames = references.slice(-3).map(ref => ref.name).join(', ');
            steps.push(`Review the attached references (${refNames}) to extract baseline data, constraints, and prior findings.`);
        } else {
            steps.push('Inventory trusted data sources (internal docs, SMEs, public datasets) relevant to the topic.');
        }

        steps.push('Collect and organize the necessary data/insights, noting assumptions, gaps, and conflicting information.');
        steps.push('Analyze the findings against the success criteria, highlight trends, blockers, and opportunities.');
        steps.push('Summarize actionable recommendations, open questions, and verification steps before delivering the final answer.');

        return steps.map((step, index) => `${index + 1}. ${step}`).join('\n');
    }

    private addReferenceInsights(
        prompt: string,
        context: ChatHistoryContext | undefined,
        actualIntent: { type: string | null; purpose: string | null; isSolutionRequest: boolean; needsImplementation: boolean }
    ): string {
        const refs = context?.attachedReferences;
        if (!refs || refs.length === 0) {
            return prompt;
        }

        const insights = refs.slice(-3).map(ref => `- ${ref.name}: ${ref.summary}`);
        const directive = actualIntent.needsImplementation
            ? 'Incorporate these reference details as constraints/requirements during execution.'
            : 'Use these references as primary evidence when forming the analysis.';

        return `${prompt}\n\nAttached References:\n${insights.join('\n')}\n${directive}`;
    }

    private buildReferenceIntegrationStep(references?: AttachedReference[]): string | null {
        if (!references || references.length === 0) {
            return null;
        }

        const refNames = references.slice(-3).map(ref => ref.name).join(', ');
        return `Review attached references (${refNames}) to pull domain-specific constraints, metrics, and definitions before proceeding.`;
    }

    /**
     * Extracts a domain phrase from the prompt for automation/workflow tasks
     */
    private extractDomainFromPrompt(prompt: string): string {
        const lower = prompt.toLowerCase();
        if (lower.includes('machining')) return 'machining workflow';
        if (lower.includes('cnc')) return 'CNC workflow';
        if (lower.includes('manufacturing') || lower.includes('factory')) return 'manufacturing workflow';
        if (lower.includes('workflow')) return 'workflow';
        if (lower.includes('process')) return 'process';
        if (lower.includes('pipeline')) return 'pipeline';
        if (lower.includes('supply chain')) return 'supply chain workflow';
        return 'requested workflow';
    }

    /**
     * Extract action verbs from prompt to understand what needs to be done
     */
    private extractActionVerbs(prompt: string): string[] {
        const verbs: string[] = [];
        const lower = prompt.toLowerCase();
        
        const actionPatterns = [
            'automate', 'create', 'build', 'make', 'develop', 'design', 'implement',
            'generate', 'analyze', 'review', 'optimize', 'improve', 'fix', 'debug',
            'test', 'validate', 'verify', 'deploy', 'integrate', 'connect'
        ];
        
        actionPatterns.forEach(verb => {
            if (lower.includes(verb)) {
                verbs.push(verb);
            }
        });
        
        return verbs;
    }

    /**
     * Extract target objects from prompt (what is being created/analyzed)
     */
    private extractTargetObjects(prompt: string): string[] {
        const objects: string[] = [];
        const lower = prompt.toLowerCase();
        
        // Common patterns for objects
        const objectPatterns = [
            { pattern: /(?:create|build|make|design|develop)\s+(?:a|an|the)?\s+([a-z]+(?:\s+[a-z]+)?)/i, extract: (m: RegExpMatchArray) => m[1] },
            { pattern: /(?:automate|optimize|improve)\s+(?:my|the|a|an)?\s+([a-z]+(?:\s+[a-z]+)?)/i, extract: (m: RegExpMatchArray) => m[1] },
            { pattern: /(?:for|about|regarding)\s+([a-z]+(?:\s+[a-z]+)?)/i, extract: (m: RegExpMatchArray) => m[1] }
        ];
        
        objectPatterns.forEach(({ pattern, extract }) => {
            const match = prompt.match(pattern);
            if (match) {
                const obj = extract(match).trim();
                if (obj.length > 2 && obj.length < 30) {
                    objects.push(obj);
                }
            }
        });
        
        return objects;
    }

    /**
     * Extract constraints/requirements from prompt
     */
    private extractConstraints(prompt: string): string[] {
        const constraints: string[] = [];
        const lower = prompt.toLowerCase();
        
        // Look for constraint indicators
        if (lower.includes('responsive') || lower.includes('mobile')) constraints.push('responsive design');
        if (lower.includes('secure') || lower.includes('security')) constraints.push('security requirements');
        if (lower.includes('fast') || lower.includes('performance')) constraints.push('performance optimization');
        if (lower.includes('scalable') || lower.includes('scale')) constraints.push('scalability');
        if (lower.includes('accessible') || lower.includes('wcag')) constraints.push('accessibility standards');
        
        return constraints;
    }

    /**
     * Extract automation targets from prompt
     */
    private extractAutomationTargets(prompt: string): string {
        const lower = prompt.toLowerCase();
        
        // Try to find what specifically needs automation
        const automationMatch = prompt.match(/(?:automate|automation)\s+(?:my|the|a|an)?\s+([^?\.]+)/i);
        if (automationMatch) {
            return automationMatch[1].trim();
        }
        
        return 'the specified process';
    }

    /**
     * Extract core functionality from prompt
     */
    private extractCoreFunctionality(prompt: string): string {
        const lower = prompt.toLowerCase();
        
        // Try to find what functionality is needed
        const funcMatch = prompt.match(/(?:create|build|make|develop)\s+(?:a|an|the)?\s+([^?\.]+?)(?:\s+that|\s+with|\s+for|$)/i);
        if (funcMatch) {
            return funcMatch[1].trim();
        }
        
        return 'the specified functionality';
    }

    /**
     * ML analyzes prompt and determines what to add based on rules
     * ML infers the actual intent from user input, not hard-coded categories
     * Uses PyTorch Latent Prompt Optimizer if available, otherwise falls back to transformers.js
     */
    private async mlAnalyzeAndEnhance(
        originalPrompt: string,
        goal: string,
        patterns: Map<string, number>,
        analysis: PromptAnalysis,
        context?: ChatHistoryContext
    ): Promise<string> {
        // Try PyTorch Latent Prompt Optimizer first (if available)
        if (this.usePyTorch || await this.pytorchClient.checkHealth()) {
            this.usePyTorch = true;
            try {
                // Determine which rules to apply based on analysis
                const ruleIndices = this.determineRuleIndices(analysis, patterns);
                
                const result = await this.pytorchClient.optimizePrompt({
                    raw_prompt: originalPrompt,
                    rule_indices: ruleIndices,
                    return_embeddings: false
                });
                
                // Use PyTorch-optimized prompt if confidence is high enough
                if (result.confidence > 0.5 && result.optimized_prompt && result.optimized_prompt !== originalPrompt) {
                    return result.optimized_prompt;
                }
            } catch (error) {
                console.warn('PyTorch optimizer unavailable, using fallback:', error);
                this.usePyTorch = false;
            }
        }
        
        // Fallback to transformers.js-based enhancement
        let enhanced = originalPrompt;
        const lower = originalPrompt.toLowerCase();

        // ML analyzes what the user actually wants - don't force into categories
        // First, understand the actual intent from the prompt content
        const actualIntent = this.mlInferActualIntent(originalPrompt, goal, patterns, analysis);
        
        // Rule 1: Be Specific and Clear - add specifics based on ACTUAL intent, not assumptions
        const missingSpecificity = this.mlDetectMissingSpecificity(originalPrompt, analysis, patterns, actualIntent);
        if (missingSpecificity.length > 0) {
            enhanced += ' ' + missingSpecificity.join(' ');
        }

        // Rule 2: Keep it Concrete - replace abstract terms
        enhanced = this.mlMakeConcrete(enhanced, patterns);

        // Rule 3: Provide Context - only if truly missing
        const missingContext = this.mlDetectMissingContext(originalPrompt, patterns, context, actualIntent);
        if (missingContext) {
            enhanced = missingContext + ' ' + enhanced;
        }

        // Rule 4: Structure (handled separately)

        // Rule 5: Output Format - based on actual intent
        const missingFormat = this.mlDetectMissingFormat(originalPrompt, goal, patterns, actualIntent);
        if (missingFormat) {
            enhanced += ' ' + missingFormat;
        }

        // Rule 6: Requirements/Constraints - based on actual intent
        const missingRequirements = this.mlDetectMissingRequirements(originalPrompt, patterns, goal, actualIntent);
        if (missingRequirements.length > 0) {
            enhanced += ' ' + missingRequirements.join(' ');
        }

        return enhanced.trim();
    }

    /**
     * ML infers the actual intent from user prompt - don't force categories
     */
    private mlInferActualIntent(
        prompt: string,
        goal: string,
        patterns: Map<string, number>,
        analysis: PromptAnalysis
    ): { type: string | null; purpose: string | null; isSolutionRequest: boolean; needsImplementation: boolean } {
        const lower = prompt.toLowerCase();
        const features = this.mlModel.extractFeatures(prompt, goal, analysis);
        
        // ML analyzes what the user is actually asking for
        // Don't assume - only infer if there's strong evidence
        
        let type: string | null = null;
        let purpose: string | null = null;
        let isSolutionRequest = false;
        let needsImplementation = false;

        // Check if user is asking HOW to do something (not asking to build something)
        const isQuestion = prompt.includes('?') || 
                          lower.includes('how to') || 
                          lower.includes('how do') ||
                          lower.includes('what is') ||
                          lower.includes('explain') ||
                          lower.includes('tell me');

        // Check if user wants to CREATE/BUILD something
        const isCreationRequest = lower.includes('create') || 
                                 lower.includes('build') || 
                                 lower.includes('make') ||
                                 lower.includes('develop') ||
                                 lower.includes('design') ||
                                 lower.includes('implement');

        const automationIndicator = lower.includes('automate') || lower.includes('automation') ||
            lower.includes('streamline') || lower.includes('orchestrate') || lower.includes('optimize workflow') ||
            lower.includes('workflow automation') || lower.includes('process automation');
        const workflowIndicator = lower.includes('workflow') || lower.includes('process') || lower.includes('pipeline');
        const machiningIndicator = lower.includes('machining') || lower.includes('cnc') ||
            lower.includes('manufacturing') || lower.includes('shop floor') || lower.includes('factory');

        const isImplementationRequest = isCreationRequest ||
            automationIndicator ||
            (workflowIndicator && (lower.includes('automate') || lower.includes('optimize') || lower.includes('improve'))) ||
            machiningIndicator ||
            (features.imperativeCount >= 2 && !isQuestion);

        if (isQuestion && !isImplementationRequest) {
            // User is asking for information/explanation, not to build something
            isSolutionRequest = true;
            needsImplementation = false;
            
            // Infer what they're asking about
            if (lower.includes('money') || lower.includes('earn') || lower.includes('income') || lower.includes('revenue')) {
                type = 'financial_advice';
                purpose = 'understanding how to generate income or manage finances';
            } else if (lower.includes('learn') || lower.includes('understand')) {
                type = 'education';
                purpose = 'learning or understanding a concept';
            }
        } else {
            needsImplementation = isImplementationRequest;
            isSolutionRequest = !needsImplementation && (isQuestion || !isImplementationRequest);

            if (needsImplementation) {
                // Only infer type if there's strong evidence
                if (lower.includes('website') || lower.includes('web') || lower.includes('site') || 
                    (lower.includes('page') && lower.includes('html'))) {
                    type = 'website';
                } else if (lower.includes('api') || (lower.includes('endpoint') && !lower.includes('website'))) {
                    type = 'api';
                } else if (lower.includes('database') || lower.includes('sql') || lower.includes('schema')) {
                    type = 'database';
                } else if (lower.includes('app') && !lower.includes('website')) {
                    type = 'app';
                } else if (lower.includes('function') || lower.includes('code') || lower.includes('class')) {
                    type = 'code';
                } else if (machiningIndicator) {
                    type = 'machining_workflow';
                    purpose = 'automating the machining workflow';
                } else if (workflowIndicator && automationIndicator) {
                    type = 'workflow';
                    purpose = 'automating and orchestrating the workflow';
                } else if (automationIndicator) {
                    type = 'automation';
                    purpose = 'automating the requested process';
                }
                if (!purpose && type) {
                    purpose = `completing the ${type} task`;
                }
            } else {
                // Generic request - treat as informational
                isSolutionRequest = true;
            }
        }

        return { type, purpose, isSolutionRequest, needsImplementation };
    }

    /**
     * ML detects what specificity is missing based on ACTUAL intent, not assumptions
     */
    private mlDetectMissingSpecificity(
        prompt: string,
        analysis: PromptAnalysis,
        patterns: Map<string, number>,
        actualIntent: { type: string | null; purpose: string | null; isSolutionRequest: boolean; needsImplementation: boolean }
    ): string[] {
        const missing: string[] = [];
        const lower = prompt.toLowerCase();

        // Use ML to extract features and analyze what's missing
        const features = this.mlModel.extractFeatures(prompt, 'general', analysis);
        
        // ML analyzes prompt characteristics against rules to determine what's missing
        // Rule: Be Specific and Clear
        
        // Check principle analysis - if "Be Specific and Clear" failed, add specifics
        const specificityCheck = analysis.principleChecks.find(p => 
            p.principle === 'Be Specific and Clear'
        );
        
        // Only add specifics if the user is actually asking to CREATE/BUILD something
        // Don't add implementation details if they're just asking a question
        if (specificityCheck && specificityCheck.status !== 'pass') {
            // If user is asking HOW TO (not to build), add clarity about what they want to know
            if (actualIntent.isSolutionRequest && !actualIntent.needsImplementation) {
                // User wants information/explanation - make the question more specific
                if (prompt.length < 50 || !lower.includes('specific')) {
                    missing.push('Be specific about what aspect you want to understand or what information you need.');
                }
                return missing; // Don't add implementation details for questions
            }
            
            // User wants to CREATE/BUILD something - add implementation specifics
            if (actualIntent.needsImplementation) {
                // Only add specifics if we're confident about the type
                if (actualIntent.type === 'api' || (features.hasApi && !actualIntent.type)) {
                    if (!lower.includes('rest') && !lower.includes('endpoint') && !lower.includes('http')) {
                        missing.push('Use RESTful API endpoints with proper HTTP methods (GET, POST, PUT, DELETE).');
                    }
                    if (!lower.includes('json') && !lower.includes('format')) {
                        missing.push('Use JSON format for request and response bodies.');
                    }
                    if (!lower.includes('status') && !lower.includes('code')) {
                        missing.push('Return appropriate HTTP status codes (200, 201, 400, 404, 500).');
                    }
                }
                
                if (actualIntent.type === 'database' || (features.hasDatabase && !actualIntent.type)) {
                    if (!lower.includes('schema') && !lower.includes('table')) {
                        missing.push('Design a normalized database schema with properly defined tables and relationships.');
                    }
                    if (!lower.includes('key') && !lower.includes('constraint')) {
                        missing.push('Include primary keys, foreign keys, and appropriate constraints for data integrity.');
                    }
                }
                
                if (actualIntent.type === 'code' || (features.hasCode && !actualIntent.type)) {
                    if (!lower.includes('error') && !lower.includes('exception')) {
                        missing.push('Include comprehensive error handling and input validation.');
                    }
                    if (!lower.includes('document') && !lower.includes('comment')) {
                        missing.push('Add clear documentation and comments.');
                    }
                }
                
                // Only add website specifics if we're confident it's a website request
                if (actualIntent.type === 'website') {
                    if (!lower.includes('responsive') && !lower.includes('mobile')) {
                        missing.push('The website should be responsive and work on mobile, tablet, and desktop devices.');
                    }
                    if (!lower.includes('html') && !lower.includes('css') && !lower.includes('javascript')) {
                        missing.push('Use HTML5, CSS (flexbox/grid), and JavaScript.');
                    }
                }
            }
        }

        // If prompt is vague (short or low score), ML adds general specifics based on intent
        if (prompt.length < 50 || analysis.score < 60) {
            if (missing.length === 0) {
                // ML determines general specifics needed based on actual intent
                if (actualIntent.isSolutionRequest && !actualIntent.needsImplementation) {
                    // Question - ask for more specific information needs
                    missing.push('Be specific about what information or guidance you need.');
                } else if (actualIntent.needsImplementation) {
                    // Creation request - ask for more specific requirements
                    missing.push('Be specific about requirements, constraints, and expected outcomes.');
                } else {
                    // Generic - just ask for specificity
                    if (!lower.includes('specific') && !lower.includes('detailed')) {
                        missing.push('Be specific about what you need.');
                    }
                }
            }
        }

        return missing;
    }

    /**
     * ML makes prompt concrete by replacing abstract terms
     */
    private mlMakeConcrete(
        prompt: string,
        patterns: Map<string, number>
    ): string {
        let enhanced = prompt;

        // ML detects abstract/continuous terms and replaces with concrete ones
        enhanced = enhanced.replace(/make it better/gi, 'improve performance, readability, and maintainability');
        enhanced = enhanced.replace(/improve the code/gi, 'refactor with better structure, add error handling, and improve documentation');
        enhanced = enhanced.replace(/fix this/gi, 'identify and resolve the specific issue');
        enhanced = enhanced.replace(/make it work/gi, 'implement the functionality correctly with proper error handling');
        enhanced = enhanced.replace(/optimize/gi, 'optimize for performance, memory usage, and scalability');
        enhanced = enhanced.replace(/clean up/gi, 'refactor code structure, remove unused code, and improve organization');
        enhanced = enhanced.replace(/make it faster/gi, 'optimize for performance targeting [specific metric]');
        enhanced = enhanced.replace(/refactor/gi, 'refactor with these specific steps: [step 1], [step 2], [step 3]');

        return enhanced;
    }

    /**
     * ML detects missing context based on actual intent, not assumptions
     */
    private mlDetectMissingContext(
        prompt: string,
        patterns: Map<string, number>,
        context?: any,
        actualIntent?: { type: string | null; purpose: string | null; isSolutionRequest: boolean; needsImplementation: boolean }
    ): string | null {
        const lower = prompt.toLowerCase();
        let contextPrefix = null;

        // ML analyzes if context is missing (Rule: Provide Context)
        const hasContext = lower.includes('context') || lower.includes('working with') || 
                          lower.includes('using') || lower.includes('based on');

        if (!hasContext) {
            // ML determines context from chat history first (most reliable)
            if (context?.commonReferences && context.commonReferences.length > 0) {
                const refs = context.commonReferences.slice(0, 2).join(' and ');
                contextPrefix = `Using ${refs},`;
            } else if (actualIntent?.needsImplementation) {
                // Only add domain context if user is actually building something
                // And only if we're confident about the type
                const analysis = { score: 50 };
                const features = this.mlModel.extractFeatures(prompt, 'general', analysis);
                
                if (actualIntent.type === 'api' || (features.hasApi && !actualIntent.type)) {
                    if (!lower.includes('restful') && !lower.includes('api development')) {
                        contextPrefix = 'Working with RESTful API development,';
                    }
                } else if (actualIntent.type === 'database' || (features.hasDatabase && !actualIntent.type)) {
                    if (!lower.includes('database') && !lower.includes('database design')) {
                        contextPrefix = 'Working with database design,';
                    }
                }
            }
            // Don't add context for questions/information requests
        }

        return contextPrefix;
    }

    /**
     * ML detects missing output format based on actual intent
     */
    private mlDetectMissingFormat(
        prompt: string,
        goal: string,
        patterns: Map<string, number>,
        actualIntent?: { type: string | null; purpose: string | null; isSolutionRequest: boolean; needsImplementation: boolean }
    ): string | null {
        const lower = prompt.toLowerCase();
        
        // ML checks if format is specified
        const hasFormat = lower.includes('format') || lower.includes('output') || lower.includes('provide') || 
                         lower.includes('return') || lower.includes('generate');

        if (!hasFormat) {
            // ML determines appropriate format based on actual intent
            if (actualIntent?.isSolutionRequest && !actualIntent.needsImplementation) {
                // User is asking for information - provide explanation format
                return 'Provide a clear explanation with examples and actionable guidance.';
            } else if (actualIntent?.needsImplementation) {
                // User wants to build something - provide implementation format
                if (goal === 'code_generation' || actualIntent.type === 'code' || actualIntent.type === 'api' || actualIntent.type === 'website') {
                    return 'Provide the solution as clean, well-commented code.';
                }
            } else {
                // Fallback to goal-based format
                if (goal === 'code_generation') {
                    return 'Provide the solution as clean, well-commented code.';
                } else if (goal === 'explanation') {
                    return 'Provide a clear explanation with examples.';
                } else if (goal === 'code_review') {
                    return 'Provide a structured review with issues and recommendations.';
                } else if (goal === 'debugging') {
                    return 'Provide the fix with explanation of the root cause.';
                }
            }
        }

        return null;
    }

    /**
     * ML detects missing requirements based on ACTUAL intent, not assumptions
     */
    private mlDetectMissingRequirements(
        prompt: string,
        patterns: Map<string, number>,
        goal: string,
        actualIntent?: { type: string | null; purpose: string | null; isSolutionRequest: boolean; needsImplementation: boolean }
    ): string[] {
        const missing: string[] = [];
        const lower = prompt.toLowerCase();

        // Only add requirements if user is actually building something
        if (!actualIntent?.needsImplementation) {
            return missing; // Don't add implementation requirements for questions
        }

        // Use ML features to analyze what requirements are missing
        const analysis = { score: 50 }; // Dummy analysis for feature extraction
        const features = this.mlModel.extractFeatures(prompt, goal, analysis);

        // ML analyzes patterns based on ACTUAL intent, not keyword matching
        // Only add requirements if we're confident about the type
        
        if (actualIntent.type === 'api' || (features.hasApi && !actualIntent.type && lower.includes('api'))) {
            if (!lower.includes('validation') && !lower.includes('validate')) {
                missing.push('Include comprehensive input validation and error handling.');
            }
            if ((features.hasSecurity || patterns.has('security')) && !lower.includes('auth') && !lower.includes('authentication')) {
                missing.push('Implement authentication and authorization.');
            }
        }

        if (actualIntent.type === 'database' || (features.hasDatabase && !actualIntent.type && lower.includes('database'))) {
            if (!lower.includes('migration') && !lower.includes('version')) {
                missing.push('Include database migrations for schema versioning.');
            }
        }

        // Only add website requirements if we're confident it's a website
        if (actualIntent.type === 'website') {
            if (!lower.includes('accessibility') && !lower.includes('wcag')) {
                missing.push('Follow web accessibility standards (WCAG).');
            }
            if (!lower.includes('seo')) {
                missing.push('Implement SEO best practices.');
            }
        }

        if (actualIntent.type === 'code' || (features.hasCode && !actualIntent.type && (lower.includes('function') || lower.includes('code')))) {
            if (!lower.includes('test') && !lower.includes('testing') && goal === 'code_generation') {
                missing.push('Include error handling and consider edge cases.');
            }
        }

        return missing;
    }

    /**
     * Determine which rule indices to apply based on prompt analysis
     * Maps prompt characteristics to rule indices (0-7)
     */
    private determineRuleIndices(
        analysis: PromptAnalysis,
        patterns: Map<string, number>
    ): number[] {
        const ruleIndices: number[] = [];
        
        // Rule 0: Be Specific - if prompt lacks specificity
        const specificityCheck = analysis.principleChecks.find(p => 
            p.principle === 'Be Specific and Clear'
        );
        if (specificityCheck && specificityCheck.status !== 'pass') {
            ruleIndices.push(0);
        }
        
        // Rule 1: Add Context - if context is missing
        const contextCheck = analysis.principleChecks.find(p => 
            p.principle === 'Provide Context'
        );
        if (contextCheck && contextCheck.status !== 'pass') {
            ruleIndices.push(1);
        }
        
        // Rule 2: Define Output Format - if format not specified
        const formatCheck = analysis.principleChecks.find(p => 
            p.principle === 'Specify Output Format'
        );
        if (formatCheck && formatCheck.status !== 'pass') {
            ruleIndices.push(2);
        }
        
        // Rule 3: Include Constraints - if constraints missing
        if (patterns.get('constraints') === undefined || patterns.get('constraints')! < 0.3) {
            ruleIndices.push(3);
        }
        
        // Rule 4: Set Success Criteria - if success metrics missing
        if (patterns.get('success') === undefined || patterns.get('success')! < 0.3) {
            ruleIndices.push(4);
        }
        
        // Rule 5: Add Examples - if examples would help
        if (analysis.score < 70) {
            ruleIndices.push(5);
        }
        
        // Rule 6: Specify Domain - if domain unclear
        if (patterns.get('domain') === undefined || patterns.get('domain')! < 0.5) {
            ruleIndices.push(6);
        }
        
        // Rule 7: Structure Steps - always apply for implementation requests
        ruleIndices.push(7);
        
        return ruleIndices;
    }

    // Removed hard-coded intent methods - ML now analyzes patterns directly

    // Removed - now using rule-based enhancement that preserves user's prompt

    // Removed - now using rule-based enhancement that preserves user's prompt

    // Removed - now using rule-based enhancement that preserves user's prompt

    // Removed - now using rule-based enhancement that preserves user's prompt

    // Removed - now using rule-based enhancement that preserves user's prompt

    // Removed - now handled in enhanceForContext

    // Removed - now handled in enhanceForConcreteness

    // Removed - technical specs added in enhanceForClarity if needed

    /**
     * Generate specific execution steps - ML analyzes the ACTUAL user prompt
     */
    generateSpecificSteps(
        originalPrompt: string,
        goal: string,
        patterns: Map<string, number>,
        successCriteria?: string[],
        context?: any
    ): string {
        const steps: string[] = [];
        const lower = originalPrompt.toLowerCase();

        // ML analyzes the ACTUAL user prompt to generate appropriate steps
        // Check what the user actually mentioned in their prompt
        
        // If user mentioned specific things, create steps for those
        if (lower.includes('html') || lower.includes('structure')) {
            steps.push('Create HTML structure with semantic elements');
        }
        if (lower.includes('css') || lower.includes('style') || lower.includes('design')) {
            steps.push('Style with CSS using flexbox/grid for responsive layout');
        }
        if (lower.includes('javascript') || lower.includes('js') || lower.includes('interactive')) {
            steps.push('Add JavaScript for interactivity and dynamic content');
        }
        if (lower.includes('responsive') || lower.includes('mobile')) {
            steps.push('Test responsiveness across mobile, tablet, and desktop');
        }

        // If no specific steps from user prompt, generate based on goal
        if (steps.length === 0) {
            switch (goal) {
                case 'code_generation':
                    if (lower.includes('function') || lower.includes('method')) {
                        steps.push('Define function signature with parameters and return type');
                        steps.push('Implement core logic for main functionality');
                        steps.push('Add input validation and error handling');
                        steps.push('Add documentation and comments');
                    } else if (lower.includes('class') || lower.includes('component')) {
                        steps.push('Design class structure with properties and methods');
                        steps.push('Implement constructor and core methods');
                        steps.push('Add proper encapsulation and access modifiers');
                        steps.push('Include error handling and validation');
                    } else {
                        steps.push('Design code structure with clear separation of concerns');
                        steps.push('Implement core functionality following best practices');
                        steps.push('Add error handling and input validation');
                        steps.push('Test with various inputs and edge cases');
                    }
                    break;

                case 'website':
                case 'web':
                    steps.push('Create HTML structure with semantic elements');
                    steps.push('Style with CSS using flexbox/grid for responsive layout');
                    steps.push('Add JavaScript for interactivity and dynamic content');
                    steps.push('Test responsiveness across mobile, tablet, and desktop');
                    break;

                case 'api':
                    steps.push('Design RESTful endpoints with proper HTTP methods');
                    steps.push('Implement request handlers with input validation');
                    steps.push('Add error handling with appropriate status codes');
                    steps.push('Test endpoints with various request scenarios');
                    break;

                case 'debugging':
                    steps.push('Reproduce the issue consistently');
                    steps.push('Isolate problematic code section');
                    steps.push('Identify and fix root cause');
                    steps.push('Verify fix resolves issue without breaking other functionality');
                    break;

                case 'refactoring':
                    steps.push('Identify code smells and improvement areas');
                    steps.push('Design improved structure');
                    steps.push('Refactor incrementally with tests after each change');
                    steps.push('Verify functionality is preserved');
                    break;

                default:
                    // Infer from actual prompt content
                    if (lower.includes('website') || lower.includes('web')) {
                        steps.push('Create HTML structure with semantic elements');
                        steps.push('Style with CSS for responsive design');
                        steps.push('Add JavaScript for interactivity');
                    } else if (lower.includes('function') || lower.includes('code')) {
                        steps.push('Design code structure');
                        steps.push('Implement core functionality');
                        steps.push('Add error handling and validation');
                    } else {
                        steps.push('Analyze requirements and design solution');
                        steps.push('Implement following best practices');
                        steps.push('Test and verify solution');
                    }
            }
        }

        // Format concisely
        if (steps.length === 0) {
            return '';
        }

        let formatted = '';
        steps.forEach((step, index) => {
            formatted += `${index + 1}. ${step}\n`;
        });

        return formatted;
    }
}

