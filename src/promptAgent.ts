import { PromptAnalysis } from './promptAnalyzer';
import { ChatHistoryContext } from './chatHistoryManager';
import { ReferenceAssessment, FormatRecommendation, MLRecommendations } from './mlAdvisor';
import { PromptMLModel, PromptFeatures, ExecutionStep, ExecutionRequirements, VerificationSteps } from './mlModel';
import { PromptMLGenerator, IdealPrompt } from './promptMLGenerator';

export interface StepGoal {
    stepNumber: number;
    goal: string;
    subGoals: string[];
    rationale: string;
    successCriteria: string[]; // AI-determined success criteria
}

export interface StructuralDecision {
    recommendedStructure: string;
    rationale: string;
    sections: string[];
    ordering: string[];
    formatStyle: string;
}

export interface AgenticPlan {
    overallGoal: string;
    goalRationale: string;
    stepGoals: StepGoal[];
    executionOrder: number[];
    dependencies: Map<number, number[]>;
    estimatedComplexity: 'simple' | 'medium' | 'complex';
    confidence: number;
}

export interface AgenticPrompt {
    optimizedPrompt: string;
    plan: AgenticPlan;
    structuralDecisions: StructuralDecision;
    aiDeterminedSuccessCriteria: Map<number, string[]>; // Step number -> success criteria
    mlGenerated?: IdealPrompt; // ML-generated ideal prompt following rules
    reasoning: string;
    improvements: string[];
    structuralChanges: string[];
    readyToCopy: boolean;
}

export class PromptAgent {
    private goalKeywords: Map<string, string[]> = new Map([
        ['code_generation', ['implement', 'create', 'build', 'develop', 'write', 'generate']],
        ['debugging', ['fix', 'debug', 'resolve', 'troubleshoot', 'identify', 'solve']],
        ['refactoring', ['improve', 'refactor', 'optimize', 'enhance', 'restructure', 'clean']],
        ['code_review', ['review', 'analyze', 'evaluate', 'check', 'examine', 'assess']],
        ['explanation', ['explain', 'describe', 'clarify', 'teach', 'document', 'illustrate']],
        ['test_generation', ['test', 'verify', 'validate', 'ensure', 'cover', 'check']]
    ]);
    private mlModel: PromptMLModel;
    private mlGenerator: PromptMLGenerator;

    constructor() {
        this.mlModel = new PromptMLModel();
        this.mlGenerator = new PromptMLGenerator();
    }

    async generateAgenticPrompt(
        originalPrompt: string,
        analysis: PromptAnalysis,
        chatHistory?: ChatHistoryContext,
        referenceAssessment?: ReferenceAssessment,
        mlRecommendations?: MLRecommendations
    ): Promise<AgenticPrompt> {
        // Step 1: Determine overall goal with reasoning
        const overallGoal = this.determineOverallGoal(originalPrompt, analysis, chatHistory);
        const goalRationale = this.explainGoalRationale(originalPrompt, overallGoal, analysis, chatHistory);

        // Step 2: Make structural decisions using AI-like analysis
        const structuralDecisions = this.makeStructuralDecisions(
            originalPrompt,
            analysis,
            mlRecommendations,
            overallGoal
        );

        // Step 3: Create agentic plan with step goals
        const plan = this.createAgenticPlan(originalPrompt, overallGoal, analysis, chatHistory, referenceAssessment);

        // Step 4: AI-determined success criteria for each step
        const aiDeterminedSuccessCriteria = this.determineSuccessCriteria(
            originalPrompt,
            plan,
            analysis,
            mlRecommendations,
            chatHistory
        );

        // Step 5: Generate ideal prompt using ML-learned patterns to follow rules
        // ML learns patterns and uses them to create prompt following all rules
        const idealPromptResult = await this.mlGenerator.generateIdealPrompt(
            originalPrompt,
            analysis,
            plan.overallGoal,
            chatHistory
        );
        
        // Use the ML-generated ideal prompt (it follows rules using learned patterns)
        const optimizedPrompt = idealPromptResult.prompt;

        // Step 6: Identify improvements and structural changes
        const improvements = this.identifyImprovements(originalPrompt, optimizedPrompt, plan);
        const structuralChanges = this.identifyStructuralChanges(originalPrompt, optimizedPrompt, structuralDecisions);

        // Step 7: Generate reasoning
        const reasoning = this.generateReasoning(plan, improvements, chatHistory, structuralDecisions, aiDeterminedSuccessCriteria);

        return {
            optimizedPrompt,
            plan,
            structuralDecisions,
            aiDeterminedSuccessCriteria,
            mlGenerated: idealPromptResult,
            reasoning: this.generateMLReasoning(idealPromptResult, plan, improvements, chatHistory, structuralDecisions),
            improvements,
            structuralChanges,
            readyToCopy: true
        };
    }

    private generateMLReasoning(
        idealPrompt: IdealPrompt,
        plan: AgenticPlan,
        improvements: string[],
        chatHistory?: ChatHistoryContext,
        structuralDecisions?: StructuralDecision
    ): string {
        let reasoning = `Generated ideal prompt using ML-learned patterns to follow prompt engineering rules.\n\n`;
        
        reasoning += `ML Pattern Learning:\n`;
        reasoning += `- Confidence: ${idealPrompt.confidence}%\n`;
        reasoning += `- Rules Followed: ${idealPrompt.rulesFollowed.length}/${idealPrompt.rulesFollowed.length + (structuralDecisions?.sections.length || 0)}\n`;
        reasoning += `- Patterns Applied: ${idealPrompt.patternsApplied.length}\n\n`;
        
        reasoning += `Patterns Used:\n`;
        idealPrompt.patternsApplied.forEach(pattern => {
            reasoning += `- ${pattern.pattern} (${(pattern.confidence * 100).toFixed(0)}% confidence)\n`;
            reasoning += `  Context: ${pattern.context}\n`;
            reasoning += `  Application: ${pattern.application}\n`;
        });
        
        reasoning += `\nRules Followed:\n`;
        idealPrompt.rulesFollowed.forEach(rule => {
            reasoning += `✓ ${rule}\n`;
        });
        
        if (chatHistory && chatHistory.patterns.length > 0) {
            reasoning += `\nInformed by chat history patterns: ${chatHistory.patterns.slice(0, 2).join(', ')}\n`;
        }
        
        return reasoning;
    }

    private determineOverallGoal(
        prompt: string,
        analysis: PromptAnalysis,
        chatHistory?: ChatHistoryContext
    ): string {
        const lower = prompt.toLowerCase();
        
        // Use chat history to inform goal if available
        if (chatHistory && chatHistory.recentMessages.length > 0) {
            const recentGoals = chatHistory.recentMessages
                .filter(m => m.goal)
                .slice(-3)
                .map(m => m.goal!);
            
            if (recentGoals.length > 0) {
                const mostCommon = this.getMostCommon(recentGoals);
                if (mostCommon && this.isGoalRelevant(lower, mostCommon)) {
                    return mostCommon;
                }
            }
        }

        // Pattern-based goal detection with confidence
        const goalScores = new Map<string, number>();
        
        for (const [goal, keywords] of this.goalKeywords.entries()) {
            const matches = keywords.filter(kw => lower.includes(kw)).length;
            if (matches > 0) {
                goalScores.set(goal, matches);
            }
        }

        // Check for specific patterns
        if (lower.includes('create') || lower.includes('write') || lower.includes('generate')) {
            if (lower.includes('code') || lower.includes('function') || lower.includes('class')) {
                goalScores.set('code_generation', (goalScores.get('code_generation') || 0) + 3);
            }
        }

        if (lower.includes('fix') || lower.includes('debug') || lower.includes('error')) {
            goalScores.set('debugging', (goalScores.get('debugging') || 0) + 3);
        }

        if (lower.includes('improve') || lower.includes('refactor') || lower.includes('optimize')) {
            goalScores.set('refactoring', (goalScores.get('refactoring') || 0) + 2);
        }

        // Return highest scoring goal
        let maxScore = 0;
        let bestGoal = 'general_assistance';
        
        goalScores.forEach((score, goal) => {
            if (score > maxScore) {
                maxScore = score;
                bestGoal = goal;
            }
        });

        return bestGoal;
    }

    private explainGoalRationale(
        prompt: string,
        goal: string,
        analysis: PromptAnalysis,
        chatHistory?: ChatHistoryContext
    ): string {
        const reasons: string[] = [];
        
        // Analyze prompt content
        const lower = prompt.toLowerCase();
        const keywords = this.goalKeywords.get(goal) || [];
        const foundKeywords = keywords.filter(kw => lower.includes(kw));
        
        if (foundKeywords.length > 0) {
            reasons.push(`Keywords detected: ${foundKeywords.join(', ')}`);
        }

        // Check analysis score
        if (analysis.score < 50) {
            reasons.push('Low prompt quality score suggests need for structured approach');
        }

        // Check chat history patterns
        if (chatHistory && chatHistory.patterns.length > 0) {
            const relevantPatterns = chatHistory.patterns.filter(p => 
                p.toLowerCase().includes(goal.replace('_', ' '))
            );
            if (relevantPatterns.length > 0) {
                reasons.push(`Chat history shows pattern: ${relevantPatterns[0]}`);
            }
        }

        // Check prompt structure
        if (prompt.length < 50) {
            reasons.push('Short prompt requires goal clarification');
        }

        return reasons.length > 0 
            ? reasons.join('. ') 
            : `Goal determined based on prompt content and context analysis`;
    }

    private createAgenticPlan(
        prompt: string,
        overallGoal: string,
        analysis: PromptAnalysis,
        chatHistory?: ChatHistoryContext,
        referenceAssessment?: ReferenceAssessment
    ): AgenticPlan {
        // Decompose goal into step goals
        const stepGoals = this.decomposeIntoStepGoals(prompt, overallGoal, analysis, referenceAssessment);
        
        // Determine execution order
        const executionOrder = this.determineExecutionOrder(stepGoals);
        
        // Map dependencies
        const dependencies = this.mapDependencies(stepGoals);
        
        // Calculate complexity
        const estimatedComplexity = this.estimateComplexity(prompt, stepGoals.length, analysis);
        
        // Calculate confidence
        const confidence = this.calculatePlanConfidence(stepGoals, analysis, chatHistory, referenceAssessment);

        return {
            overallGoal,
            goalRationale: this.explainGoalRationale(prompt, overallGoal, analysis, chatHistory),
            stepGoals,
            executionOrder,
            dependencies,
            estimatedComplexity,
            confidence
        };
    }

    private decomposeIntoStepGoals(
        prompt: string,
        overallGoal: string,
        analysis: PromptAnalysis,
        referenceAssessment?: ReferenceAssessment
    ): StepGoal[] {
        const stepGoals: StepGoal[] = [];
        const lower = prompt.toLowerCase();

        switch (overallGoal) {
            case 'code_generation':
                stepGoals.push(
                    {
                        stepNumber: 1,
                        goal: 'Understand and clarify requirements',
                        subGoals: [
                            'Identify what needs to be created',
                            'Determine inputs and outputs',
                            'Identify constraints and edge cases',
                            'Clarify any ambiguities'
                        ],
                        rationale: 'Clear requirements prevent scope creep and ensure correct implementation',
                        successCriteria: [
                            'All requirements are clearly defined',
                            'Input/output formats are specified',
                            'Edge cases are identified'
                        ]
                    },
                    {
                        stepNumber: 2,
                        goal: 'Design the solution architecture',
                        subGoals: [
                            'Plan data structures',
                            'Design function/class structure',
                            'Define interfaces and relationships',
                            'Consider scalability and maintainability'
                        ],
                        rationale: 'Good architecture makes implementation easier and code more maintainable',
                        successCriteria: [
                            'Architecture is clearly defined',
                            'Components and their relationships are specified',
                            'Design follows best practices'
                        ]
                    },
                    {
                        stepNumber: 3,
                        goal: 'Implement core functionality',
                        subGoals: [
                            'Write main logic',
                            'Implement data structures',
                            'Handle basic cases',
                            'Ensure code compiles/runs'
                        ],
                        rationale: 'Core functionality is the foundation - get it right before adding features',
                        successCriteria: [
                            'Core logic is implemented',
                            'Code compiles without errors',
                            'Basic functionality works'
                        ]
                    },
                    {
                        stepNumber: 4,
                        goal: 'Add robustness and error handling',
                        subGoals: [
                            'Add input validation',
                            'Implement error handling',
                            'Handle edge cases',
                            'Add logging if needed'
                        ],
                        rationale: 'Robust code handles errors gracefully and prevents crashes',
                        successCriteria: [
                            'All inputs are validated',
                            'Errors are handled appropriately',
                            'Edge cases are covered'
                        ]
                    },
                    {
                        stepNumber: 5,
                        goal: 'Verify and refine',
                        subGoals: [
                            'Test with sample inputs',
                            'Check code quality',
                            'Refactor if needed',
                            'Add documentation'
                        ],
                        rationale: 'Verification ensures correctness and quality',
                        successCriteria: [
                            'Tests pass',
                            'Code follows best practices',
                            'Documentation is complete'
                        ]
                    }
                );
                break;

            case 'debugging':
                stepGoals.push(
                    {
                        stepNumber: 1,
                        goal: 'Reproduce and isolate the issue',
                        subGoals: [
                            'Reproduce the error consistently',
                            'Identify the failing component',
                            'Isolate the problematic code section',
                            'Gather error details'
                        ],
                        rationale: 'Isolation is essential for effective debugging',
                        successCriteria: [
                            'Error is consistently reproducible',
                            'Problem area is identified',
                            'Error details are collected'
                        ]
                    },
                    {
                        stepNumber: 2,
                        goal: 'Analyze root cause',
                        subGoals: [
                            'Examine variable values',
                            'Check data flow',
                            'Verify logic correctness',
                            'Identify the root cause'
                        ],
                        rationale: 'Understanding root cause leads to proper fix',
                        successCriteria: [
                            'Root cause is identified',
                            'Why the error occurs is understood',
                            'Contributing factors are known'
                        ]
                    },
                    {
                        stepNumber: 3,
                        goal: 'Design and implement fix',
                        subGoals: [
                            'Plan the fix approach',
                            'Implement the solution',
                            'Ensure fix addresses root cause',
                            'Maintain code quality'
                        ],
                        rationale: 'Proper fix addresses root cause, not just symptoms',
                        successCriteria: [
                            'Fix is implemented',
                            'Root cause is addressed',
                            'Code quality is maintained'
                        ]
                    },
                    {
                        stepNumber: 4,
                        goal: 'Verify fix and prevent regressions',
                        subGoals: [
                            'Test that error is resolved',
                            'Verify no regressions',
                            'Test edge cases',
                            'Update tests if needed'
                        ],
                        rationale: 'Verification ensures fix works and doesn\'t break other functionality',
                        successCriteria: [
                            'Original error is fixed',
                            'No regressions introduced',
                            'Edge cases still work'
                        ]
                    }
                );
                break;

            default:
                // Generic step goals
                stepGoals.push(
                    {
                        stepNumber: 1,
                        goal: 'Understand requirements',
                        subGoals: ['Analyze what needs to be done', 'Identify key components'],
                        rationale: 'Clear understanding prevents mistakes',
                        successCriteria: ['Requirements are clear']
                    },
                    {
                        stepNumber: 2,
                        goal: 'Plan approach',
                        subGoals: ['Determine best method', 'Break into sub-tasks'],
                        rationale: 'Planning improves efficiency',
                        successCriteria: ['Plan is defined']
                    },
                    {
                        stepNumber: 3,
                        goal: 'Execute',
                        subGoals: ['Implement the plan', 'Follow best practices'],
                        rationale: 'Execution achieves the goal',
                        successCriteria: ['Task is completed']
                    },
                    {
                        stepNumber: 4,
                        goal: 'Verify results',
                        subGoals: ['Check correctness', 'Validate quality'],
                        rationale: 'Verification ensures quality',
                        successCriteria: ['Results are verified']
                    }
                );
        }

        // Add reference-related steps if needed
        if (referenceAssessment && referenceAssessment.criticalReferencesNeeded.length > 0) {
            stepGoals.unshift({
                stepNumber: 0,
                goal: 'Gather required references',
                subGoals: referenceAssessment.criticalReferencesNeeded.map(ref => 
                    `Obtain ${ref.type}: ${ref.description}`
                ),
                rationale: 'References are critical for accurate implementation',
                successCriteria: ['All required references are available']
            });
        }

        // Renumber steps
        stepGoals.forEach((step, index) => {
            step.stepNumber = index + 1;
        });

        return stepGoals;
    }

    private determineExecutionOrder(stepGoals: StepGoal[]): number[] {
        return stepGoals.map(step => step.stepNumber);
    }

    private mapDependencies(stepGoals: StepGoal[]): Map<number, number[]> {
        const dependencies = new Map<number, number[]>();
        
        for (let i = 1; i < stepGoals.length; i++) {
            dependencies.set(stepGoals[i].stepNumber, [stepGoals[i - 1].stepNumber]);
        }
        
        return dependencies;
    }

    private estimateComplexity(
        prompt: string,
        stepCount: number,
        analysis: PromptAnalysis
    ): 'simple' | 'medium' | 'complex' {
        const wordCount = prompt.split(/\s+/).length;
        const hasMultipleComponents = /(and|also|plus|additionally|multiple|several)/i.test(prompt);
        const hasComplexLogic = /(algorithm|optimize|performance|scalability|distributed|concurrent|async)/i.test(prompt);
        
        if (wordCount > 200 || (hasMultipleComponents && hasComplexLogic) || stepCount > 6) {
            return 'complex';
        } else if (wordCount > 100 || hasMultipleComponents || hasComplexLogic || stepCount > 4) {
            return 'medium';
        }
        return 'simple';
    }

    private calculatePlanConfidence(
        stepGoals: StepGoal[],
        analysis: PromptAnalysis,
        chatHistory?: ChatHistoryContext,
        referenceAssessment?: ReferenceAssessment
    ): number {
        let confidence = analysis.score;
        
        // Boost confidence if we have clear step goals
        if (stepGoals.length >= 3) {
            confidence += 10;
        }
        
        // Boost if chat history supports the plan
        if (chatHistory && chatHistory.patterns.length > 0) {
            confidence += 5;
        }
        
        // Reduce if references are needed but not available
        if (referenceAssessment && referenceAssessment.criticalReferencesNeeded.length > 0) {
            confidence -= 15;
        }
        
        return Math.max(0, Math.min(100, Math.round(confidence)));
    }

    private makeStructuralDecisions(
        prompt: string,
        analysis: PromptAnalysis,
        mlRecommendations?: MLRecommendations,
        overallGoal?: string
    ): StructuralDecision {
        const lower = prompt.toLowerCase();
        let recommendedStructure = 'RCTF'; // Role-Context-Task-Format (default)
        let formatStyle = 'structured';
        const sections: string[] = [];
        const ordering: string[] = [];
        let rationale = '';

        // Analyze prompt characteristics to determine best structure
        const hasMultipleParts = /(and|also|plus|additionally|multiple|several|first|then|next)/i.test(prompt);
        const isComplex = prompt.split(/\s+/).length > 100 || analysis.score < 50;
        const needsExamples = !lower.includes('example') && !lower.includes('sample');
        const needsConstraints = !lower.includes('constraint') && !lower.includes('limit') && !lower.includes('must');

        // Use ML recommendations if available
        if (mlRecommendations) {
            const formatType = mlRecommendations.suggestedFormat.format;
            if (formatType === 'structured' || formatType === 'list') {
                recommendedStructure = 'STAR'; // Situation-Task-Action-Result
                formatStyle = 'step_by_step';
            } else if (formatType === 'code' || formatType === 'json') {
                recommendedStructure = 'RCTF';
                formatStyle = 'code_oriented';
            }
        }

        // Determine sections needed
        sections.push('Role');
        sections.push('Goal');
        
        if (hasMultipleParts || isComplex) {
            sections.push('Context');
            sections.push('Execution Plan');
        }
        
        sections.push('Task');
        
        if (needsExamples) {
            sections.push('Examples');
        }
        
        if (needsConstraints) {
            sections.push('Constraints');
        }
        
        if (mlRecommendations && mlRecommendations.requiredReferences.length > 0) {
            sections.push('Required References');
        }
        
        sections.push('Output Format');
        sections.push('Success Criteria');

        // Determine ordering
        ordering.push(...sections);

        // Generate rationale
        rationale = `Based on analysis: `;
        if (isComplex) rationale += 'Complex task requires structured approach. ';
        if (hasMultipleParts) rationale += 'Multiple components need step-by-step breakdown. ';
        if (needsExamples) rationale += 'Examples will improve clarity. ';
        if (mlRecommendations) rationale += `ML advisor recommends ${mlRecommendations.suggestedFormat.format} format. `;
        rationale += `Using ${recommendedStructure} framework for optimal structure.`;

        return {
            recommendedStructure,
            rationale,
            sections,
            ordering,
            formatStyle
        };
    }

    private determineSuccessCriteria(
        prompt: string,
        plan: AgenticPlan,
        analysis: PromptAnalysis,
        mlRecommendations?: MLRecommendations,
        chatHistory?: ChatHistoryContext
    ): Map<number, string[]> {
        const criteria = new Map<number, string[]>();
        const lower = prompt.toLowerCase();

        plan.stepGoals.forEach(step => {
            const stepCriteria: string[] = [];

            // AI-like decision making for success criteria based on step goal
            switch (step.goal.toLowerCase()) {
                case 'understand requirements':
                case 'analyze requirements':
                    stepCriteria.push('All requirements are clearly identified and documented');
                    stepCriteria.push('Input/output specifications are defined');
                    stepCriteria.push('Edge cases and constraints are identified');
                    if (lower.includes('api') || lower.includes('integration')) {
                        stepCriteria.push('API endpoints and data formats are specified');
                    }
                    break;

                case 'design architecture':
                case 'design solution':
                    stepCriteria.push('Architecture diagram or structure is clearly defined');
                    stepCriteria.push('Component responsibilities are specified');
                    stepCriteria.push('Data flow and interactions are mapped');
                    stepCriteria.push('Design follows best practices for the domain');
                    break;

                case 'implement':
                case 'implement core logic':
                case 'implement solution':
                    stepCriteria.push('Core functionality is implemented and working');
                    stepCriteria.push('Code compiles/runs without errors');
                    stepCriteria.push('Basic test cases pass');
                    if (lower.includes('test') || mlRecommendations?.verificationChecklist) {
                        stepCriteria.push('Unit tests are written and passing');
                    }
                    break;

                case 'add error handling':
                case 'handle errors':
                    stepCriteria.push('All inputs are validated');
                    stepCriteria.push('Error cases are handled gracefully');
                    stepCriteria.push('Error messages are clear and actionable');
                    break;

                case 'verify':
                case 'test':
                case 'verify results':
                    stepCriteria.push('All test cases pass');
                    stepCriteria.push('Edge cases are covered');
                    stepCriteria.push('No regressions introduced');
                    if (mlRecommendations?.verificationChecklist) {
                        mlRecommendations.verificationChecklist.slice(0, 2).forEach(check => {
                            stepCriteria.push(check);
                        });
                    }
                    break;

                case 'gather references':
                case 'collect references':
                    if (mlRecommendations?.requiredReferences) {
                        mlRecommendations.requiredReferences.forEach(ref => {
                            stepCriteria.push(`${ref.type} reference obtained: ${ref.description}`);
                        });
                    } else {
                        stepCriteria.push('All required references are available');
                    }
                    break;

                default:
                    // Generic AI-determined criteria based on step characteristics
                    if (step.subGoals.length > 0) {
                        step.subGoals.forEach(subGoal => {
                            stepCriteria.push(`Completed: ${subGoal}`);
                        });
                    } else {
                        stepCriteria.push(`Step ${step.stepNumber} goal achieved: ${step.goal}`);
                        stepCriteria.push('Output meets quality standards');
                    }
            }

            // Add domain-specific criteria based on prompt content
            if (lower.includes('api') || lower.includes('rest') || lower.includes('endpoint')) {
                stepCriteria.push('API follows RESTful conventions');
                stepCriteria.push('Request/response formats are validated');
            }

            if (lower.includes('database') || lower.includes('sql') || lower.includes('query')) {
                stepCriteria.push('Database schema is properly designed');
                stepCriteria.push('Queries are optimized');
            }

            if (lower.includes('security') || lower.includes('auth') || lower.includes('encrypt')) {
                stepCriteria.push('Security best practices are followed');
                stepCriteria.push('Vulnerabilities are addressed');
            }

            // Use chat history to refine criteria
            if (chatHistory && chatHistory.patterns.length > 0) {
                const relevantPatterns = chatHistory.patterns.filter(p => 
                    p.toLowerCase().includes(step.goal.toLowerCase())
                );
                if (relevantPatterns.length > 0) {
                    stepCriteria.push(`Follows established pattern: ${relevantPatterns[0]}`);
                }
            }

            criteria.set(step.stepNumber, stepCriteria);
        });

        return criteria;
    }

    private async generateOptimizedPrompt(
        originalPrompt: string,
        plan: AgenticPlan,
        analysis: PromptAnalysis,
        chatHistory?: ChatHistoryContext,
        referenceAssessment?: ReferenceAssessment,
        structuralDecisions?: StructuralDecision,
        aiSuccessCriteria?: Map<number, string[]>,
        mlRecommendations?: MLRecommendations
    ): Promise<string> {
        let optimized = '';
        const lower = originalPrompt.toLowerCase();
        
        // Use structural decisions to organize the prompt (behind the scenes)
        const structure = structuralDecisions || this.makeStructuralDecisions(originalPrompt, analysis, mlRecommendations, plan.overallGoal);
        
        // Generate a clean, ready-to-use prompt - agentic system fills in details intelligently
        const role = this.getRoleForGoal(plan.overallGoal);
        optimized += `You are a ${role}.\n\n`;
        
        // Add context naturally if available (agentic fills this in)
        if (chatHistory && chatHistory.commonReferences.length > 0) {
            optimized += `Context: Working with ${chatHistory.commonReferences.slice(0, 3).join(', ')}.\n\n`;
        }
        
        // Main task - agentic system enhances it intelligently
        let enhancedTask = originalPrompt;
        
        // Agentic system makes intelligent inferences to fill in missing details
        if (originalPrompt.length < 50 || analysis.score < 50) {
            // Infer what the user likely wants based on the prompt
            if (lower.includes('website') || lower.includes('web')) {
                enhancedTask = `Create a modern, responsive website for ${this.inferPurpose(lower)}. Include proper HTML structure, CSS styling, and JavaScript functionality as needed.`;
            } else if (lower.includes('app') || lower.includes('application')) {
                enhancedTask = `Develop a ${this.inferAppType(lower)} application with a clean user interface and core functionality.`;
            } else if (lower.includes('api') || lower.includes('endpoint')) {
                enhancedTask = `Build a RESTful API with proper endpoints, request/response handling, and error management.`;
            } else if (lower.includes('money') || lower.includes('earn') || lower.includes('income')) {
                enhancedTask = `Design and implement a solution for generating revenue or managing finances, including monetization strategy and implementation approach.`;
            } else if (lower.includes('database') || lower.includes('data')) {
                enhancedTask = `Design and implement a database solution with proper schema, relationships, and data management.`;
            } else {
                // Generic enhancement
                enhancedTask = `${originalPrompt}. Provide a complete, production-ready solution with best practices.`;
            }
        }
        
        optimized += `Task: ${enhancedTask}\n\n`;
        
        // Extract features for ML analysis
        const features = this.mlModel.extractFeatures(originalPrompt, plan.overallGoal, analysis);
        
        // Convert rule-based steps to ML format
        const ruleBasedSteps: ExecutionStep[] = plan.stepGoals.map(step => ({
            stepNumber: step.stepNumber,
            action: step.goal,
            subActions: step.subGoals,
            rationale: step.rationale
        }));
        
        // Generate ML-enhanced execution steps (async)
        let mlSteps: ExecutionStep[];
        try {
            mlSteps = await this.mlModel.generateExecutionSteps(features, ruleBasedSteps);
        } catch (error) {
            console.error('ML step generation error, using rules:', error);
            mlSteps = ruleBasedSteps;
        }
        
        // Add step-by-step execution approach (ML-enhanced, actual execution)
        if (mlSteps.length > 1) {
            optimized += `Execute this task step-by-step:\n\n`;
            mlSteps.forEach((step, index) => {
                optimized += `${index + 1}. ${step.action}\n`;
                
                // Use ML-enhanced sub-actions
                if (step.subActions.length > 0) {
                    step.subActions.forEach(subAction => {
                        // Ensure execution language
                        let action = subAction;
                        if (!action.toLowerCase().startsWith('identify') && 
                            !action.toLowerCase().startsWith('determine') &&
                            !action.toLowerCase().startsWith('create') &&
                            !action.toLowerCase().startsWith('implement') &&
                            !action.toLowerCase().startsWith('write') &&
                            !action.toLowerCase().startsWith('build') &&
                            !action.toLowerCase().startsWith('design') &&
                            !action.toLowerCase().startsWith('analyze') &&
                            !action.toLowerCase().startsWith('test') &&
                            !action.toLowerCase().startsWith('verify') &&
                            !action.toLowerCase().startsWith('execute')) {
                            // Add action verb if missing
                            if (action.toLowerCase().includes('what')) {
                                action = `Identify ${action}`;
                            } else if (action.toLowerCase().includes('how')) {
                                action = `Determine ${action}`;
                            } else {
                                action = `Execute: ${action}`;
                            }
                        }
                        optimized += `   • ${action}\n`;
                    });
                }
                
                // Add AI-determined success criteria as execution checkpoints
                const successCriteria = aiSuccessCriteria?.get(step.stepNumber) || [];
                if (successCriteria.length > 0) {
                    optimized += `   Execution checkpoint: `;
                    // Convert first criterion to execution language
                    const checkpoint = successCriteria[0].replace(/are/i, 'verify').replace(/is/i, 'confirm');
                    optimized += `${checkpoint}\n`;
                }
                
                optimized += `\n`;
            });
        }
        
        // Generate ML-enhanced execution requirements
        const ruleBasedRequirements = [
            'Implement industry best practices and coding standards',
            'Write code that is maintainable and scalable',
            'Implement proper error handling and validation',
            ...(mlRecommendations?.missingElements?.slice(0, 2).map(el => {
                let req = el.replace(/should/i, 'must').replace(/consider/i, 'implement').replace(/add/i, 'implement');
                req = req.replace(/what should/i, 'implement').replace(/you should/i, 'execute');
                if (!req.toLowerCase().startsWith('implement') && 
                    !req.toLowerCase().startsWith('execute') &&
                    !req.toLowerCase().startsWith('include') &&
                    !req.toLowerCase().startsWith('create') &&
                    !req.toLowerCase().startsWith('add')) {
                    req = `Implement ${req}`;
                }
                return req.trim();
            }).filter(r => r) || [])
        ];
        
        let mlRequirements: ExecutionRequirements;
        try {
            mlRequirements = await this.mlModel.generateExecutionRequirements(features, ruleBasedRequirements);
        } catch (error) {
            console.error('ML requirement generation error, using rules:', error);
            mlRequirements = {
                requirements: ruleBasedRequirements,
                priorities: ruleBasedRequirements.map(() => 1)
            };
        }
        
        // Add execution requirements (ML-enhanced, prioritized)
        if (mlRequirements.requirements.length > 0) {
            optimized += `Execution Requirements:\n`;
            // Sort by priority (highest first)
            const sortedRequirements = mlRequirements.requirements
                .map((req, idx) => ({ req, priority: mlRequirements.priorities[idx] }))
                .sort((a, b) => b.priority - a.priority)
                .map(item => item.req);
            
            sortedRequirements.forEach(req => {
                optimized += `- ${req}\n`;
            });
            optimized += `\n`;
        }
        
        // Add examples if available (agentic determined this is helpful)
        if (structure.sections.includes('Examples') && mlRecommendations?.suggestedFormat.example) {
            optimized += `Example:\n${mlRecommendations.suggestedFormat.example}\n\n`;
        }
        
        // Add programming languages (agentic determined these are best)
        if (mlRecommendations?.suggestedLanguages && mlRecommendations.suggestedLanguages.length > 0) {
            optimized += `Programming Languages to Use:\n`;
            mlRecommendations.suggestedLanguages.forEach(lang => {
                optimized += `- ${lang.language} (${lang.suitability}% suitability)`;
                if (lang.pros && lang.pros.length > 0) {
                    optimized += ` - ${lang.pros[0]}`;
                }
                optimized += `\n`;
            });
            optimized += `\n`;
        }
        
        // Add context requirements - user needs to provide this
        if (mlRecommendations?.requiredContext && mlRecommendations.requiredContext.length > 0) {
            optimized += `Context Information Required (please provide):\n`;
            mlRecommendations.requiredContext.forEach(ctx => {
                optimized += `- ${ctx.type}: ${ctx.description}\n`;
            });
            optimized += `\n`;
        }
        
        // Add references - include ALL references (both critical and recommended)
        const allReferences: string[] = [];
        
        // Add critical references first
        if (referenceAssessment && referenceAssessment.criticalReferencesNeeded.length > 0) {
            referenceAssessment.criticalReferencesNeeded.forEach(ref => {
                allReferences.push(`${ref.type}: ${ref.description} (${ref.whereToFind})`);
            });
        }
        
        // Add other required references from ML recommendations
        if (mlRecommendations?.requiredReferences && mlRecommendations.requiredReferences.length > 0) {
            mlRecommendations.requiredReferences.forEach(ref => {
                // Avoid duplicates
                const refStr = `${ref.type}: ${ref.description}`;
                if (!allReferences.some(r => r.includes(ref.description))) {
                    allReferences.push(refStr);
                }
            });
        }
        
        if (allReferences.length > 0) {
            optimized += `Additional Sources/References Needed:\n`;
            optimized += `Please include the following references in your response:\n`;
            allReferences.forEach(ref => {
                optimized += `- ${ref}\n`;
            });
            optimized += `\n`;
        }
        
        // Output format - execution instruction (agentic determined best format)
        if (mlRecommendations?.suggestedFormat) {
            const formatType = mlRecommendations.suggestedFormat.format;
            optimized += `Output Execution:\n`;
            if (formatType === 'code') {
                optimized += `Generate clean, well-commented code with proper structure.\n`;
            } else if (formatType === 'json') {
                optimized += `Generate the response in JSON format with proper schema.\n`;
            } else if (formatType === 'structured') {
                optimized += `Generate a structured response following the step-by-step execution above.\n`;
            } else if (formatType === 'markdown') {
                optimized += `Format the response using Markdown with clear sections.\n`;
            } else {
                optimized += `Generate a clear, comprehensive response.\n`;
            }
            if (mlRecommendations.suggestedFormat.structure) {
                optimized += `Structure: ${mlRecommendations.suggestedFormat.structure}\n`;
            }
            optimized += `\n`;
        } else {
            optimized += `Output Execution: Generate a complete solution executing all steps above.\n\n`;
        }
        
        // Generate ML-enhanced verification steps
        const ruleBasedVerification = mlRecommendations?.verificationChecklist || [];
        let mlVerification: VerificationSteps;
        try {
            mlVerification = await this.mlModel.generateVerificationSteps(features, ruleBasedVerification);
        } catch (error) {
            console.error('ML verification generation error, using rules:', error);
            mlVerification = {
                steps: ruleBasedVerification,
                order: ruleBasedVerification.map((_, i) => i)
            };
        }
        
        // Add verification execution steps (ML-optimized order)
        if (mlVerification.steps.length > 0) {
            optimized += `Verification Execution:\n`;
            optimized += `Execute the following verification steps in order:\n`;
            
            // Use ML-optimized order
            const orderedSteps = mlVerification.order.map(idx => mlVerification.steps[idx]);
            
            orderedSteps.forEach((item, index) => {
                // Convert to execution language
                let verification = item;
                if (!verification.toLowerCase().startsWith('verify') &&
                    !verification.toLowerCase().startsWith('check') &&
                    !verification.toLowerCase().startsWith('test') &&
                    !verification.toLowerCase().startsWith('validate') &&
                    !verification.toLowerCase().startsWith('confirm') &&
                    !verification.toLowerCase().startsWith('ensure')) {
                    // Add execution verb
                    if (verification.toLowerCase().includes('correct') || verification.toLowerCase().includes('accurate')) {
                        verification = `Verify ${verification}`;
                    } else if (verification.toLowerCase().includes('works') || verification.toLowerCase().includes('function')) {
                        verification = `Test that ${verification}`;
                    } else {
                        verification = `Check ${verification}`;
                    }
                }
                optimized += `${index + 1}. ${verification}\n`;
            });
            optimized += `\n`;
        }
        
        return optimized.trim();
    }

    private inferPurpose(prompt: string): string {
        const lower = prompt.toLowerCase();
        if (lower.includes('business') || lower.includes('company') || lower.includes('money')) {
            return 'a business or commercial purpose';
        } else if (lower.includes('portfolio') || lower.includes('personal')) {
            return 'a personal portfolio or showcase';
        } else if (lower.includes('blog') || lower.includes('content')) {
            return 'content publishing or blogging';
        } else if (lower.includes('ecommerce') || lower.includes('shop') || lower.includes('store')) {
            return 'e-commerce or online retail';
        }
        return 'the specified purpose';
    }

    private inferAppType(prompt: string): string {
        const lower = prompt.toLowerCase();
        if (lower.includes('mobile') || lower.includes('ios') || lower.includes('android')) {
            return 'mobile';
        } else if (lower.includes('web')) {
            return 'web';
        } else if (lower.includes('desktop')) {
            return 'desktop';
        }
        return 'software';
    }

    private identifyImprovements(
        original: string,
        optimized: string,
        plan: AgenticPlan
    ): string[] {
        const improvements: string[] = [];
        
        if (original.length < 50) {
            improvements.push('Agentic system intelligently filled in missing details based on prompt analysis');
            improvements.push('Added detailed goal decomposition and step-by-step plan');
        }
        
        if (!original.toLowerCase().includes('goal')) {
            improvements.push('Explicitly defined overall goal and step goals');
        }
        
        if (!original.toLowerCase().includes('step') && !original.toLowerCase().includes('plan')) {
            improvements.push('Added structured execution plan with dependencies');
        }
        
        if (plan.stepGoals.length > 0) {
            improvements.push(`Decomposed task into ${plan.stepGoals.length} actionable steps with clear goals`);
        }
        
        if (plan.confidence > 70) {
            improvements.push('High confidence plan based on analysis and history');
        }
        
        improvements.push('AI-determined success criteria for each step based on task analysis');
        improvements.push('Structural optimization using recommended framework');
        improvements.push('Intelligent inference of missing details from context');
        
        return improvements;
    }

    private identifyStructuralChanges(
        original: string,
        optimized: string,
        structuralDecisions: StructuralDecision
    ): string[] {
        const changes: string[] = [];
        
        changes.push(`Applied ${structuralDecisions.recommendedStructure} framework structure`);
        changes.push(`Organized into ${structuralDecisions.sections.length} sections: ${structuralDecisions.sections.join(', ')}`);
        changes.push(`Format style: ${structuralDecisions.formatStyle}`);
        changes.push(`Rationale: ${structuralDecisions.rationale}`);
        
        return changes;
    }

    private generateReasoning(
        plan: AgenticPlan,
        improvements: string[],
        chatHistory?: ChatHistoryContext,
        structuralDecisions?: StructuralDecision,
        aiSuccessCriteria?: Map<number, string[]>
    ): string {
        let reasoning = `Generated agentic prompt with AI-powered goal decomposition, structural optimization, and dynamic success criteria.\n\n`;
        
        reasoning += `Overall Goal: ${plan.overallGoal.replace(/_/g, ' ')}\n`;
        reasoning += `Rationale: ${plan.goalRationale}\n\n`;
        
        reasoning += `Plan Details:\n`;
        reasoning += `- ${plan.stepGoals.length} steps with defined goals\n`;
        reasoning += `- Complexity: ${plan.estimatedComplexity}\n`;
        reasoning += `- Confidence: ${plan.confidence}%\n\n`;
        
        if (structuralDecisions) {
            reasoning += `Structural Decisions:\n`;
            reasoning += `- Framework: ${structuralDecisions.recommendedStructure}\n`;
            reasoning += `- Format Style: ${structuralDecisions.formatStyle}\n`;
            reasoning += `- Sections: ${structuralDecisions.sections.join(', ')}\n`;
            reasoning += `- Rationale: ${structuralDecisions.rationale}\n\n`;
        }
        
        if (aiSuccessCriteria && aiSuccessCriteria.size > 0) {
            reasoning += `AI-Determined Success Criteria:\n`;
            aiSuccessCriteria.forEach((criteria, stepNum) => {
                reasoning += `- Step ${stepNum}: ${criteria.length} criteria determined based on task analysis\n`;
            });
            reasoning += `\n`;
        }
        
        if (chatHistory && chatHistory.patterns.length > 0) {
            reasoning += `Informed by chat history patterns: ${chatHistory.patterns.slice(0, 2).join(', ')}\n\n`;
        }
        
        reasoning += `Key Improvements:\n`;
        improvements.forEach(improvement => {
            reasoning += `- ${improvement}\n`;
        });
        
        return reasoning;
    }

    private getRoleForGoal(goal: string): string {
        const roles: Record<string, string> = {
            'code_generation': 'senior software developer',
            'debugging': 'experienced software engineer',
            'code_review': 'senior code reviewer',
            'explanation': 'technical educator',
            'refactoring': 'senior software architect',
            'test_generation': 'QA engineer'
        };
        
        return roles[goal] || 'AI assistant';
    }

    private getMostCommon(items: string[]): string | null {
        const counts = new Map<string, number>();
        items.forEach(item => {
            counts.set(item, (counts.get(item) || 0) + 1);
        });
        
        let maxCount = 0;
        let mostCommon: string | null = null;
        counts.forEach((count, item) => {
            if (count > maxCount) {
                maxCount = count;
                mostCommon = item;
            }
        });
        
        return mostCommon;
    }

    private isGoalRelevant(prompt: string, goal: string): boolean {
        const keywords = this.goalKeywords.get(goal) || [];
        return keywords.some(kw => prompt.includes(kw));
    }
}



