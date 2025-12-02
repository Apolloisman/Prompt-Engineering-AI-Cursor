/**
 * ML-Powered Task Decomposition
 * Breaks down tasks into sub-tasks with success criteria and execution paths
 */

import { PromptAnalysis } from './promptAnalyzer';

export interface DecomposedCriterion {
    criterion: string;
    subCriteria: DecomposedCriterion[];
    executionSteps: ExecutionPath[];
    inputRelationships: InputRelationship[];
    rationale: string;
    confidence: number;
}

export interface ExecutionPath {
    step: string;
    subSteps: string[];
    howToAchieve: string; // How to get from input to this step
    verification: string; // How to verify this step is complete
    dependencies: string[]; // What needs to be done first
}

export interface InputRelationship {
    input: string; // What input is needed
    transformsTo: string; // What it becomes
    process: string; // How the transformation happens
    why: string; // Why this relationship exists
}

export interface TaskBreakdown {
    originalTask: string;
    mainCriteria: DecomposedCriterion[];
    relationships: PromptRelationship[];
    overallExecution: ExecutionPath[];
}

export interface PromptRelationship {
    pattern: string; // Pattern from good prompts
    appliesTo: string; // What task type it applies to
    breakdown: string; // How it breaks down tasks
    successCriteria: string[]; // Success criteria it generates
    executionSteps: string[]; // Execution steps it suggests
    confidence: number;
}

export class PromptDecomposer {
    private learnedRelationships: Map<string, PromptRelationship[]> = new Map();

    constructor() {
        this.initializeRelationships();
    }

    private initializeRelationships() {
        // Learn relationships from good prompts - how they break down tasks
        this.learnedRelationships.set('code_generation', [
            {
                pattern: 'step-by-step implementation with validation',
                appliesTo: 'code_generation',
                breakdown: 'Break into: analyze → design → implement → test → verify',
                successCriteria: [
                    'Requirements are clearly understood',
                    'Architecture is designed',
                    'Code is implemented',
                    'Tests pass',
                    'Verification confirms correctness'
                ],
                executionSteps: [
                    'Analyze requirements by identifying inputs, outputs, and constraints',
                    'Design architecture by planning data structures and component relationships',
                    'Implement code by writing functions/classes following the design',
                    'Test implementation by running test cases and checking outputs',
                    'Verify correctness by validating against requirements and edge cases'
                ],
                confidence: 0.95
            },
            {
                pattern: 'error handling and edge cases',
                appliesTo: 'code_generation',
                breakdown: 'For each step, add: identify edge cases → handle errors → validate inputs',
                successCriteria: [
                    'Edge cases are identified',
                    'Error handling is implemented',
                    'Input validation is in place'
                ],
                executionSteps: [
                    'Identify edge cases by analyzing boundary conditions and unusual inputs',
                    'Implement error handling by adding try-catch blocks and error messages',
                    'Validate inputs by checking data types, ranges, and constraints'
                ],
                confidence: 0.90
            }
        ]);

        this.learnedRelationships.set('debugging', [
            {
                pattern: 'systematic isolation approach',
                appliesTo: 'debugging',
                breakdown: 'Break into: reproduce → isolate → analyze → fix → verify',
                successCriteria: [
                    'Issue is consistently reproducible',
                    'Problematic code section is isolated',
                    'Root cause is identified',
                    'Fix is implemented',
                    'Issue is resolved and verified'
                ],
                executionSteps: [
                    'Reproduce issue by identifying conditions that trigger the problem',
                    'Isolate problematic code by narrowing down to specific functions/lines',
                    'Analyze root cause by examining code logic and data flow',
                    'Implement fix by addressing the root cause, not just symptoms',
                    'Verify fix by testing that issue is resolved and no new issues introduced'
                ],
                confidence: 0.95
            }
        ]);

        this.learnedRelationships.set('api', [
            {
                pattern: 'RESTful conventions and error codes',
                appliesTo: 'api',
                breakdown: 'For API tasks: design endpoints → implement handlers → add error handling → test responses',
                successCriteria: [
                    'Endpoints follow RESTful conventions',
                    'Request/response formats are defined',
                    'Error codes are properly implemented',
                    'API is tested and working'
                ],
                executionSteps: [
                    'Design endpoints by defining URL patterns, HTTP methods, and request/response schemas',
                    'Implement handlers by creating functions that process requests and return responses',
                    'Add error handling by implementing proper HTTP status codes and error messages',
                    'Test API by sending requests and verifying responses match specifications'
                ],
                confidence: 0.92
            }
        ]);
    }

    /**
     * Decompose success criteria into sub-criteria with execution paths
     */
    decomposeWithML(
        originalPrompt: string,
        successCriteria: string[],
        goal: string,
        analysis: PromptAnalysis,
        context?: any
    ): TaskBreakdown {
        const mainCriteria: DecomposedCriterion[] = [];
        
        // Get learned relationships for this goal
        const relationships = this.learnedRelationships.get(goal) || [];
        
        // For each success criterion, break it down using ML relationships
        successCriteria.forEach(criterion => {
            const decomposed = this.decomposeCriterion(
                criterion,
                originalPrompt,
                goal,
                relationships,
                analysis,
                context
            );
            mainCriteria.push(decomposed);
        });

        // Generate overall execution path
        const overallExecution = this.generateOverallExecution(
            mainCriteria,
            originalPrompt,
            goal,
            relationships
        );

        return {
            originalTask: originalPrompt,
            mainCriteria,
            relationships,
            overallExecution
        };
    }

    /**
     * Decompose a single criterion into sub-criteria with execution paths
     */
    private decomposeCriterion(
        criterion: string,
        originalPrompt: string,
        goal: string,
        relationships: PromptRelationship[],
        analysis: PromptAnalysis,
        context?: any
    ): DecomposedCriterion {
        // Find relevant relationships for this criterion
        const relevantRelationships = relationships.filter(rel => 
            this.criterionMatchesRelationship(criterion, rel)
        );

        // Break down the criterion using ML relationships
        const subCriteria: DecomposedCriterion[] = [];
        const executionSteps: ExecutionPath[] = [];
        const inputRelationships: InputRelationship[] = [];

        // Use relationships to break down
        if (relevantRelationships.length > 0) {
            const bestRelationship = relevantRelationships.sort((a, b) => 
                b.confidence - a.confidence
            )[0];

            // Generate sub-criteria from relationship
            bestRelationship.successCriteria.forEach(subCriterion => {
                if (this.isSubCriterionOf(criterion, subCriterion)) {
                    const decomposed = this.decomposeCriterion(
                        subCriterion,
                        originalPrompt,
                        goal,
                        relationships,
                        analysis,
                        context
                    );
                    subCriteria.push(decomposed);
                }
            });

            // Generate execution steps from relationship
            bestRelationship.executionSteps.forEach((step, index) => {
                const executionPath = this.createExecutionPath(
                    step,
                    originalPrompt,
                    criterion,
                    bestRelationship,
                    index,
                    context
                );
                executionSteps.push(executionPath);
            });

            // Generate input relationships
            inputRelationships.push(...this.generateInputRelationships(
                originalPrompt,
                criterion,
                bestRelationship,
                context
            ));
        } else {
            // No specific relationship found - use generic breakdown
            const genericBreakdown = this.genericBreakdown(criterion, originalPrompt, goal);
            subCriteria.push(...genericBreakdown.subCriteria);
            executionSteps.push(...genericBreakdown.executionSteps);
            inputRelationships.push(...genericBreakdown.inputRelationships);
        }

        // Generate rationale explaining the breakdown
        const rationale = this.generateRationale(
            criterion,
            subCriteria,
            executionSteps,
            inputRelationships,
            relevantRelationships
        );

        // Calculate confidence
        const confidence = this.calculateDecompositionConfidence(
            relevantRelationships,
            subCriteria,
            executionSteps
        );

        return {
            criterion,
            subCriteria,
            executionSteps,
            inputRelationships,
            rationale,
            confidence
        };
    }

    /**
     * Create execution path for a step
     */
    private createExecutionPath(
        step: string,
        originalPrompt: string,
        parentCriterion: string,
        relationship: PromptRelationship,
        index: number,
        context?: any
    ): ExecutionPath {
        // Parse step to extract action and details
        const action = this.extractAction(step);
        const subSteps = this.generateSubSteps(action, originalPrompt, parentCriterion, context);
        const howToAchieve = this.explainHowToAchieve(action, originalPrompt, parentCriterion, context);
        const verification = this.generateVerification(action, parentCriterion);
        const dependencies = this.identifyDependencies(action, index, relationship);

        return {
            step: action,
            subSteps,
            howToAchieve,
            verification,
            dependencies
        };
    }

    /**
     * Generate input relationships showing how inputs transform to outputs
     */
    private generateInputRelationships(
        originalPrompt: string,
        criterion: string,
        relationship: PromptRelationship,
        context?: any
    ): InputRelationship[] {
        const relationships: InputRelationship[] = [];

        // Extract inputs from original prompt
        const inputs = this.extractInputs(originalPrompt);
        
        // For each input, determine how it transforms
        inputs.forEach(input => {
            const transformsTo = this.determineTransformation(input, criterion, relationship);
            const process = this.determineProcess(input, transformsTo, criterion);
            const why = this.explainWhy(input, transformsTo, criterion);

            relationships.push({
                input,
                transformsTo,
                process,
                why
            });
        });

        return relationships;
    }

    /**
     * Generate overall execution path from all criteria
     */
    private generateOverallExecution(
        mainCriteria: DecomposedCriterion[],
        originalPrompt: string,
        goal: string,
        relationships: PromptRelationship[]
    ): ExecutionPath[] {
        const execution: ExecutionPath[] = [];

        // Flatten all execution steps from all criteria
        mainCriteria.forEach((criterion, index) => {
            // Add main execution steps
            execution.push(...criterion.executionSteps);

            // Add transitions between criteria
            if (index < mainCriteria.length - 1) {
                const transition = this.createTransition(
                    criterion,
                    mainCriteria[index + 1],
                    originalPrompt
                );
                execution.push(transition);
            }
        });

        // Order by dependencies
        return this.orderByDependencies(execution);
    }

    /**
     * Helper methods
     */
    private criterionMatchesRelationship(criterion: string, relationship: PromptRelationship): boolean {
        const lower = criterion.toLowerCase();
        return relationship.successCriteria.some(sc => 
            lower.includes(sc.toLowerCase()) || sc.toLowerCase().includes(lower)
        );
    }

    private isSubCriterionOf(parent: string, child: string): boolean {
        const parentLower = parent.toLowerCase();
        const childLower = child.toLowerCase();
        
        // Check if child is more specific than parent
        if (parentLower.includes('requirements') && childLower.includes('requirement')) return true;
        if (parentLower.includes('implement') && childLower.includes('code')) return true;
        if (parentLower.includes('verify') && childLower.includes('test')) return true;
        
        return false;
    }

    private extractAction(step: string): string {
        // Extract the main action from step description
        const actionMatch = step.match(/^([A-Z][^:]+)/);
        return actionMatch ? actionMatch[1].trim() : step;
    }

    private generateSubSteps(
        action: string,
        originalPrompt: string,
        parentCriterion: string,
        context?: any
    ): string[] {
        const subSteps: string[] = [];
        const lower = action.toLowerCase();

        if (lower.includes('analyze') || lower.includes('identify')) {
            subSteps.push('Extract key information from the prompt');
            subSteps.push('Identify specific requirements and constraints');
            subSteps.push('Document findings in a structured format');
        } else if (lower.includes('design') || lower.includes('plan')) {
            subSteps.push('Create a high-level structure or architecture');
            subSteps.push('Define components and their relationships');
            subSteps.push('Plan the implementation approach');
        } else if (lower.includes('implement') || lower.includes('create') || lower.includes('write')) {
            subSteps.push('Write the code following the design');
            subSteps.push('Follow coding best practices and standards');
            subSteps.push('Add comments and documentation');
        } else if (lower.includes('test') || lower.includes('verify')) {
            subSteps.push('Create test cases covering normal and edge cases');
            subSteps.push('Execute tests and check results');
            subSteps.push('Validate that all requirements are met');
        }

        return subSteps;
    }

    private explainHowToAchieve(
        action: string,
        originalPrompt: string,
        parentCriterion: string,
        context?: any
    ): string {
        const lower = action.toLowerCase();
        
        if (lower.includes('analyze')) {
            return `To analyze requirements: Read the prompt carefully, identify all mentioned requirements, extract constraints and edge cases, and organize this information into a clear structure. Use the original prompt: "${originalPrompt.substring(0, 100)}..." as the source.`;
        } else if (lower.includes('design')) {
            return `To design the solution: Based on the analyzed requirements, create a structure that addresses all needs. Break down into components, define interfaces, and plan how they interact. Consider scalability and maintainability.`;
        } else if (lower.includes('implement')) {
            return `To implement: Write code that follows the design. Start with core functionality, then add error handling and edge cases. Ensure code is clean, well-documented, and follows best practices.`;
        } else if (lower.includes('test')) {
            return `To test: Create test cases that cover normal scenarios, edge cases, and error conditions. Execute tests and verify outputs match expected results. Fix any issues found.`;
        }

        return `To achieve this: Follow the step-by-step process, ensuring each sub-step is completed before moving to the next.`;
    }

    private generateVerification(action: string, parentCriterion: string): string {
        const lower = action.toLowerCase();
        
        if (lower.includes('analyze')) {
            return `Verify by checking that all requirements are documented, constraints are identified, and the analysis is complete.`;
        } else if (lower.includes('design')) {
            return `Verify by ensuring the design addresses all requirements, components are well-defined, and the structure is clear.`;
        } else if (lower.includes('implement')) {
            return `Verify by checking that code compiles/runs, implements the design correctly, and handles edge cases.`;
        } else if (lower.includes('test')) {
            return `Verify by confirming all tests pass, edge cases are handled, and the solution meets all requirements.`;
        }

        return `Verify by checking that ${parentCriterion.toLowerCase()} is achieved.`;
    }

    private identifyDependencies(
        action: string,
        index: number,
        relationship: PromptRelationship
    ): string[] {
        const dependencies: string[] = [];
        
        if (index > 0) {
            // Previous steps in the relationship
            for (let i = 0; i < index; i++) {
                dependencies.push(relationship.executionSteps[i]);
            }
        }

        return dependencies;
    }

    private extractInputs(originalPrompt: string): string[] {
        const inputs: string[] = [];
        
        // Extract mentioned inputs
        if (originalPrompt.toLowerCase().includes('input')) {
            const inputMatch = originalPrompt.match(/input[:\s]+([^\.]+)/i);
            if (inputMatch) inputs.push(inputMatch[1].trim());
        }
        
        // Extract requirements
        if (originalPrompt.toLowerCase().includes('requirement')) {
            inputs.push('Requirements from prompt');
        }
        
        // Extract constraints
        if (originalPrompt.toLowerCase().includes('constraint')) {
            inputs.push('Constraints from prompt');
        }

        // Default: the prompt itself is an input
        if (inputs.length === 0) {
            inputs.push('Task description from prompt');
        }

        return inputs;
    }

    private determineTransformation(
        input: string,
        criterion: string,
        relationship: PromptRelationship
    ): string {
        // Determine what the input transforms into based on the criterion
        if (criterion.toLowerCase().includes('requirement')) {
            return 'Clear, documented requirements';
        } else if (criterion.toLowerCase().includes('design')) {
            return 'Solution architecture and structure';
        } else if (criterion.toLowerCase().includes('implement')) {
            return 'Working code implementation';
        } else if (criterion.toLowerCase().includes('test')) {
            return 'Test results and validation';
        }

        return 'Completed task output';
    }

    private determineProcess(
        input: string,
        transformsTo: string,
        criterion: string
    ): string {
        return `Process ${input} through analysis and implementation to produce ${transformsTo}. This involves breaking down the task, applying best practices, and ensuring quality.`;
    }

    private explainWhy(input: string, transformsTo: string, criterion: string): string {
        return `The ${input} needs to be transformed into ${transformsTo} to achieve "${criterion}". This relationship exists because the input contains the raw information that must be processed and structured to meet the success criterion.`;
    }

    private genericBreakdown(
        criterion: string,
        originalPrompt: string,
        goal: string
    ): { subCriteria: DecomposedCriterion[]; executionSteps: ExecutionPath[]; inputRelationships: InputRelationship[] } {
        // Generic breakdown when no specific relationship is found
        const subCriteria: DecomposedCriterion[] = [];
        const executionSteps: ExecutionPath[] = [];
        const inputRelationships: InputRelationship[] = [];

        // Break down into: understand → plan → execute → verify
        const genericSteps = [
            'Understand what needs to be done',
            'Plan how to achieve it',
            'Execute the plan',
            'Verify the result'
        ];

        genericSteps.forEach((step, index) => {
            executionSteps.push({
                step,
                subSteps: [`Sub-step ${index + 1}.1`, `Sub-step ${index + 1}.2`],
                howToAchieve: `To ${step.toLowerCase()}: Follow standard practices for this type of task.`,
                verification: `Verify by checking that ${step.toLowerCase()} is complete.`,
                dependencies: index > 0 ? [genericSteps[index - 1]] : []
            });
        });

        return { subCriteria, executionSteps, inputRelationships };
    }

    private generateRationale(
        criterion: string,
        subCriteria: DecomposedCriterion[],
        executionSteps: ExecutionPath[],
        inputRelationships: InputRelationship[],
        relationships: PromptRelationship[]
    ): string {
        let rationale = `Breaking down "${criterion}" into ${subCriteria.length} sub-criteria with ${executionSteps.length} execution steps. `;
        
        if (relationships.length > 0) {
            rationale += `Using learned pattern: "${relationships[0].pattern}" which applies ${relationships[0].breakdown}. `;
        }
        
        rationale += `The execution path shows how to transform inputs through ${inputRelationships.length} key relationships to achieve the criterion. `;
        rationale += `Each step has clear sub-actions and verification methods to ensure progress toward the goal.`;

        return rationale;
    }

    private calculateDecompositionConfidence(
        relationships: PromptRelationship[],
        subCriteria: DecomposedCriterion[],
        executionSteps: ExecutionPath[]
    ): number {
        let confidence = 0.5; // Base confidence

        // Boost confidence if we have relevant relationships
        if (relationships.length > 0) {
            confidence += relationships[0].confidence * 0.3;
        }

        // Boost confidence if we have detailed breakdown
        if (subCriteria.length > 0) {
            confidence += 0.1;
        }

        if (executionSteps.length > 0) {
            confidence += 0.1;
        }

        return Math.min(confidence, 1.0);
    }

    private createTransition(
        from: DecomposedCriterion,
        to: DecomposedCriterion,
        originalPrompt: string
    ): ExecutionPath {
        return {
            step: `Transition from "${from.criterion}" to "${to.criterion}"`,
            subSteps: [
                'Complete all verification for the previous criterion',
                'Prepare inputs needed for the next criterion',
                'Begin execution of the next criterion'
            ],
            howToAchieve: `After achieving "${from.criterion}", use its outputs as inputs for "${to.criterion}". Ensure all dependencies are satisfied.`,
            verification: `Verify that ${from.criterion.toLowerCase()} is complete and outputs are ready for ${to.criterion.toLowerCase()}.`,
            dependencies: [from.criterion]
        };
    }

    private orderByDependencies(execution: ExecutionPath[]): ExecutionPath[] {
        // Simple topological sort - order by dependencies
        const ordered: ExecutionPath[] = [];
        const added = new Set<string>();

        const addStep = (step: ExecutionPath) => {
            if (added.has(step.step)) return;

            // Add dependencies first
            step.dependencies.forEach(dep => {
                const depStep = execution.find(e => e.step === dep);
                if (depStep && !added.has(depStep.step)) {
                    addStep(depStep);
                }
            });

            ordered.push(step);
            added.add(step.step);
        };

        execution.forEach(step => addStep(step));
        return ordered;
    }

    /**
     * Learn new relationship from a good prompt
     */
    learnRelationship(relationship: PromptRelationship, goal: string) {
        if (!this.learnedRelationships.has(goal)) {
            this.learnedRelationships.set(goal, []);
        }
        this.learnedRelationships.get(goal)!.push(relationship);
    }
}

