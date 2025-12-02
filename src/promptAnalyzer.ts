import * as vscode from 'vscode';

export class PromptAnalyzer {
    private corePrinciples = [
        {
            name: 'Be Specific and Clear',
            check: (text: string) => this.checkSpecificity(text),
            weight: 20
        },
        {
            name: 'Keep It Concrete, Not Continuous',
            check: (text: string) => this.checkConcreteness(text),
            weight: 20
        },
        {
            name: 'Provide Context',
            check: (text: string) => this.checkContext(text),
            weight: 15
        },
        {
            name: 'Structure Your Prompt',
            check: (text: string) => this.checkStructure(text),
            weight: 15
        },
        {
            name: 'Include Examples',
            check: (text: string) => this.checkExamples(text),
            weight: 15
        },
        {
            name: 'Specify Output Format',
            check: (text: string) => this.checkOutputFormat(text),
            weight: 15
        }
    ];

    analyze(text: string): PromptAnalysis {
        const principleChecks: PrincipleCheck[] = [];
        let totalScore = 0;
        const suggestions: string[] = [];

        for (const principle of this.corePrinciples) {
            const result = principle.check(text);
            principleChecks.push({
                principle: principle.name,
                status: result.status,
                feedback: result.feedback
            });

            const score = result.status === 'pass' ? principle.weight : 
                         result.status === 'warning' ? principle.weight * 0.5 : 0;
            totalScore += score;

            if (result.suggestions.length > 0) {
                suggestions.push(...result.suggestions);
            }
        }

        return {
            score: Math.round(totalScore),
            principleChecks,
            suggestions: [...new Set(suggestions)] // Remove duplicates
        };
    }

    suggestImprovements(text: string): string {
        const analysis = this.analyze(text);
        let improved = text;

        // Apply improvements based on analysis
        if (analysis.score < 60) {
            improved = this.applyBasicImprovements(improved);
        }

        // Add structure if missing
        if (!this.checkStructure(text).status) {
            improved = this.addStructure(improved);
        }

        // Make concrete if abstract
        if (!this.checkConcreteness(text).status) {
            improved = this.makeConcrete(improved);
        }

        // Add context if missing
        if (!this.checkContext(text).status) {
            improved = this.addContext(improved);
        }

        return improved;
    }

    private checkSpecificity(text: string): CheckResult {
        const vagueWords = ['thing', 'stuff', 'better', 'improve', 'fix', 'make', 'do'];
        const vagueCount = vagueWords.filter(word => 
            new RegExp(`\\b${word}\\b`, 'i').test(text)
        ).length;

        const hasSpecifics = /\d+/.test(text) || 
                            /['"]([^'"]+)['"]/.test(text) ||
                            /(function|class|method|variable|file|line)/i.test(text);

        if (vagueCount > 2 && !hasSpecifics) {
            return {
                status: 'fail',
                feedback: 'Prompt contains vague language. Use specific terms, numbers, and concrete examples.',
                suggestions: ['Replace vague words with specific terms', 'Include exact function/class names', 'Add specific numbers or metrics']
            };
        } else if (vagueCount > 0) {
            return {
                status: 'warning',
                feedback: 'Some vague language detected. Consider being more specific.',
                suggestions: ['Replace vague words with specific terms']
            };
        }

        return {
            status: 'pass',
            feedback: 'Prompt uses specific and clear language.',
            suggestions: []
        };
    }

    private checkConcreteness(text: string): CheckResult {
        const abstractPhrases = [
            /improve\s+(the\s+)?(code|quality|performance)/i,
            /make\s+it\s+(better|faster|cleaner)/i,
            /refactor\s+(to\s+be\s+)?(better|cleaner|more\s+maintainable)/i
        ];

        const hasAbstract = abstractPhrases.some(pattern => pattern.test(text));
        const hasSteps = /(step|first|then|next|finally|\d+\.)/i.test(text);
        const hasSpecificActions = /(add|remove|extract|replace|create|implement|fix)\s+[a-z]+/i.test(text);

        if (hasAbstract && !hasSteps && !hasSpecificActions) {
            return {
                status: 'fail',
                feedback: 'Prompt is abstract/continuous. Break into concrete, discrete steps.',
                suggestions: [
                    'Break the task into numbered steps',
                    'Use action verbs: "add", "remove", "extract", "replace"',
                    'Specify exact changes: "Extract function X", "Add type hints to Y"'
                ]
            };
        } else if (hasAbstract && (hasSteps || hasSpecificActions)) {
            return {
                status: 'warning',
                feedback: 'Some abstract language, but has concrete elements.',
                suggestions: ['Consider making all steps more specific']
            };
        }

        return {
            status: 'pass',
            feedback: 'Prompt uses concrete, discrete steps.',
            suggestions: []
        };
    }

    private checkContext(text: string): CheckResult {
        const contextIndicators = [
            /(working\s+on|project|codebase|application|system)/i,
            /(language|framework|library|version)/i,
            /(context|background|currently|existing)/
        ];

        const hasContext = contextIndicators.some(pattern => pattern.test(text));
        const length = text.length;

        if (!hasContext && length < 100) {
            return {
                status: 'fail',
                feedback: 'Missing context. Add information about your project, language, or domain.',
                suggestions: [
                    'Add project context: "Working on a Python web scraping project..."',
                    'Mention language/framework: "Using React 18 with TypeScript..."',
                    'Include relevant background information'
                ]
            };
        } else if (!hasContext) {
            return {
                status: 'warning',
                feedback: 'Could benefit from more context.',
                suggestions: ['Add project or domain context']
            };
        }

        return {
            status: 'pass',
            feedback: 'Prompt includes relevant context.',
            suggestions: []
        };
    }

    private checkStructure(text: string): CheckResult {
        const hasSections = /(role|context|task|format|requirements|problem|solution|expected)/i.test(text);
        const hasFormatting = /(-|\*|\d+\.|:)/.test(text);
        const hasLineBreaks = text.split('\n').length > 3;

        if (!hasSections && !hasFormatting && !hasLineBreaks) {
            return {
                status: 'fail',
                feedback: 'Prompt lacks structure. Use sections, bullets, or numbered lists.',
                suggestions: [
                    'Use clear sections: Role, Context, Task, Format',
                    'Break into bullet points or numbered steps',
                    'Use formatting to separate different parts'
                ]
            };
        } else if (!hasSections && (hasFormatting || hasLineBreaks)) {
            return {
                status: 'warning',
                feedback: 'Has some structure, but could use clearer sections.',
                suggestions: ['Add section headers: Role, Context, Task, Format']
            };
        }

        return {
            status: 'pass',
            feedback: 'Prompt is well-structured.',
            suggestions: []
        };
    }

    private checkExamples(text: string): CheckResult {
        const hasExamples = /(example|sample|for\s+instance|such\s+as|like\s+this)/i.test(text);
        const hasCodeBlocks = /```/.test(text);
        const hasInputOutput = /(input|output|result|expected|actual)/i.test(text);

        if (!hasExamples && !hasCodeBlocks && text.length > 200) {
            return {
                status: 'warning',
                feedback: 'Consider adding examples to illustrate your needs.',
                suggestions: [
                    'Add example inputs and outputs',
                    'Include code examples if relevant',
                    'Show what good output looks like'
                ]
            };
        }

        return {
            status: hasExamples || hasCodeBlocks ? 'pass' : 'warning',
            feedback: hasExamples || hasCodeBlocks 
                ? 'Prompt includes examples.' 
                : 'No examples found. Examples help clarify requirements.',
            suggestions: hasExamples || hasCodeBlocks ? [] : ['Add concrete examples']
        };
    }

    private checkOutputFormat(text: string): CheckResult {
        const formatIndicators = [
            /(format|output|return|provide|generate)/i,
            /(json|markdown|code|table|list|string|array|object)/i,
            /(with|include|should\s+be|must\s+be)/
        ];

        const hasFormat = formatIndicators.some(pattern => pattern.test(text));

        if (!hasFormat && text.length > 150) {
            return {
                status: 'warning',
                feedback: 'Consider specifying the desired output format.',
                suggestions: [
                    'Specify format: "Return JSON", "Output as markdown", "Provide code with comments"',
                    'Mention structure: "Return a table", "Include type hints"'
                ]
            };
        }

        return {
            status: hasFormat ? 'pass' : 'warning',
            feedback: hasFormat 
                ? 'Output format is specified.' 
                : 'Output format not explicitly specified.',
            suggestions: hasFormat ? [] : ['Specify the desired output format']
        };
    }

    private applyBasicImprovements(text: string): string {
        let improved = text;

        // Capitalize first letter
        if (improved[0] && improved[0] === improved[0].toLowerCase()) {
            improved = improved[0].toUpperCase() + improved.slice(1);
        }

        // Ensure it ends with punctuation
        if (!/[.!?]$/.test(improved.trim())) {
            improved = improved.trim() + '.';
        }

        return improved;
    }

    private addStructure(text: string): string {
        if (text.includes('Role:') || text.includes('Task:') || text.includes('Context:')) {
            return text; // Already has structure
        }

        return `[Role]: You are a [specific role/expertise]
[Context]: ${text}
[Task]: [What needs to be accomplished]
[Format]: [Desired output format]`;
    }

    private makeConcrete(text: string): string {
        // Replace abstract phrases with concrete suggestions
        let concrete = text
            .replace(/improve\s+(the\s+)?code/gi, 'Refactor the code with these steps:\n1. [specific action 1]\n2. [specific action 2]\n3. [specific action 3]')
            .replace(/make\s+it\s+better/gi, 'Apply these improvements:\n- [specific improvement 1]\n- [specific improvement 2]')
            .replace(/fix\s+this/gi, 'Fix the following issues:\n1. [issue 1]\n2. [issue 2]');

        return concrete;
    }

    private addContext(text: string): string {
        if (/working\s+on|project|context/i.test(text)) {
            return text; // Already has context
        }

        return `[Context]: Working on [your project/domain] where [relevant background]

${text}`;
    }
}

interface CheckResult {
    status: 'pass' | 'warning' | 'fail';
    feedback: string;
    suggestions: string[];
}

export interface PromptAnalysis {
    score: number;
    principleChecks: PrincipleCheck[];
    suggestions: string[];
}

export interface PrincipleCheck {
    principle: string;
    status: 'pass' | 'warning' | 'fail';
    feedback: string;
}

interface CheckResult {
    status: 'pass' | 'warning' | 'fail';
    feedback: string;
    suggestions: string[];
}

