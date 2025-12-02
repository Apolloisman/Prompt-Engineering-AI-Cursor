export class PromptTemplates {
    private templates: Map<string, string> = new Map([
        ['code-generation-rctf', `[Role]: You are a [language] developer
[Context]: Working on [project/feature] where [background info]
[Task]: Create [what] that [does what]
[Requirements]:
- [requirement 1]
- [requirement 2]
- [requirement 3]
[Format]: [code/markdown/JSON/etc]
[Style]: [coding standards]`],

        ['code-review', `Review the following [language] code:
[code block]

Focus on:
- [aspect 1: performance/readability/security]
- [aspect 2]
- [aspect 3]

Provide:
- Issues found
- Suggestions for improvement
- Refactored code (if applicable)`],

        ['problem-solving-star', `[Situation]: [Describe current state/problem]
[Task]: [What needs to be accomplished]
[Action]: [Specific steps or approach]
[Result]: [Expected outcome or format]

Please:
1. Analyze the problem
2. Propose solution(s)
3. Explain trade-offs
4. Provide implementation`],

        ['explanation', `Explain [concept/topic] to someone with [knowledge level].

Include:
- Core concepts
- How it works
- Practical examples
- Common use cases
- Potential pitfalls`],

        ['debugging', `I'm experiencing an issue:
- Expected: [what should happen]
- Actual: [what actually happens]
- Error: [error message if any]
- Code: [relevant code]
- Environment: [relevant details]

Please help identify the problem and suggest fixes.`],

        ['chain-of-thought', `[Problem/Task]: [describe]

Think step by step:
1. First, analyze [aspect 1]
2. Then, identify [aspect 2]
3. Next, create [aspect 3]
4. Finally, verify [aspect 4]

Provide your reasoning at each step, then give the final solution.`]
    ]);

    applyFramework(text: string, framework: string): string {
        if (framework === 'auto') {
            framework = this.detectBestFramework(text);
        }

        switch (framework) {
            case 'RCTF':
                return this.applyRCTF(text);
            case 'STAR':
                return this.applySTAR(text);
            case 'Chain-of-Thought':
                return this.applyChainOfThought(text);
            default:
                return text;
        }
    }

    private detectBestFramework(text: string): string {
        const lower = text.toLowerCase();

        if (/review|analyze|check|examine/i.test(text)) {
            return 'STAR';
        }
        if (/explain|teach|learn|understand/i.test(text)) {
            return 'Chain-of-Thought';
        }
        if (/create|write|generate|build|implement/i.test(text)) {
            return 'RCTF';
        }
        if (/problem|issue|bug|error|debug/i.test(text)) {
            return 'STAR';
        }

        return 'RCTF'; // Default
    }

    private applyRCTF(text: string): string {
        if (text.includes('[Role]') || text.includes('Role:')) {
            return text; // Already structured
        }

        return `[Role]: You are a [specific role/expertise]
[Context]: ${text}
[Task]: [Specific action to perform]
[Format]: Output should be [format requirements]`;
    }

    private applySTAR(text: string): string {
        if (text.includes('[Situation]') || text.includes('Situation:')) {
            return text; // Already structured
        }

        return `[Situation]: [Describe current state]
[Task]: ${text}
[Action]: [Specific steps or approach]
[Result]: [Expected outcome or format]`;
    }

    private applyChainOfThought(text: string): string {
        if (text.includes('step') || text.includes('Step') || /\d+\./.test(text)) {
            return text; // Already has steps
        }

        return `${text}

Think step by step:
1. First, analyze...
2. Then, identify...
3. Next, create...
4. Finally, verify...`;
    }

    static getTemplate(key: string): string {
        const templates = new PromptTemplates();
        return templates.templates.get(key) || '';
    }

    getAllTemplates(): Map<string, string> {
        return this.templates;
    }
}



