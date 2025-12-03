import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export class GuideViewer {
    private context: vscode.ExtensionContext;
    private panel: vscode.WebviewPanel | undefined;

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
    }

    show() {
        if (this.panel) {
            this.panel.reveal(vscode.ViewColumn.Beside);
            return;
        }

        this.panel = vscode.window.createWebviewPanel(
            'promptGuide',
            'Prompt Engineering Guide',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        // Try to load the guide file
        const guidePath = path.join(this.context.extensionPath, '..', 'prompt_engineering_guide.txt');
        let guideContent = '';

        if (fs.existsSync(guidePath)) {
            guideContent = fs.readFileSync(guidePath, 'utf-8');
        } else {
            // Fallback: use embedded guide
            guideContent = this.getEmbeddedGuide();
        }

        this.panel.webview.html = this.generateGuideHTML(guideContent);

        this.panel.onDidDispose(() => {
            this.panel = undefined;
        });
    }

    private generateGuideHTML(content: string): string {
        // Convert markdown-like text to HTML
        const html = this.convertToHTML(content);

        return `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: var(--vscode-font-family);
            padding: 20px;
            color: var(--vscode-foreground);
            background: var(--vscode-editor-background);
            line-height: 1.6;
        }
        h1 {
            color: var(--vscode-textLink-foreground);
            border-bottom: 2px solid var(--vscode-textLink-foreground);
            padding-bottom: 10px;
        }
        h2 {
            color: var(--vscode-textLink-foreground);
            margin-top: 30px;
            border-left: 4px solid var(--vscode-textLink-foreground);
            padding-left: 10px;
        }
        h3 {
            color: var(--vscode-foreground);
            margin-top: 20px;
        }
        code {
            background: var(--vscode-textCodeBlock-background);
            padding: 2px 6px;
            border-radius: 3px;
            font-family: var(--vscode-editor-font-family);
        }
        pre {
            background: var(--vscode-textCodeBlock-background);
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid var(--vscode-textLink-foreground);
        }
        ul, ol {
            margin-left: 20px;
        }
        li {
            margin: 5px 0;
        }
        .principle {
            background: var(--vscode-editor-background);
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid var(--vscode-textLink-foreground);
        }
        .example {
            background: var(--vscode-textCodeBlock-background);
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            font-style: italic;
        }
        .bad {
            color: #F44336;
        }
        .good {
            color: #4CAF50;
        }
        .checklist {
            background: var(--vscode-editor-background);
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .checklist-item {
            margin: 8px 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }
        th, td {
            border: 1px solid var(--vscode-panel-border);
            padding: 8px;
            text-align: left;
        }
        th {
            background: var(--vscode-textCodeBlock-background);
            font-weight: bold;
        }
        .nav {
            position: sticky;
            top: 0;
            background: var(--vscode-editor-background);
            padding: 10px;
            border-bottom: 2px solid var(--vscode-panel-border);
            margin-bottom: 20px;
        }
        .nav a {
            color: var(--vscode-textLink-foreground);
            text-decoration: none;
            margin-right: 15px;
        }
        .nav a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="nav">
        <a href="#principles">Core Principles</a>
        <a href="#frameworks">Frameworks</a>
        <a href="#patterns">Patterns</a>
        <a href="#templates">Templates</a>
        <a href="#checklist">Checklist</a>
    </div>
    ${html}
</body>
</html>`;
    }

    private convertToHTML(text: string): string {
        let html = text;

        // Convert headers
        html = html.replace(/^([A-Z][A-Z\s]+)$/gm, '<h1>$1</h1>');
        html = html.replace(/^([A-Z][a-z\s]+)$/gm, (match) => {
            if (match.length < 50 && !match.includes(':')) {
                return `<h2>${match}</h2>`;
            }
            return match;
        });

        // Convert numbered lists
        html = html.replace(/^(\d+)\.\s+(.+)$/gm, '<li>$2</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ol>$&</ol>');

        // Convert bullet points
        html = html.replace(/^[-*]\s+(.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => {
            if (!match.includes('<ol>')) {
                return `<ul>${match}</ul>`;
            }
            return match;
        });

        // Convert code blocks
        html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Convert examples
        html = html.replace(/Example \(([^)]+)\):/g, '<div class="example"><strong>Example ($1):</strong>');
        html = html.replace(/Example:/g, '<div class="example"><strong>Example:</strong>');

        // Convert weak/strong prompts
        html = html.replace(/WEAK PROMPT:/g, '<div class="bad"><strong>WEAK PROMPT:</strong>');
        html = html.replace(/STRONG PROMPT:/g, '</div><div class="good"><strong>STRONG PROMPT:</strong>');

        // Convert principles
        html = html.replace(/^(\d+)\.\s+([A-Z][^:]+):/gm, '<div class="principle"><h3>$1. $2</h3>');

        // Convert checkboxes
        html = html.replace(/□/g, '<input type="checkbox" class="checklist-item">');

        // Preserve line breaks
        html = html.replace(/\n/g, '<br>');

        return html;
    }

    private getEmbeddedGuide(): string {
        // Fallback guide content if file not found
        return `PROMPT ENGINEERING GUIDE
========================

This guide contains best practices, frameworks, and tips for creating effective prompts.

CORE PRINCIPLES
---------------

1. BE SPECIFIC AND CLEAR
2. KEEP IT CONCRETE, NOT CONTINUOUS
3. PROVIDE CONTEXT
4. STRUCTURE YOUR PROMPT
5. INCLUDE EXAMPLES
6. ITERATE AND REFINE

See the full guide file (prompt_engineering_guide.txt) for complete details.`;
    }
}





