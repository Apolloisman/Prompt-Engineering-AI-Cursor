import * as vscode from 'vscode';
import { PromptAnalyzer, PromptAnalysis } from './promptAnalyzer';
import { PromptTemplates } from './promptTemplates';
import { GuideViewer } from './guideViewer';
import { MLAdvisor, MLRecommendations } from './mlAdvisor';

export function activate(context: vscode.ExtensionContext) {
    const analyzer = new PromptAnalyzer();
    const templates = new PromptTemplates();
    const guideViewer = new GuideViewer(context);
    const mlAdvisor = new MLAdvisor();

    // Command: Suggest improved prompt
    const suggestCommand = vscode.commands.registerCommand(
        'promptAssistant.suggestPrompt',
        async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor');
                return;
            }

            const selection = editor.document.getText(editor.selection);
            if (!selection.trim()) {
                vscode.window.showWarningMessage('Please select text to improve');
                return;
            }

            const improved = analyzer.suggestImprovements(selection);
            const framework = vscode.workspace.getConfiguration('promptAssistant').get<string>('preferredFramework', 'auto');
            const structured = templates.applyFramework(improved, framework);

            const result = await vscode.window.showInformationMessage(
                'Improved prompt generated!',
                'Insert',
                'Show Analysis',
                'Cancel'
            );

            if (result === 'Insert') {
                editor.edit(editBuilder => {
                    editBuilder.replace(editor.selection, structured);
                });
            } else if (result === 'Show Analysis') {
                const analysis = analyzer.analyze(selection);
                showAnalysisPanel(analysis, structured);
            }
        }
    );

    // Command: Analyze current prompt
    const analyzeCommand = vscode.commands.registerCommand(
        'promptAssistant.analyzePrompt',
        async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor');
                return;
            }

            const selection = editor.document.getText(editor.selection);
            if (!selection.trim()) {
                vscode.window.showWarningMessage('Please select text to analyze');
                return;
            }

            const analysis = analyzer.analyze(selection);
            const improved = analyzer.suggestImprovements(selection);
            const framework = vscode.workspace.getConfiguration('promptAssistant').get<string>('preferredFramework', 'auto');
            const structured = templates.applyFramework(improved, framework);

            showAnalysisPanel(analysis, structured);
        }
    );

    // Command: Show guide
    const guideCommand = vscode.commands.registerCommand(
        'promptAssistant.showGuide',
        () => {
            guideViewer.show();
        }
    );

    // Command: Get ML-powered recommendations
    const mlRecommendCommand = vscode.commands.registerCommand(
        'promptAssistant.getMLRecommendations',
        async () => {
            const editor = vscode.window.activeTextEditor;
            let prompt = '';
            
            if (editor && !editor.selection.isEmpty) {
                prompt = editor.document.getText(editor.selection);
            } else {
                // Try clipboard
                prompt = await vscode.env.clipboard.readText();
            }
            
            if (!prompt.trim()) {
                const input = await vscode.window.showInputBox({
                    placeHolder: 'Enter your prompt to get ML-powered recommendations...',
                    prompt: 'ML Advisor will analyze and provide comprehensive recommendations'
                });
                if (!input) return;
                prompt = input;
            }
            
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Analyzing prompt with ML Advisor...",
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 0 });
                
                const recommendations = mlAdvisor.analyzeAndRecommend(prompt);
                
                progress.report({ increment: 100 });
                
                showMLRecommendationsPanel(recommendations, prompt);
            });
        }
    );

    // Command: Insert template
    const templateCommand = vscode.commands.registerCommand(
        'promptAssistant.insertTemplate',
        async () => {
            const templates = [
                { label: 'Code Generation (RCTF)', template: 'code-generation-rctf' },
                { label: 'Code Review', template: 'code-review' },
                { label: 'Problem Solving (STAR)', template: 'problem-solving-star' },
                { label: 'Explanation/Teaching', template: 'explanation' },
                { label: 'Debugging', template: 'debugging' },
                { label: 'Chain-of-Thought', template: 'chain-of-thought' }
            ];

            const selected = await vscode.window.showQuickPick(templates, {
                placeHolder: 'Select a prompt template'
            });

            if (selected) {
                const editor = vscode.window.activeTextEditor;
                if (editor) {
                    const template = PromptTemplates.getTemplate(selected.template);
                    const position = editor.selection.active;
                    editor.edit(editBuilder => {
                        editBuilder.insert(position, template);
                    });
                }
            }
        }
    );

    // Command: Improve prompt from clipboard (for chat interface)
    const clipboardCommand = vscode.commands.registerCommand(
        'promptAssistant.improveClipboard',
        async () => {
            const clipboardText = await vscode.env.clipboard.readText();
            if (!clipboardText.trim()) {
                vscode.window.showWarningMessage('Clipboard is empty');
                return;
            }

            // Offer ML recommendations by default
            const choice = await vscode.window.showQuickPick([
                { label: '🤖 Get ML-Powered Recommendations', description: 'Full analysis with model, parameters, context, and more', value: 'ml' },
                { label: '✨ Quick Improve', description: 'Fast prompt improvement', value: 'quick' }
            ], {
                placeHolder: 'Choose improvement type'
            });

            if (!choice) return;

            if (choice.value === 'ml') {
                // Use ML advisor
                vscode.window.withProgress({
                    location: vscode.ProgressLocation.Notification,
                    title: "Analyzing prompt with ML Advisor...",
                    cancellable: false
                }, async (progress) => {
                    progress.report({ increment: 0 });
                    const recommendations = mlAdvisor.analyzeAndRecommend(clipboardText);
                    progress.report({ increment: 100 });
                    showMLRecommendationsPanel(recommendations, clipboardText);
                });
            } else {
                // Quick improve
                const improved = analyzer.suggestImprovements(clipboardText);
                const framework = vscode.workspace.getConfiguration('promptAssistant').get<string>('preferredFramework', 'auto');
                const structured = templates.applyFramework(improved, framework);

                const result = await vscode.window.showInformationMessage(
                    'Improved prompt ready!',
                    'Copy to Clipboard',
                    'Show Analysis',
                    'Open in Editor',
                    'Cancel'
                );

                if (result === 'Copy to Clipboard') {
                    await vscode.env.clipboard.writeText(structured);
                    vscode.window.showInformationMessage('Improved prompt copied to clipboard! Paste it into Cursor chat.');
                } else if (result === 'Show Analysis') {
                    const analysis = analyzer.analyze(clipboardText);
                    showAnalysisPanel(analysis, structured);
                } else if (result === 'Open in Editor') {
                    const doc = await vscode.workspace.openTextDocument({
                        content: structured,
                        language: 'plaintext'
                    });
                    await vscode.window.showTextDocument(doc);
                }
            }
        }
    );

    // Status bar item for quick access
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'promptAssistant.improveClipboard';
    statusBarItem.text = '$(sparkle) Improve Prompt';
    statusBarItem.tooltip = 'Improve prompt from clipboard (for Cursor chat)';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    context.subscriptions.push(suggestCommand, analyzeCommand, guideCommand, templateCommand, clipboardCommand, mlRecommendCommand);

    // Auto-suggest on typing (if enabled)
    const config = vscode.workspace.getConfiguration('promptAssistant');
    if (config.get<boolean>('autoSuggest', false)) {
        setupAutoSuggest(context, analyzer, templates);
    }
}

function showAnalysisPanel(analysis: PromptAnalysis, improvedPrompt: string) {
    const panel = vscode.window.createWebviewPanel(
        'promptAnalysis',
        'Prompt Analysis',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    const html = generateAnalysisHTML(analysis, improvedPrompt);
    panel.webview.html = html;
}

function generateAnalysisHTML(analysis: PromptAnalysis, improvedPrompt: string): string {
    const score = analysis.score;
    const scoreColor = score >= 80 ? '#4CAF50' : score >= 60 ? '#FF9800' : '#F44336';
    
    return `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: var(--vscode-font-family); padding: 20px; color: var(--vscode-foreground); }
        .score { font-size: 48px; font-weight: bold; color: ${scoreColor}; }
        .section { margin: 20px 0; }
        .principle { margin: 10px 0; padding: 10px; background: var(--vscode-editor-background); border-left: 3px solid; }
        .pass { border-color: #4CAF50; }
        .fail { border-color: #F44336; }
        .warning { border-color: #FF9800; }
        .improved-prompt { background: var(--vscode-textCodeBlock-background); padding: 15px; border-radius: 5px; margin: 10px 0; }
        .copy-btn { padding: 8px 16px; background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; cursor: pointer; }
        .copy-btn:hover { opacity: 0.8; }
    </style>
</head>
<body>
    <h1>Prompt Analysis</h1>
    <div class="score">Score: ${score}/100</div>
    
    <div class="section">
        <h2>Core Principles Check</h2>
        ${analysis.principleChecks.map(p => `
            <div class="principle ${p.status}">
                <strong>${p.principle}</strong>: ${p.status === 'pass' ? '✓' : p.status === 'warning' ? '⚠' : '✗'}
                <p>${p.feedback}</p>
            </div>
        `).join('')}
    </div>

    <div class="section">
        <h2>Suggested Improvements</h2>
        <ul>
            ${analysis.suggestions.map(s => `<li>${s}</li>`).join('')}
        </ul>
    </div>

    <div class="section">
        <h2>Improved Prompt</h2>
        <div class="improved-prompt">
            <pre>${escapeHtml(improvedPrompt)}</pre>
        </div>
        <button class="copy-btn" onclick="copyToClipboard()">Copy Improved Prompt</button>
    </div>

    <script>
        function copyToClipboard() {
            const text = \`${escapeHtml(improvedPrompt)}\`;
            navigator.clipboard.writeText(text).then(() => {
                alert('Copied to clipboard!');
            });
        }
    </script>
</body>
</html>`;
}

function showMLRecommendationsPanel(recommendations: MLRecommendations, originalPrompt: string) {
    const panel = vscode.window.createWebviewPanel(
        'mlRecommendations',
        'ML-Powered Prompt Recommendations',
        vscode.ViewColumn.Beside,
        { enableScripts: true, retainContextWhenHidden: true }
    );

    const html = generateMLRecommendationsHTML(recommendations, originalPrompt);
    panel.webview.html = html;

    // Handle messages from webview
    panel.webview.onDidReceiveMessage(
        async message => {
            switch (message.command) {
                case 'copyPrompt':
                    await vscode.env.clipboard.writeText(message.text);
                    vscode.window.showInformationMessage('Prompt copied to clipboard!');
                    break;
                case 'copySettings':
                    await vscode.env.clipboard.writeText(message.text);
                    vscode.window.showInformationMessage('Settings copied to clipboard!');
                    break;
            }
        },
        undefined,
        []
    );
}

function generateMLRecommendationsHTML(rec: MLRecommendations, original: string): string {
    const confidenceColor = rec.confidence >= 80 ? '#4CAF50' : rec.confidence >= 60 ? '#FF9800' : '#F44336';
    
    return `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body { 
            font-family: var(--vscode-font-family); 
            padding: 20px; 
            color: var(--vscode-foreground);
            background: var(--vscode-editor-background);
        }
        .header { 
            border-bottom: 2px solid var(--vscode-textLink-foreground);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .confidence { 
            font-size: 24px; 
            font-weight: bold; 
            color: ${confidenceColor};
            margin: 10px 0;
        }
        .section { 
            margin: 25px 0; 
            padding: 15px;
            background: var(--vscode-textCodeBlock-background);
            border-radius: 5px;
            border-left: 4px solid var(--vscode-textLink-foreground);
        }
        .section h2 { 
            color: var(--vscode-textLink-foreground);
            margin-top: 0;
        }
        .goal { 
            font-size: 18px; 
            font-weight: bold; 
            color: var(--vscode-textLink-foreground);
            padding: 10px;
            background: var(--vscode-editor-background);
            border-radius: 5px;
        }
        .model-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 10px 0;
        }
        .info-item {
            padding: 8px;
            background: var(--vscode-editor-background);
            border-radius: 3px;
        }
        .param-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin: 10px 0;
        }
        .param-item {
            padding: 10px;
            background: var(--vscode-editor-background);
            border-radius: 3px;
        }
        .param-label {
            font-weight: bold;
            color: var(--vscode-textLink-foreground);
        }
        .param-value {
            font-size: 20px;
            margin: 5px 0;
        }
        .context-item {
            padding: 10px;
            margin: 8px 0;
            background: var(--vscode-editor-background);
            border-radius: 3px;
            border-left: 3px solid;
        }
        .priority-high { border-color: #F44336; }
        .priority-medium { border-color: #FF9800; }
        .priority-low { border-color: #4CAF50; }
        .missing {
            background: #F44336;
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
            display: inline-block;
            margin: 5px 5px 5px 0;
        }
        .filled-prompt {
            background: var(--vscode-textCodeBlock-background);
            padding: 15px;
            border-radius: 5px;
            white-space: pre-wrap;
            font-family: var(--vscode-editor-font-family);
            margin: 10px 0;
        }
        .checklist-item {
            padding: 8px;
            margin: 5px 0;
            background: var(--vscode-editor-background);
            border-radius: 3px;
        }
        .checklist-item:before {
            content: "☐ ";
            margin-right: 8px;
        }
        .btn {
            padding: 10px 20px;
            margin: 5px;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .btn:hover {
            opacity: 0.8;
        }
        .btn-primary {
            background: var(--vscode-textLink-foreground);
        }
        .iteration-step {
            padding: 8px;
            margin: 5px 0;
            background: var(--vscode-editor-background);
            border-radius: 3px;
            border-left: 3px solid var(--vscode-textLink-foreground);
        }
        code {
            background: var(--vscode-textCodeBlock-background);
            padding: 2px 6px;
            border-radius: 3px;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            margin-left: 8px;
        }
        .badge-success { background: #4CAF50; color: white; }
        .badge-warning { background: #FF9800; color: white; }
        .badge-info { background: #2196F3; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 ML-Powered Prompt Recommendations</h1>
        <div class="confidence">Confidence Score: ${rec.confidence}%</div>
    </div>

    <div class="section">
        <h2>🎯 Detected Goal</h2>
        <div class="goal">${rec.goal.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
        <p>The ML advisor has identified your prompt's primary goal and optimized recommendations accordingly.</p>
    </div>

    <div class="section">
        <h2>🤖 Recommended Model</h2>
        <div class="model-info">
            <div class="info-item">
                <strong>Model:</strong> <code>${rec.suggestedModel.model}</code>
            </div>
            <div class="info-item">
                <strong>Cost:</strong> ${rec.suggestedModel.costEstimate}
            </div>
            <div class="info-item">
                <strong>Speed:</strong> ${rec.suggestedModel.speedEstimate}
            </div>
            <div class="info-item">
                <strong>Alternatives:</strong> ${rec.suggestedModel.alternatives.join(', ')}
            </div>
        </div>
        <p><strong>Reason:</strong> ${rec.suggestedModel.reason}</p>
        <button class="btn" onclick="copySettings()">Copy Model Settings</button>
    </div>

    <div class="section">
        <h2>🌡️ Recommended Parameters</h2>
        <div class="param-grid">
            <div class="param-item">
                <div class="param-label">Temperature</div>
                <div class="param-value">${rec.temperature}</div>
                <small>${rec.temperature < 0.3 ? 'Deterministic (good for code/facts)' : rec.temperature > 0.7 ? 'Creative (good for brainstorming)' : 'Balanced'}</small>
            </div>
            ${rec.samplingParams.topP ? `
            <div class="param-item">
                <div class="param-label">Top-P</div>
                <div class="param-value">${rec.samplingParams.topP}</div>
            </div>
            ` : ''}
            ${rec.samplingParams.topK ? `
            <div class="param-item">
                <div class="param-label">Top-K</div>
                <div class="param-value">${rec.samplingParams.topK}</div>
            </div>
            ` : ''}
            ${rec.samplingParams.frequencyPenalty ? `
            <div class="param-item">
                <div class="param-label">Frequency Penalty</div>
                <div class="param-value">${rec.samplingParams.frequencyPenalty}</div>
            </div>
            ` : ''}
        </div>
        <button class="btn" onclick="copySettings()">Copy Parameter Settings</button>
    </div>

    <div class="section">
        <h2>📋 Required Context</h2>
        <p>The ML advisor recommends providing the following context for best results:</p>
        ${rec.requiredContext.map(ctx => `
            <div class="context-item priority-${ctx.priority}">
                <strong>${ctx.type.replace(/_/g, ' ').toUpperCase()}</strong> 
                <span class="badge badge-${ctx.priority === 'high' ? 'warning' : ctx.priority === 'medium' ? 'info' : 'success'}">${ctx.priority}</span>
                <p>${ctx.description}</p>
                <small><strong>Example:</strong> ${ctx.example}</small>
            </div>
        `).join('')}
        ${rec.requiredContext.length === 0 ? '<p>✅ Your prompt has sufficient context!</p>' : ''}
    </div>

    <div class="section">
        <h2>📝 Recommended Format</h2>
        <p><strong>Format:</strong> <code>${rec.suggestedFormat.format}</code></p>
        <p><strong>Structure:</strong> ${rec.suggestedFormat.structure}</p>
        <p><strong>Reason:</strong> ${rec.suggestedFormat.reason}</p>
        <details>
            <summary>Example Format</summary>
            <pre style="background: var(--vscode-textCodeBlock-background); padding: 10px; border-radius: 5px; margin-top: 10px;">${escapeHtml(rec.suggestedFormat.example)}</pre>
        </details>
    </div>

    ${rec.missingElements.length > 0 ? `
    <div class="section">
        <h2>⚠️ Missing Elements</h2>
        <p>The ML advisor identified these missing elements:</p>
        ${rec.missingElements.map(elem => `<span class="missing">${escapeHtml(elem)}</span>`).join('')}
    </div>
    ` : ''}

    <div class="section">
        <h2>✨ Enhanced Prompt</h2>
        <p>The ML advisor has filled in missing parts and optimized your prompt:</p>
        <div class="filled-prompt">${escapeHtml(rec.filledPrompt)}</div>
        <button class="btn btn-primary" onclick="copyPrompt()">Copy Enhanced Prompt</button>
    </div>

    <div class="section">
        <h2>🔄 Iteration Strategy</h2>
        <p><strong>Expected Iterations:</strong> ${rec.iterationStrategy.expectedIterations}</p>
        <h3>Steps:</h3>
        ${rec.iterationStrategy.steps.map((step, i) => `
            <div class="iteration-step">${step}</div>
        `).join('')}
        ${rec.iterationStrategy.refinementPoints.length > 0 ? `
        <h3>Refinement Points:</h3>
        <ul>
            ${rec.iterationStrategy.refinementPoints.map(point => `<li>${point}</li>`).join('')}
        </ul>
        ` : ''}
    </div>

    <div class="section">
        <h2>✅ Verification Checklist</h2>
        <p>Use this checklist to verify the AI's response:</p>
        ${rec.verificationChecklist.map(item => `
            <div class="checklist-item">${escapeHtml(item)}</div>
        `).join('')}
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        
        function copyPrompt() {
            const prompt = \`${escapeHtml(rec.filledPrompt)}\`;
            vscode.postMessage({
                command: 'copyPrompt',
                text: prompt
            });
        }
        
        function copySettings() {
            const settings = \`Model: ${rec.suggestedModel.model}
Temperature: ${rec.temperature}
${rec.samplingParams.topP ? `Top-P: ${rec.samplingParams.topP}` : ''}
${rec.samplingParams.topK ? `Top-K: ${rec.samplingParams.topK}` : ''}
${rec.samplingParams.frequencyPenalty ? `Frequency Penalty: ${rec.samplingParams.frequencyPenalty}` : ''}
Format: ${rec.suggestedFormat.format}\`;
            vscode.postMessage({
                command: 'copySettings',
                text: settings
            });
        }
    </script>
</body>
</html>`;
}

function escapeHtml(text: string): string {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;')
        .replace(/\n/g, '<br>');
}

function setupAutoSuggest(context: vscode.ExtensionContext, analyzer: PromptAnalyzer, templates: PromptTemplates) {
    // This would implement auto-suggestions as you type
    // For now, it's a placeholder for future enhancement
}

export function deactivate() {}

