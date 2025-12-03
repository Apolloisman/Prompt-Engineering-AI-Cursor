import * as vscode from 'vscode';

export interface ChatMessage {
    prompt: string;
    timestamp: number;
    goal?: string;
    referencesMentioned?: string[];
    contextProvided?: string[];
    modelUsed?: string;
}

export interface AttachedReference {
    id: string;
    name: string;
    summary: string;
    content: string;
    source?: string;
    addedAt: number;
    size: number;
}

export interface ChatHistory {
    messages: ChatMessage[];
    sessionStart: number;
    detectedPatterns: string[];
    commonReferences: Map<string, number>;
    commonContext: Map<string, number>;
    attachedReferences: AttachedReference[];
}

export interface ChatHistoryContext {
    recentMessages: ChatMessage[];
    patterns: string[];
    alreadyMentionedReferences: string[];
    alreadyProvidedContext: string[];
    commonReferences: string[];
    sessionDuration: number;
    attachedReferences: AttachedReference[];
    referenceSummaries: string[];
}

export class ChatHistoryManager {
    private history: ChatHistory;
    private maxHistorySize: number = 50;
    private maxReferences: number = 12;
    private maxReferenceChars: number = 120000; // ~120k chars (~80k tokens)
    private storageKey: string = 'promptAssistant.chatHistory';

    constructor(private context: vscode.ExtensionContext) {
        this.history = this.loadHistory();
    }

    addMessage(prompt: string, metadata?: Partial<ChatMessage>): void {
        const message: ChatMessage = {
            prompt,
            timestamp: Date.now(),
            ...metadata
        };

        this.history.messages.push(message);
        
        // Keep only recent messages
        if (this.history.messages.length > this.maxHistorySize) {
            this.history.messages = this.history.messages.slice(-this.maxHistorySize);
        }

        // Update patterns and common items
        this.updatePatterns();
        this.saveHistory();
    }

    getRecentMessages(count: number = 10): ChatMessage[] {
        return this.history.messages.slice(-count);
    }

    getRelevantHistory(currentPrompt: string, lookback: number = 5): ChatMessage[] {
        const recent = this.getRecentMessages(lookback);
        
        // Filter for relevant messages (same topic/goal)
        const currentLower = currentPrompt.toLowerCase();
        return recent.filter(msg => {
            const msgLower = msg.prompt.toLowerCase();
            // Check for keyword overlap
            const currentWords = new Set(currentLower.split(/\s+/));
            const msgWords = new Set(msgLower.split(/\s+/));
            const overlap = [...currentWords].filter(w => msgWords.has(w) && w.length > 3).length;
            return overlap > 2 || msg.goal; // Relevant if shared words or has goal
        });
    }

    getReferencesAlreadyMentioned(): string[] {
        const mentioned = new Set<string>();
        this.history.messages.forEach(msg => {
            if (msg.referencesMentioned) {
                msg.referencesMentioned.forEach(ref => mentioned.add(ref));
            }
        });
        return Array.from(mentioned);
    }

    getContextAlreadyProvided(): string[] {
        const provided = new Set<string>();
        this.history.messages.forEach(msg => {
            if (msg.contextProvided) {
                msg.contextProvided.forEach(ctx => provided.add(ctx));
            }
        });
        return Array.from(provided);
    }

    detectPatterns(): string[] {
        const patterns: string[] = [];
        const goals = new Map<string, number>();
        const languages = new Map<string, number>();
        
        this.history.messages.forEach(msg => {
            if (msg.goal) {
                goals.set(msg.goal, (goals.get(msg.goal) || 0) + 1);
            }
            
            // Detect languages
            const lower = msg.prompt.toLowerCase();
            if (lower.includes('python')) languages.set('python', (languages.get('python') || 0) + 1);
            if (lower.includes('javascript') || lower.includes('js')) languages.set('javascript', (languages.get('javascript') || 0) + 1);
            if (lower.includes('typescript') || lower.includes('ts')) languages.set('typescript', (languages.get('typescript') || 0) + 1);
        });

        // Detect common goals
        goals.forEach((count, goal) => {
            if (count >= 3) {
                patterns.push(`Frequent goal: ${goal.replace(/_/g, ' ')}`);
            }
        });

        // Detect common languages
        languages.forEach((count, lang) => {
            if (count >= 3) {
                patterns.push(`Working with ${lang}`);
            }
        });

        return patterns;
    }

    getCommonReferences(): string[] {
        const refCounts = new Map<string, number>();
        this.history.messages.forEach(msg => {
            if (msg.referencesMentioned) {
                msg.referencesMentioned.forEach(ref => {
                    refCounts.set(ref, (refCounts.get(ref) || 0) + 1);
                });
            }
        });

        return Array.from(refCounts.entries())
            .filter(([_, count]) => count >= 2)
            .sort(([_, a], [__, b]) => b - a)
            .map(([ref, _]) => ref);
    }

    shouldSuggestReference(reference: string): boolean {
        // Don't suggest if already mentioned recently
        const alreadyMentioned = this.getReferencesAlreadyMentioned();
        return !alreadyMentioned.some(ref => 
            ref.toLowerCase().includes(reference.toLowerCase()) || 
            reference.toLowerCase().includes(ref.toLowerCase())
        );
    }

    getSessionDuration(): number {
        return Date.now() - this.history.sessionStart;
    }

    clearHistory(): void {
        this.history = {
            messages: [],
            sessionStart: Date.now(),
            detectedPatterns: [],
            commonReferences: new Map(),
            commonContext: new Map(),
            attachedReferences: []
        };
        this.saveHistory();
    }

    private updatePatterns(): void {
        this.history.detectedPatterns = this.detectPatterns();
        
        // Update common references
        const refCounts = new Map<string, number>();
        this.history.messages.forEach(msg => {
            if (msg.referencesMentioned) {
                msg.referencesMentioned.forEach(ref => {
                    refCounts.set(ref, (refCounts.get(ref) || 0) + 1);
                });
            }
        });
        this.history.commonReferences = refCounts;

        // Update common context
        const ctxCounts = new Map<string, number>();
        this.history.messages.forEach(msg => {
            if (msg.contextProvided) {
                msg.contextProvided.forEach(ctx => {
                    ctxCounts.set(ctx, (ctxCounts.get(ctx) || 0) + 1);
                });
            }
        });
        this.history.commonContext = ctxCounts;
    }

    private loadHistory(): ChatHistory {
        const stored = this.context.globalState.get<ChatHistory>(this.storageKey);
        if (stored) {
            // Reset session if older than 24 hours
            const hoursSinceStart = (Date.now() - stored.sessionStart) / (1000 * 60 * 60);
            if (hoursSinceStart > 24) {
                return {
                    messages: [],
                    sessionStart: Date.now(),
                    detectedPatterns: [],
                    commonReferences: new Map(),
                    commonContext: new Map(),
                    attachedReferences: []
                };
            }
            if (!stored.attachedReferences) {
                stored.attachedReferences = [];
            }
            return stored;
        }
        
        return {
            messages: [],
            sessionStart: Date.now(),
            detectedPatterns: [],
            commonReferences: new Map(),
            commonContext: new Map(),
            attachedReferences: []
        };
    }

    private saveHistory(): void {
        this.context.globalState.update(this.storageKey, this.history);
    }

    // Method to manually add chat history (for when user provides it)
    addChatHistory(messages: string[]): void {
        messages.forEach((msg, index) => {
            this.addMessage(msg, {
                timestamp: Date.now() - (messages.length - index) * 60000 // Stagger timestamps
            });
        });
    }

    // Get summary of history for context
    getHistorySummary(): string {
        if (this.history.messages.length === 0) {
            return 'No chat history available.';
        }

        const patterns = this.detectPatterns();
        const commonRefs = this.getCommonReferences();
        const recentGoals = this.history.messages
            .filter(m => m.goal)
            .slice(-5)
            .map(m => m.goal)
            .filter((v, i, a) => a.indexOf(v) === i); // Unique

        let summary = `Chat History (${this.history.messages.length} messages):\n`;
        
        if (patterns.length > 0) {
            summary += `\nPatterns: ${patterns.join(', ')}\n`;
        }
        
        if (recentGoals.length > 0) {
            summary += `Recent goals: ${recentGoals.join(', ')}\n`;
        }
        
        if (commonRefs.length > 0) {
            summary += `Common references: ${commonRefs.slice(0, 5).join(', ')}\n`;
        }

        return summary;
    }

    getContextForAnalysis(): ChatHistoryContext {
        return {
            recentMessages: this.getRecentMessages(10),
            patterns: this.detectPatterns(),
            alreadyMentionedReferences: this.getReferencesAlreadyMentioned(),
            alreadyProvidedContext: this.getContextAlreadyProvided(),
            commonReferences: this.getCommonReferences(),
            sessionDuration: this.getSessionDuration(),
            attachedReferences: this.getAttachedReferences(5),
            referenceSummaries: this.getAttachedReferences(5).map(ref => `${ref.name}: ${ref.summary}`)
        };
    }

    getAttachedReferences(limit: number = 5): AttachedReference[] {
        if (!this.history.attachedReferences || this.history.attachedReferences.length === 0) {
            return [];
        }
        return this.history.attachedReferences.slice(-limit);
    }

    listAllReferences(): AttachedReference[] {
        return this.history.attachedReferences || [];
    }

    attachReference(name: string, content: string, source?: string): AttachedReference | null {
        const trimmed = (content || '').trim();
        if (!trimmed) {
            return null;
        }

        const normalized = trimmed.length > this.maxReferenceChars
            ? `${trimmed.slice(0, this.maxReferenceChars)}\n\n...[truncated]`
            : trimmed;

        const reference: AttachedReference = {
            id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
            name: name || `Reference ${this.history.attachedReferences.length + 1}`,
            summary: this.createReferenceSummary(normalized),
            content: normalized,
            source,
            addedAt: Date.now(),
            size: normalized.length
        };

        if (!this.history.attachedReferences) {
            this.history.attachedReferences = [];
        }

        this.history.attachedReferences.push(reference);

        if (this.history.attachedReferences.length > this.maxReferences) {
            this.history.attachedReferences = this.history.attachedReferences.slice(-this.maxReferences);
        }

        this.saveHistory();
        return reference;
    }

    clearReferences(): void {
        this.history.attachedReferences = [];
        this.saveHistory();
    }

    private createReferenceSummary(content: string): string {
        const cleaned = content.replace(/\s+/g, ' ').trim();
        if (!cleaned) {
            return 'Reference provided with no readable content.';
        }

        const sentences = cleaned.split(/(?<=[.!?])\s+/).slice(0, 3);
        let summary = sentences.join(' ');
        if (summary.length > 500) {
            summary = `${summary.slice(0, 500)}...`;
        }

        const keyBullets = content.split('\n').filter(line => line.trim().startsWith('-') || line.trim().startsWith('*')).slice(0, 2);
        if (keyBullets.length > 0) {
            summary += ` Highlights: ${keyBullets.map(b => b.replace(/^[\-\*\s]+/, '').trim()).join(' | ')}`;
        }

        return summary;
    }
}

