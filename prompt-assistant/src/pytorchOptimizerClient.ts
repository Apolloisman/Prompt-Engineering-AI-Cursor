/**
 * TypeScript Client for PyTorch Latent Prompt Optimizer API
 * Connects VS Code extension to Python PyTorch model
 */

export interface OptimizeRequest {
    raw_prompt: string;
    rule_indices?: number[];
    return_embeddings?: boolean;
}

export interface OptimizeResponse {
    optimized_prompt: string;
    raw_embedding?: number[];
    optimized_embedding?: number[];
    confidence: number;
}

export interface FeedbackRequest {
    raw_prompt: string;
    edited_prompt: string;
    success_score: number; // 0.0 to 1.0
    rule_indices?: number[];
}

export interface FeedbackResponse {
    success: boolean;
    message: string;
}

export interface RuleInfo {
    index: number;
    name: string;
    description: string;
}

export class PyTorchOptimizerClient {
    private apiUrl: string;
    private isAvailable: boolean = false;

    constructor(apiUrl: string = 'http://127.0.0.1:8000') {
        this.apiUrl = apiUrl;
    }

    /**
     * Check if the API server is available
     */
    async checkHealth(): Promise<boolean> {
        try {
            const response = await fetch(`${this.apiUrl}/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.isAvailable = data.model_loaded === true;
                return this.isAvailable;
            }
            return false;
        } catch (error) {
            this.isAvailable = false;
            return false;
        }
    }

    /**
     * Optimize a prompt using the PyTorch Latent Prompt Optimizer
     */
    async optimizePrompt(request: OptimizeRequest): Promise<OptimizeResponse> {
        if (!this.isAvailable) {
            const available = await this.checkHealth();
            if (!available) {
                throw new Error('PyTorch Optimizer API is not available. Make sure the server is running.');
            }
        }

        try {
            const response = await fetch(`${this.apiUrl}/optimize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    raw_prompt: request.raw_prompt,
                    rule_indices: request.rule_indices,
                    return_embeddings: request.return_embeddings || false
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Optimization failed');
            }

            return await response.json();
        } catch (error) {
            console.error('PyTorch optimizer error:', error);
            throw error;
        }
    }

    /**
     * Submit user feedback to improve the model
     */
    async submitFeedback(request: FeedbackRequest): Promise<FeedbackResponse> {
        if (!this.isAvailable) {
            const available = await this.checkHealth();
            if (!available) {
                throw new Error('PyTorch Optimizer API is not available.');
            }
        }

        try {
            const response = await fetch(`${this.apiUrl}/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    raw_prompt: request.raw_prompt,
                    edited_prompt: request.edited_prompt,
                    success_score: request.success_score,
                    rule_indices: request.rule_indices
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Feedback submission failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Feedback submission error:', error);
            throw error;
        }
    }

    /**
     * Get available rules
     */
    async getRules(): Promise<RuleInfo[]> {
        try {
            const response = await fetch(`${this.apiUrl}/rules`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch rules');
            }

            const data = await response.json();
            return data.rules || [];
        } catch (error) {
            console.error('Failed to get rules:', error);
            return [];
        }
    }

    /**
     * Check if the client is ready to use
     */
    isReady(): boolean {
        return this.isAvailable;
    }
}



