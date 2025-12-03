# Quick Start: Using PyTorch Prompt Optimizer

## ✅ Yes, You Can Use It Now!

The PyTorch Latent Prompt Optimizer is **fully integrated** with your prompt assistant. Here's how to use it:

## Quick Setup (3 Steps)

### 1. Install Dependencies

```bash
cd prompt-assistant/pytorch_optimizer
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python start_api.py
```

You should see:
```
Starting Prompt Optimizer API Server...
Server will be available at http://127.0.0.1:8000
Model loaded successfully
```

**Note**: First run downloads ~1.5GB of models (one-time download).

### 3. Use Your Extension

That's it! Your VS Code extension will automatically:
- Detect the PyTorch API server
- Use it for prompt optimization
- Fall back to transformers.js if server is unavailable

## How It Works

When you use the prompt assistant:

1. **You enter**: "Create a website"
2. **PyTorch Optimizer** transforms it to:
   ```
   Create a modern, responsive website using HTML5, CSS3, and JavaScript. 
   The website should be mobile-friendly, follow WCAG accessibility standards, 
   and implement SEO best practices.
   ```
3. **You get**: The ideal, optimized prompt ready to copy!

## Testing

### Test the API Directly

```bash
curl -X POST http://127.0.0.1:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"raw_prompt": "Create a function"}'
```

### Test in VS Code

1. Open VS Code/Cursor
2. Type a prompt: "Fix my code"
3. Select it
4. Press `Ctrl+Shift+M` (or Command Palette → "Get ML-Powered Recommendations")
5. See the PyTorch-optimized prompt!

## What Makes It Different?

### Before (transformers.js):
- Lightweight, fast
- Rule-based enhancements
- Good for simple prompts

### Now (PyTorch Optimizer):
- **Learns from contrastive pairs** (raw → ideal prompts)
- **Latent space optimization** (understands prompt semantics)
- **Rule seeding** (starts with heuristic rules, learns better patterns)
- **Feedback integration** (learns from your edits)
- **Higher quality** optimizations for complex prompts

## Architecture

```
Your Prompt
    ↓
[VS Code Extension]
    ↓
[PromptEnhancer] → Checks: PyTorch API available?
    ↓ YES                    ↓ NO
[PyTorch API]          [transformers.js]
    ↓
[Latent Prompt Optimizer]
    ├─ Encoder (RoBERTa-large): prompt → embedding z
    ├─ Delta Network: z → optimized z'
    └─ Decoder (GPT-2): z' → optimized prompt
    ↓
Ideal Prompt ✨
```

## Next Steps

1. **Start using it**: Just start the API server and use your extension normally
2. **Train on your data**: See `train.py` for training on your prompt pairs
3. **Submit feedback**: When you edit prompts, the model learns from your changes
4. **Monitor performance**: Check confidence scores in API responses

## Troubleshooting

**Q: Server won't start**
- Check Python version: `python --version` (needs 3.8+)
- Install dependencies: `pip install -r requirements.txt`

**Q: Extension not using PyTorch**
- Check server is running: `curl http://127.0.0.1:8000/health`
- Extension automatically falls back if unavailable

**Q: Slow first request**
- Normal! First request loads models (~2-3 seconds)
- Subsequent requests are fast (~0.5-1 second)

## API Documentation

When server is running, visit: `http://127.0.0.1:8000/docs`

Full integration guide: See `INTEGRATION.md`



