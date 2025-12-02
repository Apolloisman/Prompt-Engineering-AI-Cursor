# Quick Start Guide

## Installation Steps

1. **Install Dependencies**
   ```bash
   cd prompt-assistant
   npm install
   ```

2. **Compile TypeScript**
   ```bash
   npm run compile
   ```

3. **Install Extension in Cursor/VS Code**
   - Open Cursor/VS Code
   - Press `F5` to open Extension Development Host
   - Or package and install:
     ```bash
     npm install -g vsce
     vsce package
     # Then install the .vsix file via Extensions: Install from VSIX
     ```

## Usage Examples

### Example 1: Analyze a Weak Prompt

1. Type or paste this prompt:
   ```
   Fix my code
   ```

2. Select the text

3. Right-click → **"Analyze Current Prompt"**

4. You'll see:
   - Score: ~20/100
   - Issues: Vague language, no context, no structure
   - Improved version with framework applied

### Example 2: Get Suggestions

1. Type:
   ```
   Make the function faster
   ```

2. Select text → Right-click → **"Suggest Improved Prompt"**

3. Choose **"Insert"** to replace with:
   ```
   [Role]: You are a [language] developer
   [Context]: Working on [project] where [background]
   [Task]: Optimize this function for performance
   [Requirements]:
   - Current execution time: [X]ms for [Y] items
   - Target: <[Z]ms
   - Focus on algorithmic improvements
   [Format]: Production-ready code with performance comments
   ```

### Example 3: Use Templates

1. Place cursor where you want the template

2. Command Palette (`Ctrl+Shift+P`) → **"Insert Prompt Template"**

3. Select **"Code Generation (RCTF)"**

4. Fill in the placeholders

## Tips

- **Always select text** before using analyze/suggest commands
- Use **"Show Guide"** command to reference principles anytime
- Templates are great starting points - customize them!
- The analyzer checks 6 core principles automatically
- Higher scores (80+) mean better prompts

## Keyboard Shortcuts

- `Ctrl+Shift+A` / `Cmd+Shift+A`: Analyze prompt
- `Ctrl+Shift+P` / `Cmd+Shift+P`: Suggest improvements (when text selected)

## Troubleshooting

**Extension not working?**
- Make sure TypeScript compiled: `npm run compile`
- Check Output panel for errors
- Restart Cursor/VS Code

**Guide not showing?**
- Ensure `prompt_engineering_guide.txt` exists in workspace root
- Or use the embedded guide (fallback)

**Templates not inserting?**
- Make sure you have an active editor
- Check cursor position



