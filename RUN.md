# How to Run the Extension

## Quick Start

### Step 1: Install Dependencies

Open a terminal in the `prompt-assistant` directory and run:

```bash
cd prompt-assistant
npm install
```

This will install:
- TypeScript compiler
- VS Code extension types
- Node.js types

### Step 2: Compile TypeScript

```bash
npm run compile
```

This compiles all TypeScript files from `src/` to `out/`.

### Step 3: Run the Extension

**Option A: Using VS Code/Cursor (Recommended)**

1. Open the `prompt-assistant` folder in Cursor/VS Code
2. Press `F5` (or go to Run → Start Debugging)
3. A new Extension Development Host window will open
4. The extension is now active in that window!

**Option B: Using Command Line**

```bash
# First, compile
npm run compile

# Then package (if you have vsce installed)
npm install -g @vscode/vsce
vsce package

# Install the .vsix file
code --install-extension prompt-assistant-0.1.0.vsix
```

## Testing the Extension

Once the Extension Development Host window opens:

1. **Test ML Recommendations:**
   - Type a prompt like "Fix my code" in a text file
   - Select it
   - Press `Ctrl+Shift+M` (or `Cmd+Shift+M` on Mac)
   - Or Command Palette → "Get ML-Powered Recommendations"

2. **Test Prompt Analysis:**
   - Select a prompt
   - Right-click → "Analyze Current Prompt"
   - Or press `Ctrl+Shift+A`

3. **Test Clipboard Improvement:**
   - Copy some text
   - Click the "✨ Improve Prompt" button in status bar
   - Or press `Ctrl+Shift+I`

4. **Test Templates:**
   - Command Palette → "Insert Prompt Template"
   - Select a template

5. **Test Guide Viewer:**
   - Command Palette → "Show Prompt Engineering Guide"

## Troubleshooting

### "Cannot find module" errors
- Run `npm install` again
- Make sure you're in the `prompt-assistant` directory

### Extension not loading
- Check Output panel for errors
- Make sure `out/extension.js` exists (run `npm run compile`)
- Restart the Extension Development Host

### TypeScript errors
- Run `npm run compile` to see errors
- Fix any type errors
- Recompile

### Commands not appearing
- Check `package.json` has all commands registered
- Reload the Extension Development Host window
- Check activation events are correct

## Development Mode

For active development:

```bash
# Terminal 1: Watch mode (auto-compiles on changes)
npm run watch

# Terminal 2: Run extension
# Press F5 in VS Code/Cursor
```

Now any changes you make will auto-compile!

## Building for Distribution

```bash
# Install vsce if not already installed
npm install -g @vscode/vsce

# Package the extension
vsce package

# This creates: prompt-assistant-0.1.0.vsix
```

## File Structure

```
prompt-assistant/
├── src/              # TypeScript source files
│   ├── extension.ts
│   ├── mlAdvisor.ts
│   ├── promptAnalyzer.ts
│   ├── promptTemplates.ts
│   └── guideViewer.ts
├── out/              # Compiled JavaScript (generated)
├── package.json      # Extension manifest
├── tsconfig.json     # TypeScript config
└── .vscode/          # VS Code config
    ├── launch.json   # Debug configuration
    └── tasks.json    # Build tasks
```

## Next Steps

1. ✅ Install dependencies: `npm install`
2. ✅ Compile: `npm run compile`
3. ✅ Run: Press `F5` in VS Code/Cursor
4. ✅ Test all features
5. ✅ Start using it!

## Quick Commands Reference

- `F5` - Run extension in new window
- `Ctrl+Shift+M` - Get ML recommendations
- `Ctrl+Shift+I` - Improve clipboard prompt
- `Ctrl+Shift+A` - Analyze prompt
- `Ctrl+Shift+P` - Command Palette





