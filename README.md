# Prompt Engineering Assistant

A Cursor/VS Code extension that helps you create better prompts by automatically analyzing and suggesting improvements based on proven prompt engineering principles.

## Features

### 🎯 **Automatic Prompt Analysis**
- Analyzes your prompts against 6 core principles
- Provides a score (0-100) and detailed feedback
- Identifies areas for improvement

### ✨ **Smart Prompt Suggestions**
- Automatically improves prompts using best practices
- Applies frameworks (RCTF, STAR, Chain-of-Thought)
- Makes prompts concrete and discrete instead of abstract

### 📚 **Prompt Engineering Guide**
- Quick access to the complete prompt engineering guide
- View principles, frameworks, patterns, and templates
- Reference best practices anytime

### 📝 **Prompt Templates**
- Pre-built templates for common scenarios:
  - Code Generation (RCTF)
  - Code Review
  - Problem Solving (STAR)
  - Explanation/Teaching
  - Debugging
  - Chain-of-Thought

### 🤖 **ML-Powered Recommendations** (NEW!)
- **Intelligent goal detection** - Automatically identifies your prompt's purpose
- **Model recommendations** - Suggests the best AI model for your task
- **Parameter optimization** - Recommends temperature, top-p, and other settings
- **Context suggestions** - Tells you what context to provide
- **Format recommendations** - Suggests best output format
- **Auto-fills missing parts** - Completes your prompt intelligently
- **Iteration strategy** - Provides step-by-step refinement plan
- **Verification checklist** - Ensures response quality

## Installation

1. Open Cursor/VS Code
2. Go to Extensions (Ctrl+Shift+X / Cmd+Shift+X)
3. Search for "Prompt Engineering Assistant"
4. Click Install

Or install manually:
```bash
cd prompt-assistant
npm install
npm run compile
# Then use "Extensions: Install from VSIX" command
```

## Usage

### 🚀 **Quick: Improve Prompt for Cursor Chat**

**Best for Cursor's chat interface:**

1. Type your prompt in Cursor's chat input
2. Copy it (`Ctrl+C` / `Cmd+C`)
3. Click the **"✨ Improve Prompt"** button in the status bar (bottom right)
   - Or use keyboard shortcut: `Ctrl+Shift+I` / `Cmd+Shift+I`
4. Choose:
   - **Copy to Clipboard** → Paste back into Cursor chat
   - **Show Analysis** → See detailed feedback
   - **Open in Editor** → Edit before using

### Analyze a Prompt

1. Select the text of your prompt
2. Right-click → **"Analyze Current Prompt"**
   - Or use keyboard shortcut: `Ctrl+Shift+A` (Windows/Linux) or `Cmd+Shift+A` (Mac)
3. View the analysis panel with:
   - Score and principle checks
   - Suggestions for improvement
   - Improved prompt version

### Get Prompt Suggestions

1. Select your prompt text
2. Right-click → **"Suggest Improved Prompt"**
   - Or use keyboard shortcut: `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
3. Choose to:
   - **Insert** the improved prompt
   - **Show Analysis** for detailed feedback
   - **Cancel**

### Get ML-Powered Recommendations (NEW!)

1. Select your prompt text (or it will use clipboard)
2. Open Command Palette → **"Get ML-Powered Recommendations"**
   - Or use keyboard shortcut: `Ctrl+Shift+M` / `Cmd+Shift+M`
3. View comprehensive recommendations:
   - Detected goal
   - Recommended model and parameters
   - Required context
   - Suggested format
   - Enhanced prompt (with missing parts filled)
   - Iteration strategy
   - Verification checklist
4. Copy enhanced prompt or settings to clipboard

### View the Guide

1. Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Type: **"Show Prompt Engineering Guide"**
3. Browse principles, frameworks, patterns, and templates

### Insert a Template

1. Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Type: **"Insert Prompt Template"**
3. Select a template from the list
4. Fill in the placeholders

## Core Principles Checked

The extension analyzes prompts against these principles:

1. **Be Specific and Clear** (20 points)
   - Checks for vague language
   - Looks for specific terms and examples

2. **Keep It Concrete, Not Continuous** (20 points)
   - Identifies abstract phrases
   - Ensures discrete, actionable steps

3. **Provide Context** (15 points)
   - Checks for project/domain context
   - Looks for relevant background information

4. **Structure Your Prompt** (15 points)
   - Verifies clear sections
   - Checks for formatting (bullets, lists)

5. **Include Examples** (15 points)
   - Looks for example inputs/outputs
   - Checks for code blocks

6. **Specify Output Format** (15 points)
   - Ensures format is specified
   - Checks for structure requirements

## Configuration

Open Settings (`Ctrl+,` / `Cmd+,`) and search for "Prompt Assistant":

- **Auto Suggest**: Automatically suggest improvements while typing (default: false)
- **Preferred Framework**: Choose default framework (RCTF, STAR, Chain-of-Thought, or auto-detect)

## Examples

### ML-Powered Recommendations Example

**Input:** "Fix my code"

**ML Advisor Provides:**
- **Goal:** Debugging
- **Recommended Model:** GPT-4 (best for complex debugging)
- **Temperature:** 0.1 (deterministic, precise)
- **Required Context:** 
  - Error message (high priority)
  - Code file (high priority)
  - Environment details (medium priority)
- **Format:** Structured (Problem → Root Cause → Solution → Fixed Code)
- **Enhanced Prompt:** 
  ```
  [Role]: You are an experienced software engineer
  [Context]: Working on [project] where [background]
  [Task]: Debug and fix the following issue
  [Error]: [Error message to be provided]
  [Code]: [Code to be provided]
  [Format]: Problem analysis, root cause, solution, and fixed code
  ```
- **Iteration Strategy:** 2 expected iterations with refinement points
- **Verification Checklist:** 8 items including testing and regression checks

### Before (Weak Prompt)
```
Fix my code
```

### After (Improved Prompt)
```
[Role]: You are a Python developer
[Context]: Working on a web application that processes user data
[Task]: Fix the following Python function that's throwing a TypeError
[Requirements]:
- The function processes user input and should return a dictionary
- Error occurs when input is None
- Add proper error handling
[Format]: Production-ready code with docstrings
```

## Keyboard Shortcuts

- `Ctrl+Shift+M` / `Cmd+Shift+M`: **Get ML-Powered Recommendations** (NEW! Best feature!)
- `Ctrl+Shift+I` / `Cmd+Shift+I`: **Improve prompt from clipboard** (best for Cursor chat!)
- `Ctrl+Shift+A` / `Cmd+Shift+A`: Analyze current prompt
- `Ctrl+Shift+P` / `Cmd+Shift+P`: Suggest improved prompt (when text selected)

## Requirements

- Cursor IDE or VS Code 1.74.0 or higher

## Extension Settings

This extension contributes the following settings:

* `promptAssistant.autoSuggest`: Enable/disable automatic suggestions
* `promptAssistant.preferredFramework`: Set preferred prompt framework

## Known Issues

- Auto-suggest feature is currently a placeholder for future enhancement
- Guide viewer requires the `prompt_engineering_guide.txt` file in the workspace root

## Release Notes

### 0.1.0

Initial release:
- Prompt analysis and scoring
- Automatic prompt improvement suggestions
- Framework application (RCTF, STAR, Chain-of-Thought)
- Prompt templates
- Guide viewer
- Context menu integration

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## License

MIT

## Support

For issues and feature requests, please use the GitHub Issues page.

