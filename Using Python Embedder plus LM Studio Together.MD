# Using Python Embedder + LM Studio Together
## Complete RAG Integration Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Workflow                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  You ask question → Python RAG System → Searches embeddings │
│                           ↓                                  │
│                   Finds relevant files                       │
│                           ↓                                  │
│              Builds context from files                       │
│                           ↓                                  │
│         Sends question + context → LM Studio API            │
│                           ↓                                  │
│              LLM generates answer                            │
│                           ↓                                  │
│                 You get response                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Running Simultaneously:
┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  LM Studio     │  │  Python RAG     │  │  NPU Indexer     │
│  (Port 1234)   │  │  (Port 8000)    │  │  (Background)    │
│                │  │                 │  │                  │
│  ARC B580 GPU  │  │  Web Server     │  │  Auto-updates    │
│  Qwen 14B      │  │  Search API     │  │  on file change  │
└────────────────┘  └─────────────────┘  └──────────────────┘
```

---

## Method 1: Web Interface (RECOMMENDED - Easiest)

### Complete System with Web UI

This creates a simple web interface that automatically searches your project and sends results to LM Studio.

#### Step 1: Create the RAG Server

Save as `rag_server.py`:

```python
#!/usr/bin/env python3
"""
Lambda Project RAG Server
Runs on: http://localhost:8000
Connects to: LM Studio at http://localhost:1234
"""

from flask import Flask, request, jsonify, render_template_string
from lambda_indexer import LambdaProjectIndexer
import requests
import os

app = Flask(__name__)

# Initialize indexer (uses NPU)
print("Loading embeddings on NPU...")
indexer = LambdaProjectIndexer(device="NPU")

# Load pre-built index or create new one
if os.path.exists("lambda_index_embeddings.npz"):
    indexer.load_index("lambda_index")
    print("✓ Loaded existing index")
else:
    print("Building new index...")
    indexer.index_project("./")  # Current directory
    indexer.save_index("lambda_index")
    print("✓ Index created")

# LM Studio API endpoint
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Lambda RAG Assistant</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .input-group {
            margin: 20px 0;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #0056b3;
        }
        .results {
            margin-top: 20px;
        }
        .relevant-files {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .file-item {
            padding: 8px;
            margin: 5px 0;
            background: white;
            border-left: 3px solid #28a745;
            border-radius: 3px;
        }
        .response {
            background: #e7f3ff;
            padding: 15px;
            border-radius: 4px;
            margin-top: 10px;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
        }
        .loading {
            display: none;
            color: #007bff;
            margin: 10px 0;
        }
        .settings {
            background: #fff3cd;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Lambda Project RAG Assistant</h1>
        
        <div class="settings">
            <strong>System Status:</strong><br>
            📊 Embedder: NPU (Background)<br>
            🤖 LLM: LM Studio (Port 1234)<br>
            📁 Indexed Files: <span id="file-count">Loading...</span>
        </div>

        <div class="input-group">
            <label for="question">Ask a question about the Lambda Project:</label>
            <textarea id="question" rows="3" placeholder="e.g., How does the gateway pattern work in this project?"></textarea>
        </div>

        <div class="input-group">
            <label for="num-files">Number of relevant files to retrieve:</label>
            <input type="number" id="num-files" value="3" min="1" max="10" style="width: 100px;">
        </div>

        <button onclick="askQuestion()">🔍 Search & Ask LLM</button>
        
        <div class="loading" id="loading">⏳ Searching project files and querying LLM...</div>

        <div class="results" id="results"></div>
    </div>

    <script>
        // Get file count
        fetch('/api/stats')
            .then(r => r.json())
            .then(data => {
                document.getElementById('file-count').textContent = data.indexed_files;
            });

        async function askQuestion() {
            const question = document.getElementById('question').value;
            const numFiles = document.getElementById('num-files').value;
            
            if (!question) {
                alert('Please enter a question');
                return;
            }

            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';

            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: question,
                        top_k: parseInt(numFiles)
                    })
                });

                const data = await response.json();

                let html = '<div class="relevant-files"><h3>📂 Relevant Files Found:</h3>';
                data.relevant_files.forEach(file => {
                    html += `<div class="file-item">
                        <strong>${file.file}</strong> (${(file.score * 100).toFixed(1)}% match)
                    </div>`;
                });
                html += '</div>';

                html += '<div class="response"><h3>🤖 LLM Response:</h3>' + 
                        data.llm_response + '</div>';

                document.getElementById('results').innerHTML = html;
            } catch (error) {
                document.getElementById('results').innerHTML = 
                    '<div style="color: red;">Error: ' + error.message + '</div>';
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/stats')
def stats():
    """Get system stats"""
    return jsonify({
        'indexed_files': len(indexer.embeddings),
        'status': 'ready'
    })

@app.route('/api/ask', methods=['POST'])
def ask():
    """Main RAG endpoint"""
    data = request.json
    question = data.get('question', '')
    top_k = data.get('top_k', 3)
    
    # Step 1: Search for relevant files
    relevant_files = indexer.search(question, top_k=top_k)
    
    # Step 2: Build context from files
    context = ""
    for result in relevant_files:
        filepath = result['metadata']['absolute_path']
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Limit size per file
                if len(content) > 2000:
                    content = content[:2000] + "\n... (truncated)"
                context += f"\n\n=== {result['file']} (relevance: {result['score']:.2f}) ===\n{content}\n"
        except Exception as e:
            context += f"\n\n=== {result['file']} ===\n[Error reading file: {e}]\n"
    
    # Step 3: Build prompt for LLM
    system_prompt = """You are a helpful assistant with access to the Lambda Execution Engine project files. 
Answer questions based on the provided context from the project files. 
If the context doesn't contain the answer, say so clearly."""
    
    user_prompt = f"""Context from relevant project files:
{context}

---

Question: {question}

Answer based on the context above. Be specific and reference the files when relevant."""
    
    # Step 4: Query LM Studio
    try:
        llm_response = requests.post(
            LM_STUDIO_URL,
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            },
            timeout=60
        )
        
        llm_response.raise_for_status()
        llm_data = llm_response.json()
        llm_answer = llm_data['choices'][0]['message']['content']
        
    except requests.exceptions.ConnectionError:
        llm_answer = "❌ Error: Cannot connect to LM Studio. Make sure it's running on port 1234."
    except Exception as e:
        llm_answer = f"❌ Error querying LLM: {str(e)}"
    
    return jsonify({
        'relevant_files': [
            {'file': r['file'], 'score': r['score']} 
            for r in relevant_files
        ],
        'llm_response': llm_answer
    })

@app.route('/api/reindex', methods=['POST'])
def reindex():
    """Rebuild the index"""
    try:
        indexer.index_project("./")
        indexer.save_index("lambda_index")
        return jsonify({'status': 'success', 'message': 'Index rebuilt'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Lambda RAG Server Starting")
    print("="*60)
    print(f"📊 Indexed files: {len(indexer.embeddings)}")
    print(f"🌐 Web UI: http://localhost:8000")
    print(f"🤖 LM Studio: {LM_STUDIO_URL}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8000, debug=False)
```

#### Step 2: Start Everything

**Terminal 1 - Start LM Studio:**
```bash
# Open LM Studio GUI
# Load Qwen 2.5 Coder 14B
# Click "Start Server" (should show port 1234)
```

**Terminal 2 - Start RAG Server:**
```bash
# Install dependencies first (one time)
pip install flask requests optimum-intel openvino-dev

# Start the RAG server
python rag_server.py
```

**Terminal 3 - (Optional) Auto-reindex on file changes:**
```bash
python auto_index.py
```

#### Step 3: Use It!

1. Open browser: http://localhost:8000
2. Type your question: "How does the gateway pattern work?"
3. Click "Search & Ask LLM"
4. See relevant files + LLM answer with context

---

## Method 2: CLI Tool (For Terminal Users)

### Command-Line RAG Assistant

Save as `lambda_ask.py`:

```python
#!/usr/bin/env python3
"""
Command-line RAG assistant for Lambda Project
Usage: python lambda_ask.py "your question here"
"""

import sys
import requests
from lambda_indexer import LambdaProjectIndexer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

def ask_question(question: str, top_k: int = 3):
    """Ask a question with RAG context"""
    
    # Initialize indexer
    console.print("[yellow]Loading index...[/yellow]")
    indexer = LambdaProjectIndexer(device="NPU")
    indexer.load_index("lambda_index")
    
    # Search for relevant files
    console.print(f"[cyan]Searching {len(indexer.embeddings)} files...[/cyan]")
    relevant_files = indexer.search(question, top_k=top_k)
    
    # Show found files
    console.print("\n[green]📂 Relevant Files:[/green]")
    for i, result in enumerate(relevant_files, 1):
        console.print(f"  {i}. {result['file']} ({result['score']:.2%})")
    
    # Build context
    context = ""
    for result in relevant_files:
        filepath = result['metadata']['absolute_path']
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()[:2000]  # Limit per file
            context += f"\n\n=== {result['file']} ===\n{content}\n"
    
    # Query LLM
    console.print("\n[yellow]Querying LLM...[/yellow]")
    
    prompt = f"""Context from Lambda Project files:
{context}

Question: {question}

Answer based on the context above."""
    
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Answer based on the provided Lambda project context."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            },
            timeout=60
        )
        
        answer = response.json()['choices'][0]['message']['content']
        
        # Display answer
        console.print("\n")
        console.print(Panel(
            Markdown(answer),
            title="🤖 LLM Response",
            border_style="blue"
        ))
        
    except requests.exceptions.ConnectionError:
        console.print("[red]❌ Cannot connect to LM Studio on port 1234[/red]")
        console.print("[yellow]Make sure LM Studio is running with server enabled[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Usage: python lambda_ask.py \"your question\"[/red]")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    ask_question(question)
```

**Usage:**
```bash
# Install rich for pretty output
pip install rich

# Ask questions directly from terminal
python lambda_ask.py "How does the gateway pattern work?"
python lambda_ask.py "What files handle logging?"
python lambda_ask.py "Show me the cache implementation"
```

---

## Method 3: Direct Python API (For Your Own Scripts)

### Use in Your Own Code

```python
# your_script.py

from lambda_indexer import LambdaProjectIndexer
import requests

# Initialize once
indexer = LambdaProjectIndexer(device="NPU")
indexer.load_index("lambda_index")

def ask_with_context(question: str):
    """Ask LLM with automatic context retrieval"""
    
    # Get relevant files
    relevant = indexer.search(question, top_k=3)
    
    # Build context
    context = ""
    for r in relevant:
        with open(r['metadata']['absolute_path'], 'r') as f:
            context += f"\n\n=== {r['file']} ===\n{f.read()[:2000]}\n"
    
    # Query LM Studio
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        }
    )
    
    return response.json()['choices'][0]['message']['content']

# Use it
answer = ask_with_context("Explain the SUGA architecture")
print(answer)
```

---

## Method 4: VSCode Extension Style (Advanced)

### Auto-Context in Your Editor

Save as `vscode_context.py`:

```python
#!/usr/bin/env python3
"""
Watch clipboard for questions, automatically add context
"""

import pyperclip
import time
from lambda_indexer import LambdaProjectIndexer

indexer = LambdaProjectIndexer(device="NPU")
indexer.load_index("lambda_index")

last_clipboard = ""

print("👀 Watching clipboard for questions...")
print("Copy any question and it will be enhanced with project context\n")

while True:
    try:
        current = pyperclip.paste()
        
        # New question detected
        if current != last_clipboard and current.endswith("?"):
            print(f"\n📋 Detected question: {current[:50]}...")
            
            # Search for context
            results = indexer.search(current, top_k=3)
            
            # Build enhanced prompt
            enhanced = f"{current}\n\nRelevant project files:\n"
            for r in results:
                enhanced += f"- {r['file']} ({r['score']:.1%} relevant)\n"
            
            # Put back in clipboard
            pyperclip.copy(enhanced)
            print("✅ Enhanced with context - paste into LM Studio!")
            
            last_clipboard = current
        
        time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n👋 Stopped watching")
        break
```

**Usage:**
```bash
pip install pyperclip
python vscode_context.py

# Now copy any question, it automatically adds context!
```

---

## Complete Startup Script

Save as `start_lambda_rag.sh` (Linux/Mac) or `start_lambda_rag.bat` (Windows):

```bash
#!/bin/bash
# start_lambda_rag.sh

echo "🚀 Starting Lambda RAG System..."

# Check if index exists
if [ ! -f "lambda_index_embeddings.npz" ]; then
    echo "📊 Building initial index..."
    python -c "from lambda_indexer import LambdaProjectIndexer; i = LambdaProjectIndexer(device='NPU'); i.index_project('./'); i.save_index('lambda_index')"
fi

# Start auto-indexer in background
echo "📁 Starting auto-indexer..."
python auto_index.py &
INDEXER_PID=$!

# Start RAG server
echo "🌐 Starting RAG server..."
python rag_server.py &
RAG_PID=$!

echo ""
echo "✅ System Ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Web UI: http://localhost:8000"
echo "🤖 Make sure LM Studio is running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop"

# Wait and cleanup
trap "kill $INDEXER_PID $RAG_PID 2>/dev/null" EXIT
wait
```

**Windows version** (`start_lambda_rag.bat`):
```batch
@echo off
echo Starting Lambda RAG System...

start /B python auto_index.py
start /B python rag_server.py

echo.
echo System Ready!
echo Web UI: http://localhost:8000
echo Make sure LM Studio is running!
echo.
pause
```

---

## Workflow Summary

### Daily Usage:

1. **Morning:**
   ```bash
   # Terminal 1: Start LM Studio GUI, load model, enable server
   # Terminal 2:
   ./start_lambda_rag.sh
   ```

2. **Working:**
   - Write code in your editor
   - Files auto-index in background (NPU)
   - When you have questions:
     - Option A: Open http://localhost:8000, ask there
     - Option B: `python lambda_ask.py "your question"`
     - Option C: Use LM Studio directly (context will be automatically added)

3. **Evening:**
   - Ctrl+C to stop servers
   - Index is saved automatically

### What's Running:

```
Process          | Resource | Purpose
-----------------|----------|---------------------------
LM Studio        | ARC B580 | Main LLM (Qwen 14B)
rag_server.py    | CPU      | Web UI + API
auto_index.py    | NPU      | Background file indexing
Your work        | CPU      | Normal development
```

**Total overhead**: ~500MB RAM, minimal CPU, NPU doing indexing doesn't impact anything else.

---

## Troubleshooting

### "Cannot connect to LM Studio"
```bash
# In LM Studio:
# 1. Load a model
# 2. Click "Local Server" tab
# 3. Click "Start Server"
# 4. Verify it says "Server running on port 1234"
```

### "Module not found: lambda_indexer"
```bash
# Make sure lambda_indexer.py is in the same directory
ls lambda_indexer.py  # Should exist

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### "NPU not found"
```python
# In lambda_indexer.py, change:
indexer = LambdaProjectIndexer(device="NPU")
# To:
indexer = LambdaProjectIndexer(device="CPU")
```

### Index not updating
```bash
# Manually rebuild:
python -c "from lambda_indexer import LambdaProjectIndexer; i = LambdaProjectIndexer(); i.index_project('./'); i.save_index('lambda_index')"
```

---

## Performance Expectations

| Component | Resource | Load |
|-----------|----------|------|
| LM Studio | ARC B580 | High (10GB VRAM) |
| RAG Server | CPU | Low (100MB RAM) |
| Auto-indexer | NPU | Low (background) |
| **Total** | **All** | **Smooth** |

**You can:**
- ✅ Code while LLM generates
- ✅ Files index while you work
- ✅ Search while LLM is thinking
- ✅ Run multiple queries simultaneously

---

This gives you a **complete local AI development environment** with automatic context retrieval! 🎉
