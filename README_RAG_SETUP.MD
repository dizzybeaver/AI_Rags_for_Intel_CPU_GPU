# Lambda Project RAG System

Local Retrieval-Augmented Generation (RAG) system for the Lambda Execution Engine project. Combines semantic search with LM Studio for context-aware code assistance.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure for Your Hardware

**For Dual Intel Arc B580 Setup:**

Edit `lambda_indexer.py`, `rag_server.py`, and `auto_index.py` to set device:

```python
# Use GPU.1 (second Arc B580) for embeddings
indexer = LambdaProjectIndexer(device="GPU.1")

# Or use NPU if available
indexer = LambdaProjectIndexer(device="NPU")
```

Check available devices:
```bash
python gpu_monitor.py --once
```

### 3. Start LM Studio

1. Open LM Studio
2. Load recommended model (Qwen 2.5 Coder 32B for dual GPU, or 14B for single GPU)
3. Settings → GPU → Select your GPU
4. Start Server (port 1234)

### 4. Start RAG System

**Linux/Mac:**
```bash
chmod +x start_lambda_rag.sh
./start_lambda_rag.sh
```

**Windows:**
```batch
start_lambda_rag.bat
```

**Or manually:**
```bash
# Terminal 1: Auto-indexer (background file watching)
python auto_index.py

# Terminal 2: RAG web server
python rag_server.py

# Browser: http://localhost:8000
```

## Usage

### Web Interface (Easiest)

1. Open http://localhost:8000
2. Type your question about the Lambda project
3. Click "Search & Ask LLM"
4. See relevant files + AI-generated answer

### Command Line

```bash
python lambda_ask.py "How does the gateway pattern work?"
python lambda_ask.py "What files handle logging?"
python lambda_ask.py "Explain the SUGA architecture"
```

### Clipboard Enhancement (Advanced)

```bash
# Start clipboard watcher
python vscode_context.py

# Now copy any question ending with "?"
# It will automatically be enhanced with project context
# Paste the enhanced question into LM Studio
```

## File Descriptions

| File | Purpose |
|------|---------|
| `lambda_indexer.py` | Core indexing engine (embeddings) |
| `rag_server.py` | Web UI + API server |
| `lambda_ask.py` | CLI query tool |
| `auto_index.py` | Auto-reindex on file changes |
| `vscode_context.py` | Clipboard context enhancer |
| `gpu_monitor.py` | Check available GPUs/NPU |
| `start_lambda_rag.sh` | Startup script (Linux/Mac) |
| `start_lambda_rag.bat` | Startup script (Windows) |

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Your Hardware                       │
├─────────────────────────────────────────────────┤
│                                                  │
│  GPU 0 (Arc B580):  LM Studio (Qwen 32B)        │
│                     Port 1234                    │
│                                                  │
│  GPU 1 (Arc B580):  Embeddings (BGE-Small)      │
│                     Background indexing          │
│                                                  │
│  NPU:              Alternative for embeddings    │
│                                                  │
│  CPU:              RAG server (port 8000)        │
│                    File watching                 │
│                                                  │
└─────────────────────────────────────────────────┘
```

## How It Works

1. **Indexing**: All project files (.py, .md, .json, etc.) are embedded into vectors
2. **Search**: Your question is embedded and compared to all file vectors
3. **Context**: Top 3-5 most relevant files are retrieved
4. **LLM Query**: Question + file context sent to LM Studio
5. **Answer**: LLM generates response based on actual project code

## Configuration

### Change Embedding Device

In `lambda_indexer.py`, `rag_server.py`, and `auto_index.py`:

```python
# Options:
indexer = LambdaProjectIndexer(device="NPU")     # Intel NPU
indexer = LambdaProjectIndexer(device="GPU.0")   # First Arc B580
indexer = LambdaProjectIndexer(device="GPU.1")   # Second Arc B580
indexer = LambdaProjectIndexer(device="CPU")     # CPU fallback
```

### Change LM Studio Port

In `rag_server.py` and `lambda_ask.py`:

```python
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# Or for dual model setup:
LM_STUDIO_CODE = "http://localhost:1234/v1/chat/completions"  # Fast model
LM_STUDIO_CHAT = "http://localhost:1235/v1/chat/completions"  # Quality model
```

### Change Number of Results

In web UI: Use the dropdown (1-10 files)

In CLI:
```python
# Edit lambda_ask.py
relevant_files = indexer.search(question, top_k=5)  # Change 5 to desired number
```

## Troubleshooting

### "Cannot connect to LM Studio"

Make sure LM Studio is:
1. Running
2. Has a model loaded
3. Server is started (shows "Running on port 1234")

### "Module not found: lambda_indexer"

Make sure all Python files are in the same directory:
```bash
ls *.py
# Should show: lambda_indexer.py, rag_server.py, lambda_ask.py, etc.
```

### "NPU not found"

Change device to CPU or GPU:
```python
indexer = LambdaProjectIndexer(device="CPU")
```

### Index not updating

Manually rebuild:
```bash
python -c "from lambda_indexer import LambdaProjectIndexer; i = LambdaProjectIndexer(); i.index_project('./'); i.save_index('lambda_index')"
```

### Slow performance

**For dual B580 setup:**
- Use GPU.1 for embeddings (keep GPU.0 free for LLM)
- Reduce `top_k` to 3 files instead of 5
- Use smaller model in LM Studio for faster responses

## Performance Tips

### Dual Arc B580 (24GB) - Optimal Setup

```
GPU 0: Qwen 2.5 32B Instruct (LM Studio)
  - 18GB VRAM
  - 30-40 tokens/sec
  - Port 1234

GPU 1: BGE-Small embeddings (Python)
  - 500-1000 files/sec
  - Background indexing
  - No impact on LLM

Result: Fast responses with project context
```

### Single Arc B580 (12GB)

```
GPU: Qwen 2.5 Coder 14B (LM Studio)
  - 10GB VRAM
  - 25-35 tokens/sec
  - Port 1234

NPU/CPU: BGE-Small embeddings
  - Background indexing
  - No GPU impact
```

## API Usage

### Query the RAG Server Directly

```python
import requests

response = requests.post(
    "http://localhost:8000/api/ask",
    json={
        "question": "How does the gateway pattern work?",
        "top_k": 3
    }
)

data = response.json()
print(data['llm_response'])
```

### Reindex Programmatically

```python
import requests

response = requests.post("http://localhost:8000/api/reindex")
print(response.json())
```

## License

Copyright 2025 Joseph Hersey

Licensed under the Apache License, Version 2.0
