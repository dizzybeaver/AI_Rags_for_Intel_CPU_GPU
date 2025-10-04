# Lambda Project Local RAG System

**Intelligent code assistant with semantic search and context-aware AI responses**

A complete local Retrieval-Augmented Generation (RAG) system designed for the Lambda Execution Engine project. Combines Intel NPU/GPU-accelerated embeddings with LM Studio for fast, context-aware code assistance without cloud dependencies.

**NEW: Now supports 100+ file queries with smart chunking and hierarchical search!**

---

## 🚀 Features

- **100+ File Support**: Advanced chunking enables queries across entire codebase
- **Three Search Strategies**: Hierarchical, chunk-level, and file-level modes
- **Semantic Code Search**: Find relevant files by meaning, not just keywords
- **Context-Aware AI**: LLM responses based on actual project code
- **Multi-Device Optimization**: Uses NPU for embeddings, GPU for LLM
- **Auto-Indexing**: Automatically re-indexes files when you save changes
- **Web Interface**: Beautiful UI at http://localhost:8000
- **CLI Tools**: Terminal-friendly workflow for power users
- **100% Local**: No cloud APIs, no data leaves your machine
- **Dual GPU Support**: Optimized for 2x Intel Arc B580 setup

---

## 🎯 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Choose Your System

**Basic System** (10-20 files per query):
```bash
python dual_gpu_config.py  # Auto-configure
python rag_server.py       # Start server
```

**Advanced System** (100+ files per query) - **RECOMMENDED**:
```bash
python upgrade_to_chunked.py       # One-time upgrade
python rag_server_advanced.py      # Start advanced server
```

### 3. Start LM Studio

1. Open **LM Studio**
2. Download and load **Qwen 2.5 Coder 32B** (dual GPU) or **14B** (single GPU)
3. Settings → GPU → Select GPU device
4. Click **Start Server** (port 1234)

### 4. Launch RAG System

**Linux/Mac:**
```bash
chmod +x start_lambda_rag.sh
./start_lambda_rag.sh
```

**Windows:**
```bash
start_lambda_rag.bat
```

**Manual:**
```bash
# Terminal 1: Auto-indexer
python auto_index.py

# Terminal 2: RAG server (basic or advanced)
python rag_server_advanced.py  # Recommended

# Browser: http://localhost:8000
```

---

## 💻 Usage

### Web Interface (Recommended)

1. Open **http://localhost:8000**
2. **Choose search strategy** (Advanced system only):
   - **Hierarchical** (default): 20 files × 5 chunks = 100 results
   - **Chunk-Level**: 100+ precise chunks
   - **File-Level**: Fast overview, 10-20 files
3. Type question: *"How does the gateway pattern work?"*
4. Click **Search & Ask LLM**
5. View relevant files + AI-generated answer

### Search Strategy Guide

**Use Hierarchical for:**
- Broad questions: "How does error handling work?"
- Architecture questions: "Explain the SUGA pattern"
- Cross-cutting concerns: "How is logging used?"
- **Best all-around choice**

**Use Chunk-Level for:**
- Specific code: "Show me the exact routing implementation"
- Debugging: "Where is this function called?"
- Precision over diversity

**Use File-Level for:**
- Quick exploration: "What files handle caching?"
- Understanding structure
- Fast responses needed

### Command Line

```bash
# Ask questions from terminal
python lambda_ask.py "How does the gateway pattern work?"
python lambda_ask.py "What files implement caching?"
python lambda_ask.py "Explain the SUGA architecture"
```

### Clipboard Enhancement

```bash
# Start clipboard watcher
python vscode_context.py

# Copy any question ending with "?"
# It's automatically enhanced with project context
# Paste into LM Studio or any chat interface
```

### Python API

```python
from lambda_indexer_chunked import LambdaProjectIndexer

# Initialize
indexer = LambdaProjectIndexer(device="NPU")
indexer.load_index("lambda_index")

# Hierarchical search (100 results)
results = indexer.search_hierarchical(
    "gateway architecture",
    file_top_k=20,
    chunks_per_file=5
)

# Chunk-level search (precise)
results = indexer.search_chunks("specific function", top_k=100)

# File-level search (fast)
results = indexer.search_files("what handles auth", top_k=20)
```

---

## 📁 Project Structure

### Core System Files

```
lambda-rag-system/
├── lambda_indexer.py           # Basic file-level indexer
├── lambda_indexer_chunked.py   # Advanced chunked indexer ⭐ NEW
├── rag_server.py               # Basic web server
├── rag_server_advanced.py      # Advanced server (100+ files) ⭐ NEW
├── lambda_ask.py               # CLI query tool
├── auto_index.py               # Auto-reindex on file changes
├── vscode_context.py           # Clipboard context enhancer
├── gpu_monitor.py              # Hardware detection
├── dual_gpu_config.py          # Auto-configuration helper
├── upgrade_to_chunked.py       # Migration script ⭐ NEW
├── search_examples.py          # Strategy demonstrations ⭐ NEW
├── start_lambda_rag.sh         # Startup script (Linux/Mac)
├── start_lambda_rag.bat        # Startup script (Windows)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── README_RAG_SETUP.md         # Detailed setup guide
├── ADVANCED_FEATURES.md        # 100+ file support guide ⭐ NEW
└── lambda_index_*.npz          # Generated index files
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│            Lambda RAG System (Advanced)                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │   You Ask    │───▶│  RAG Server  │───▶│ LM Studio│ │
│  │   Question   │    │  Port 8000   │    │ Port 1234│ │
│  └──────────────┘    └──────────────┘    └──────────┘ │
│                             │                    ▲      │
│                             ▼                    │      │
│                  ┌────────────────────┐         │      │
│                  │  Search Strategy:   │         │      │
│                  │  • Hierarchical     │         │      │
│                  │  • Chunk-Level      │         │      │
│                  │  • File-Level       │         │      │
│                  └────────────────────┘         │      │
│                             │                    │      │
│                             ▼                    │      │
│                      ┌──────────────┐           │      │
│                      │   Embedder   │           │      │
│                      │   (NPU/GPU)  │           │      │
│                      │  - Files      │           │      │
│                      │  - Chunks     │           │      │
│                      └──────────────┘           │      │
│                             │                    │      │
│                             ▼                    │      │
│                      ┌──────────────┐           │      │
│                      │ Find Top 100 │───────────┘      │
│                      │   Results    │                  │
│                      └──────────────┘                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Hardware Allocation (Dual Arc B580)

```
GPU 0 (12GB):     LM Studio - Qwen 2.5 32B Instruct
                  Main code generation, 30-40 tok/s
                  Port 1234

GPU 1 (12GB):     Embeddings - BGE-Small-EN-v1.5
                  Background indexing, 500-1000 files/s
                  Both file and chunk embeddings

NPU (13 TOPS):    Alternative for embeddings
                  Zero GPU impact

CPU (20 cores):   RAG server + file watching
                  Coordination layer
```

### Chunking Architecture

```
Original File (gateway.py - 500 lines)
         │
         ▼
   Smart Chunking (code-aware)
         │
         ├─ Chunk 0: Imports + setup
         ├─ Chunk 1: Class definition
         ├─ Chunk 2: execute_operation()  ◄─ 98% match!
         ├─ Chunk 3: _route_to_module()   ◄─ 96% match!
         ├─ Chunk 4: Error handling       ◄─ 92% match!
         └─ Chunk 5-9: Other functions
         
Query: "How does operation routing work?"
Result: Gets chunks 2, 3, 4 (exact relevant code)
Instead of: Entire 500-line file with lots of irrelevant code
```

---

## 📊 Performance Comparison

### Basic vs Advanced System

| Feature | Basic System | Advanced System |
|---------|--------------|-----------------|
| **Max Files/Query** | 10-20 | 100+ |
| **Indexing Level** | File only | File + Chunk |
| **Search Modes** | 1 (file-level) | 3 (hierarchical/chunk/file) |
| **Context Size** | Fixed (~40KB) | Configurable (8-20KB) |
| **Code Awareness** | None | Smart chunking |
| **Precision** | Good | Excellent |
| **Speed** | Fast | Fast-Medium |
| **Best For** | Simple queries | Complex projects |

### Search Strategy Performance

| Strategy | Files | Chunks | Speed | Precision | Coverage | Best For |
|----------|-------|--------|-------|-----------|----------|----------|
| **Hierarchical** | 20 | 100 | Medium | Great | Excellent | Default choice |
| **Chunk-Level** | Varies | 100-200 | Slower | Excellent | Great | Specific code |
| **File-Level** | 20 | 60 | Fast | Good | Good | Quick exploration |

### Real-World Timings (Dual B580)

| Operation | Basic System | Advanced System |
|-----------|--------------|-----------------|
| File search | 100-200ms | 100-200ms |
| Hierarchical search | N/A | 300-500ms |
| Chunk search (100) | N/A | 500-800ms |
| Indexing (full project) | 30-60s | 2-5min (one-time) |
| LLM query (32B model) | 1-3s | 1-3s |
| **Total response** | **1.5-3.5s** | **2-4s** |

---

## ⚙️ Configuration

### Device Selection

Edit `lambda_indexer_chunked.py` and `rag_server_advanced.py`:

```python
# Options for dual GPU setup:
indexer = LambdaProjectIndexer(device="GPU.1")   # Second Arc B580 ⭐
indexer = LambdaProjectIndexer(device="GPU.0")   # First Arc B580
indexer = LambdaProjectIndexer(device="NPU")     # Intel AI Boost NPU
indexer = LambdaProjectIndexer(device="CPU")     # CPU fallback
```

**Or use auto-configuration:**
```bash
python dual_gpu_config.py
```

### LM Studio Models

**Dual Arc B580 (24GB total):**
- **Qwen 2.5 72B** (Q4) - Maximum quality, 15-25 tok/s
- **Qwen 2.5 32B** (FP16) - Best balance, 30-40 tok/s ✅ Recommended
- **DeepSeek Coder V2 236B** (Q2) - Ultimate coding, 5-10 tok/s

**Single Arc B580 (12GB):**
- **Qwen 2.5 Coder 14B** (Q5) - Excellent quality, 25-35 tok/s ✅ Recommended
- **Qwen 2.5 14B Instruct** (Q5) - General purpose, 25-35 tok/s

### Search Strategy Tuning

**In Web UI:**
- Select from dropdown: Hierarchical / Chunk-Level / File-Level
- Set number of results: 10-500 (default: 100)
- Set max context: 2000-20000 chars (default: 8000)

**In Code:**
```python
# Hierarchical (balanced)
results = indexer.search_hierarchical(
    query,
    file_top_k=20,        # Top 20 files
    chunks_per_file=5     # 5 chunks each = 100 total
)

# Chunk-level (precise)
results = indexer.search_chunks(
    query,
    top_k=150             # Get 150 best chunks
)

# File-level (fast)
results = indexer.search_files(
    query,
    top_k=15              # 15 files
)
```

### Context Length by Model

```python
# Small models (7-14B):
max_context_length = 6000  # Leaves room for response

# Medium models (32B):
max_context_length = 8000  # Default, good balance ⭐

# Large models (70B+):
max_context_length = 15000  # Can handle more context

# Models with huge context (128K+):
max_context_length = 20000  # Maximum precision
```

---

## 🛠️ Advanced Usage

### Migration to Chunked System

```bash
# Automatic upgrade (recommended)
python upgrade_to_chunked.py

# Manual upgrade
python -c "from lambda_indexer_chunked import LambdaProjectIndexer; i = LambdaProjectIndexer(device='NPU'); i.index_project('./'); i.save_index('lambda_index')"
```

### Running Search Examples

```bash
# Interactive demonstrations of all search strategies
python search_examples.py
```

### Custom Search Pipelines

```python
from lambda_indexer_chunked import LambdaProjectIndexer

indexer = LambdaProjectIndexer(device="NPU")
indexer.load_index("lambda_index")

# Step 1: Quick file overview
files = indexer.search_files("authentication", top_k=10)
print(f"Found {len(files)} relevant files")

# Step 2: Deep dive into top file
top_file = files[0]['file']
all_chunks = indexer.search_chunks("authentication", top_k=200)
file_chunks = [c for c in all_chunks if c['file'] == top_file]

# Step 3: Get exact implementation
for chunk in file_chunks[:5]:
    print(f"Chunk {chunk['chunk_num']}: {chunk['score']:.1%}")
    print(chunk['content'][:200])
```

### Quality Filtering

```python
results = indexer.search_chunks(query, top_k=200)

# Filter by relevance threshold
high_quality = [r for r in results if r['score'] > 0.85]  # Very relevant
good_quality = [r for r in results if r['score'] > 0.75]  # Pretty relevant
all_results = results  # Everything

# Use high_quality for small models, all_results for large models
```

### API Endpoints (Advanced Server)

**GET /api/stats**
```bash
curl http://localhost:8000/api/stats
# Returns: {"indexed_files": 42, "indexed_chunks": 523, "status": "ready"}
```

**POST /api/ask** (with search strategy)
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does caching work?",
    "search_mode": "hierarchical",
    "num_results": 100,
    "max_context_length": 8000
  }'
```

**POST /api/quick_search** (no LLM, just file list)
```bash
curl -X POST http://localhost:8000/api/quick_search \
  -H "Content-Type: application/json" \
  -d '{"question": "authentication files", "top_k": 20}'
```

---

## 🐛 Troubleshooting

### Cannot connect to LM Studio

**Check:**
1. LM Studio is running
2. Model is loaded
3. Server is started (click "Start Server")
4. Port is 1234 (or update `rag_server_advanced.py`)

**Test connection:**
```bash
curl http://localhost:1234/v1/models
```

### Module not found errors

**Solution:**
```bash
# Make sure all .py files are in same directory
ls *.py

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

### Chunked index not found

**If using advanced system:**
```bash
# Check for chunked index files
ls lambda_index_*

# Should see:
# lambda_index_file_embeddings.npz
# lambda_index_chunk_embeddings.npz
# lambda_index_file_metadata.json
# lambda_index_chunk_metadata.json

# If missing, rebuild:
python upgrade_to_chunked.py
```

### Search results not relevant

**Try:**
1. Switch search strategy (hierarchical → chunk-level)
2. Increase results (100 → 150)
3. Rephrase question more specifically
4. Check if files are indexed: `python gpu_monitor.py --once`
5. Rebuild index: `python upgrade_to_chunked.py`

### Slow performance

**Optimize:**
1. Use GPU.1 for embeddings (keep GPU.0 for LLM)
2. Switch to file-level search for exploration
3. Reduce number of results
4. Use smaller model in LM Studio

### Out of memory

**For dual B580:**
- Move embeddings to GPU.1: `device="GPU.1"`
- Keep GPU.0 free for LLM
- Reduce context length

**For single GPU:**
- Use NPU or CPU for embeddings: `device="NPU"`
- Reduce LLM context length
- Use smaller model (14B instead of 32B)

---

## 📚 Documentation

- **README.md** (this file) - Main documentation
- **README_RAG_SETUP.md** - Detailed setup guide
- **ADVANCED_FEATURES.md** - 100+ file support, search strategies
- **Local LLM Setup Guide** (artifacts) - Intel Arc B580 configuration
- **Dual GPU Configuration** (artifacts) - Multi-device optimization

---

## 🔐 Privacy & Security

- **100% Local**: No data sent to cloud services
- **No Telemetry**: No usage tracking or analytics
- **Offline Capable**: Works without internet (after initial model download)
- **Open Source**: All code is visible and auditable
- **Your Hardware**: Models run on your GPUs, your control

---

## 🚧 Roadmap

**Current Features:**
- ✅ File and chunk-level indexing
- ✅ Three search strategies
- ✅ Smart code-aware chunking
- ✅ 100+ file support
- ✅ Auto-indexing on file changes
- ✅ Web UI with strategy selection
- ✅ Dual GPU optimization

**Planned:**
- [ ] Multi-project support (switch between projects)
- [ ] Chat history with context retention
- [ ] Code diff visualization in web UI
- [ ] Integration with Git (search by commit)
- [ ] Fine-tuned embedding model for Lambda project
- [ ] Batch query support
- [ ] Export conversations to Markdown
- [ ] VSCode extension
- [ ] Cross-file reference tracking
- [ ] Automatic query expansion

---

## 📝 License

```
Copyright 2025 Joseph Hersey

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 🙏 Acknowledgments

- **Qwen Team** - Excellent open-source models
- **Intel** - OpenVINO toolkit and Arc GPU support
- **BAAI** - BGE embedding models
- **LM Studio** - Great local LLM interface
- **Lambda Execution Engine** - The project this system serves

---

## 📞 Support

**Having issues?**
1. Check [Troubleshooting](#-troubleshooting) section
2. Run `python dual_gpu_config.py` to verify setup
3. Check logs: `auto_index.log` and `rag_server.log`
4. Monitor GPUs: `python gpu_monitor.py`
5. Read **ADVANCED_FEATURES.md** for search strategy help

**For Lambda Project specific questions:**
- Use the RAG system itself! It's designed to answer questions about the project.

---

## 🎓 Quick Reference

### Which System Should I Use?

**Use Basic System if:**
- Simple project (< 50 files)
- Questions about 1-2 files
- Want fastest possible responses
- Limited computational resources

**Use Advanced System if:**
- Large project (100+ files) ✅
- Complex, multi-file questions
- Want maximum precision
- Have dual GPUs or NPU available
- **Recommended for Lambda Project**

### Which Search Strategy?

**Hierarchical** (Default):
- "How does X work across the project?"
- "Explain the architecture of Y"
- Best all-around choice

**Chunk-Level**:
- "Show me the exact code for X"
- "Where is function Y implemented?"
- Need maximum precision

**File-Level**:
- "What files handle X?"
- "Give me an overview of Y"
- Need fast response

---

**Built with ❤️ for the Lambda Execution Engine Project**

*Empowering local AI development on Intel hardware with intelligent code assistance for 100+ files*

---

## 🚀 Get Started Now

```bash
# 1. Install
pip install -r requirements.txt

# 2. Upgrade to advanced system (recommended)
python upgrade_to_chunked.py

# 3. Start
python rag_server_advanced.py

# 4. Ask questions!
# Open http://localhost:8000
```
