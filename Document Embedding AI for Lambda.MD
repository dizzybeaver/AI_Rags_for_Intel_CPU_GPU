# Document Embedding AI for Lambda Project Files

## Overview

Document embedders convert your code, documentation, and text files into vector representations for semantic search, RAG (Retrieval-Augmented Generation), and similarity matching.

**Use Cases for Lambda Project**:
- Find similar code patterns across files
- Retrieve relevant context for LLM queries
- Identify architecture violations automatically
- Search documentation by meaning, not keywords
- Build knowledge base for the 5-step ritual searches

---

## Top Recommended Embedders for Code + Documentation

### 🥇 Tier 1: Best for Lambda Project

#### 1. **BGE-Small-EN-v1.5** (RECOMMENDED for NPU)
```
Model: BAAI/bge-small-en-v1.5
Size: 133M parameters
Dimensions: 384
Best For: General code + docs, NPU-optimized
```

**Why Perfect for Lambda**:
- ✅ Excellent code understanding
- ✅ Fast inference on NPU (optimized for INT8)
- ✅ Small enough for continuous indexing
- ✅ Great retrieval quality
- ✅ Works with Python, JavaScript, Markdown

**Setup**:
```python
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer

model = OVModelForFeatureExtraction.from_pretrained(
    "BAAI/bge-small-en-v1.5",
    export=True,
    device="NPU"  # Or "CPU" if NPU unavailable
)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", 
                      padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()
```

**Performance**:
- NPU: ~500-1000 embeddings/sec
- CPU: ~100-200 embeddings/sec
- VRAM: <500MB

---

#### 2. **Nomic Embed Text v1.5**
```
Model: nomic-ai/nomic-embed-text-v1.5
Size: 137M parameters
Dimensions: 768
Best For: Long context code files
```

**Why Great for Lambda**:
- ✅ 8192 token context (handles large files)
- ✅ Excellent code + documentation mixing
- ✅ Trained on diverse programming languages
- ✅ Open source and actively maintained
- ✅ Matryoshka embeddings (flexible dimensions)

**Setup**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    device="cpu"  # or "cuda" for GPU
)

# Prefix for optimal performance
def embed_code(code, is_query=False):
    prefix = "search_query: " if is_query else "search_document: "
    embedding = model.encode(prefix + code)
    return embedding
```

**Performance**:
- CPU: ~50-100 embeddings/sec
- GPU: ~300-500 embeddings/sec
- Context: Up to 8192 tokens

---

#### 3. **E5-Small-v2**
```
Model: intfloat/e5-small-v2
Size: 109M parameters  
Dimensions: 384
Best For: Balanced speed + quality
```

**Why Good for Lambda**:
- ✅ Very fast inference
- ✅ Good code understanding
- ✅ Low resource usage
- ✅ Strong retrieval performance

**Setup**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/e5-small-v2')

def embed_text(text, is_query=False):
    prefix = "query: " if is_query else "passage: "
    return model.encode(prefix + text)
```

---

### 🥈 Tier 2: Specialized Code Embedders

#### 4. **CodeBERT**
```
Model: microsoft/codebert-base
Size: 125M parameters
Best For: Pure code similarity
```

**Strengths**:
- Pretrained specifically on code
- Understands programming language syntax
- Good for cross-language code search

**Limitations**:
- Not as good with natural language docs
- Older architecture (2020)

---

#### 5. **StarEncoder**
```
Model: bigcode/starencoder
Size: 110M parameters
Best For: Multi-language code
```

**Strengths**:
- Trained on The Stack (massive code dataset)
- Excellent for code-to-code search
- Supports 80+ programming languages

**Limitations**:
- Focused purely on code, less good for markdown/docs

---

### 🥉 Tier 3: Lightweight Options

#### 6. **All-MiniLM-L6-v2**
```
Model: sentence-transformers/all-MiniLM-L6-v2
Size: 22M parameters
Dimensions: 384
Best For: Speed-critical applications
```

**When to Use**:
- Need extremely fast embeddings
- Limited compute resources
- Real-time search requirements

**Trade-off**: Lower quality than BGE/Nomic

---

## Complete Lambda Project Indexing System

### Full Implementation with BGE-Small on NPU

```python
# lambda_indexer.py - Run this on NPU for continuous indexing

import os
from pathlib import Path
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np
import json
from typing import List, Dict

class LambdaProjectIndexer:
    def __init__(self, device="NPU"):
        """Initialize embedder on NPU for background indexing"""
        self.model = OVModelForFeatureExtraction.from_pretrained(
            "BAAI/bge-small-en-v1.5",
            export=True,
            device=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-small-en-v1.5"
        )
        self.embeddings = {}
        self.metadata = {}
        
    def embed_file(self, filepath: str) -> np.ndarray:
        """Embed a single file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Chunk large files
        chunks = self._chunk_text(content, max_length=512)
        chunk_embeddings = []
        
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            chunk_embeddings.append(embedding)
        
        # Average embeddings for full file
        return np.mean(chunk_embeddings, axis=0)
    
    def _chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk).split()) >= max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def index_project(self, project_path: str):
        """Index entire Lambda project"""
        extensions = {'.py', '.md', '.json', '.yaml', '.yml', '.txt'}
        
        for root, dirs, files in os.walk(project_path):
            # Skip virtual environments and caches
            dirs[:] = [d for d in dirs if d not in {
                'venv', '__pycache__', '.git', 'node_modules'
            }]
            
            for file in files:
                if Path(file).suffix in extensions:
                    filepath = os.path.join(root, file)
                    
                    try:
                        embedding = self.embed_file(filepath)
                        rel_path = os.path.relpath(filepath, project_path)
                        
                        self.embeddings[rel_path] = embedding
                        self.metadata[rel_path] = {
                            'absolute_path': filepath,
                            'extension': Path(file).suffix,
                            'size': os.path.getsize(filepath)
                        }
                        
                        print(f"✓ Indexed: {rel_path}")
                    except Exception as e:
                        print(f"✗ Failed: {filepath} - {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar files"""
        query_embedding = self._embed_query(query)
        
        # Calculate similarities
        similarities = {}
        for filepath, embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, embedding.T) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities[filepath] = float(similarity)
        
        # Get top-k results
        top_files = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        results = []
        for filepath, score in top_files:
            results.append({
                'file': filepath,
                'score': score,
                'metadata': self.metadata[filepath]
            })
        
        return results
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a search query"""
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    def save_index(self, output_path: str):
        """Save embeddings and metadata"""
        np.savez(
            f"{output_path}_embeddings.npz",
            **{k: v for k, v in self.embeddings.items()}
        )
        with open(f"{output_path}_metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load_index(self, input_path: str):
        """Load pre-computed embeddings"""
        data = np.load(f"{input_path}_embeddings.npz")
        self.embeddings = {k: data[k] for k in data.files}
        
        with open(f"{input_path}_metadata.json", 'r') as f:
            self.metadata = json.load(f)


# Usage Example
if __name__ == "__main__":
    indexer = LambdaProjectIndexer(device="NPU")
    
    # Index the entire project
    indexer.index_project("/path/to/lambda-execution-engine")
    
    # Save for later use
    indexer.save_index("lambda_index")
    
    # Search for relevant files
    results = indexer.search(
        "gateway architecture and interface patterns",
        top_k=5
    )
    
    print("\nSearch Results:")
    for result in results:
        print(f"{result['score']:.3f} - {result['file']}")
```

---

## Integration with LM Studio Workflow

### Step 1: Background Indexing on NPU
```bash
# Run once to build index (or on file changes)
python lambda_indexer.py
```

### Step 2: Query Enhancement for LLM
```python
# Before sending to LLM, enrich with relevant context

from lambda_indexer import LambdaProjectIndexer

indexer = LambdaProjectIndexer(device="NPU")
indexer.load_index("lambda_index")

# User asks a question
user_question = "How does the gateway pattern work?"

# Find relevant files
relevant_files = indexer.search(user_question, top_k=3)

# Build context for LLM
context = ""
for result in relevant_files:
    with open(result['metadata']['absolute_path'], 'r') as f:
        context += f"\n\n=== {result['file']} ===\n{f.read()}\n"

# Send to LLM with context
prompt = f"""Context from project files:
{context}

User question: {user_question}

Answer based on the project context above."""

# Send to LM Studio API
# ... (your LLM query code)
```

---

## Performance Comparison for Lambda Project

| Embedder | Device | Speed (files/sec) | Quality | Memory |
|----------|--------|-------------------|---------|---------|
| **BGE-Small** | NPU | 500-1000 | ⭐⭐⭐⭐⭐ | 400MB |
| **BGE-Small** | CPU | 100-200 | ⭐⭐⭐⭐⭐ | 400MB |
| **Nomic Embed** | CPU | 50-100 | ⭐⭐⭐⭐⭐ | 600MB |
| **E5-Small** | CPU | 150-250 | ⭐⭐⭐⭐ | 350MB |
| **All-MiniLM** | CPU | 300-500 | ⭐⭐⭐ | 90MB |

---

## My Recommendation for Lambda Project

### Primary Setup:
```
Embedder: BGE-Small-EN-v1.5
Device: NPU (background indexing)
Update: On file save/commit
Storage: Local .npz + metadata JSON
```

### Why This Works:
1. **NPU handles continuous indexing** without blocking GPU/CPU
2. **Small enough** to re-index frequently (on every save)
3. **High quality** retrieval for code + documentation
4. **Fast search** (thousands of queries/sec)
5. **Integrates with your ritual** - Step 2 (SEARCH) can use this

### Workflow:
```
1. Write/modify code
2. Auto-index on save (NPU background task)
3. When asking LLM a question → search index first
4. Send top-3 relevant files as context to LLM
5. LLM answers with project-specific knowledge
```

This gives you **local RAG** without external APIs!

---

## Quick Start Commands

```bash
# Install dependencies
pip install optimum-intel sentence-transformers openvino-dev

# Download BGE-Small
python -c "from optimum.intel import OVModelForFeatureExtraction; OVModelForFeatureExtraction.from_pretrained('BAAI/bge-small-en-v1.5', export=True)"

# Index your project
python lambda_indexer.py

# Test search
python -c "
from lambda_indexer import LambdaProjectIndexer
indexer = LambdaProjectIndexer()
indexer.load_index('lambda_index')
results = indexer.search('gateway pattern', top_k=5)
for r in results: print(f\"{r['score']:.3f} - {r['file']}\")
"
```

---

## Advanced: Real-Time File Watching

```python
# auto_index.py - Watch for file changes and re-index

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from lambda_indexer import LambdaProjectIndexer
import time

class ProjectFileHandler(FileSystemEventHandler):
    def __init__(self, indexer):
        self.indexer = indexer
    
    def on_modified(self, event):
        if event.src_path.endswith(('.py', '.md', '.json')):
            print(f"Re-indexing: {event.src_path}")
            self.indexer.index_project("/path/to/project")
            self.indexer.save_index("lambda_index")

indexer = LambdaProjectIndexer(device="NPU")
handler = ProjectFileHandler(indexer)

observer = Observer()
observer.schedule(handler, "/path/to/lambda-execution-engine", recursive=True)
observer.start()

print("👀 Watching for file changes...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
```

Now your project is **always indexed** and **always searchable**! 🚀
