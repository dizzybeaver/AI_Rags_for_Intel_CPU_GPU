# Advanced Features - 100+ File Support

## Overview

The advanced Lambda RAG system supports retrieving and using context from **100+ files** in a single query through smart chunking and hierarchical search strategies.

---

## Why Chunking?

### Problem with File-Level Indexing

**Basic System:**
- Each file = 1 embedding
- Returns top 10-20 files
- Large files may contain multiple topics
- Relevant section might be buried in large file
- Limited to ~2000 chars per file = 20-40KB total context

**Example Issue:**
```
gateway.py (500 lines)
  ├─ Lines 1-100: Imports and setup
  ├─ Lines 101-200: Core gateway logic ✓ RELEVANT
  ├─ Lines 201-300: Helper functions
  └─ Lines 301-500: Error handling

Question: "How does the gateway route operations?"
Result: Gets entire file, wastes context on irrelevant sections
```

### Solution: Chunk-Level Indexing

**Advanced System:**
- Each file → Multiple chunks (smart splits)
- Each chunk = 1 embedding
- Returns top 100+ chunks
- Gets EXACTLY the relevant code sections
- Configurable context limit (8-20KB)

**Same Example:**
```
gateway.py → 10 chunks
  ├─ Chunk 0: Imports and setup
  ├─ Chunk 1-3: Core gateway logic ✓✓✓ TOP MATCHES
  ├─ Chunk 4-6: Helper functions
  └─ Chunk 7-9: Error handling ✓ SOMEWHAT RELEVANT

Question: "How does the gateway route operations?"
Result: Gets chunks 1-3 with 95%+ relevance, plus chunk 8 for error context
```

---

## Search Strategies

### 1. Hierarchical Search (DEFAULT - Best for Most Cases)

**How it works:**
1. **Stage 1 (File-level)**: Find top 20 most relevant files
2. **Stage 2 (Chunk-level)**: Find top 5 chunks from each file
3. **Result**: 100 total chunks (20 files × 5 chunks) focused on relevant code

**When to use:**
- General questions about project
- Want broad coverage with depth
- Looking for patterns across multiple files
- Default choice - works great for most queries

**Configuration:**
```python
results = indexer.search_hierarchical(
    query="how does caching work",
    file_top_k=20,      # Top 20 files
    chunks_per_file=5   # 5 chunks per file = 100 total
)
```

**Example:**
```
Question: "How is error handling implemented?"

Stage 1 (Files):
  1. error_handler.py (92% match)
  2. gateway.py (87% match)
  3. logging.py (81% match)
  ... 17 more files

Stage 2 (Chunks within those files):
  error_handler.py:
    ✓ Chunk 2: Exception classes (95%)
    ✓ Chunk 4: Error recovery (93%)
    ✓ Chunk 7: Logging integration (88%)
    ✓ Chunk 9: Retry logic (85%)
    ✓ Chunk 1: Imports (72%)
  
  gateway.py:
    ✓ Chunk 12: Error propagation (91%)
    ... etc

Total: 100 focused chunks across 20 files
```

**Pros:**
- ✅ Ensures diversity (doesn't get stuck in one file)
- ✅ Covers multiple perspectives
- ✅ Fast (two-stage filtering)
- ✅ Good balance of breadth and depth

**Cons:**
- ⚠️ Might miss some very relevant chunks if file-level match is low
- ⚠️ Fixed limit per file (can't get 10 chunks from most relevant file)

---

### 2. Chunk-Level Search (Precise)

**How it works:**
1. Calculate similarity for EVERY chunk in entire codebase
2. Return top N chunks regardless of which file they're from
3. Could return 20 chunks from same file if it's highly relevant

**When to use:**
- Very specific technical questions
- Looking for exact code patterns
- Want maximum precision over diversity
- Need to find needle in haystack

**Configuration:**
```python
results = indexer.search_chunks(
    query="specific function implementation",
    top_k=100   # Return top 100 chunks total
)
```

**Example:**
```
Question: "How is the operation dispatch mechanism implemented?"

Results (top 10 of 100):
  1. gateway.py, chunk 15: execute_operation() (98% match)
  2. gateway.py, chunk 16: _route_to_module() (97% match)
  3. gateway.py, chunk 14: operation validation (96% match)
  4. gateway.py, chunk 17: error handling (95% match)
  5. interfaces/cache.py, chunk 8: dispatch example (94% match)
  6. gateway.py, chunk 13: initialization (93% match)
  7. tests/test_gateway.py, chunk 22: test cases (92% match)
  ... 93 more chunks

Notice: 4 of top 7 are from gateway.py because that's where the answer is!
```

**Pros:**
- ✅ Maximum precision
- ✅ Finds exact relevant code
- ✅ Not limited by file boundaries
- ✅ Great for specific questions

**Cons:**
- ⚠️ May over-focus on one file
- ⚠️ Can miss broader context
- ⚠️ Slower (checks all chunks)

---

### 3. File-Level Search (Fast Overview)

**How it works:**
1. Each file has ONE embedding (average of all chunks)
2. Find top N most relevant files
3. Return some chunks from each file

**When to use:**
- Quick exploration
- Want file list without deep dive
- Trying to understand project structure
- Need fast response over precision

**Configuration:**
```python
results = indexer.search_files(
    query="what handles authentication",
    top_k=20    # Top 20 files
)
```

**Example:**
```
Question: "What files are involved in user authentication?"

Results:
  1. auth/authenticator.py (94% match) - 12 chunks
  2. auth/validators.py (89% match) - 8 chunks
  3. middleware/auth_middleware.py (85% match) - 6 chunks
  4. models/user.py (81% match) - 15 chunks
  5. config/auth_config.py (78% match) - 4 chunks
  ... 15 more files

Speed: ~100ms (vs 500ms for chunk-level)
```

**Pros:**
- ✅ Very fast
- ✅ Good overview of relevant files
- ✅ Shows structure of codebase
- ✅ Great first step for exploration

**Cons:**
- ⚠️ Less precise than chunk-level
- ⚠️ May include files with only small relevant section
- ⚠️ Limited depth per file

---

## Smart Chunking Strategies

The system uses code-aware chunking that preserves structure:

### Python Files (.py)

**Chunks by functions/classes:**
```python
# Original file
import os
from typing import List

class GatewayManager:
    def __init__(self):
        self.modules = {}
    
    def register_module(self, name, module):
        self.modules[name] = module
    
    def execute(self, operation):
        # Implementation
        pass

def helper_function():
    return "helper"

# Chunked into:
# Chunk 0: imports + class definition start
# Chunk 1: __init__ method
# Chunk 2: register_module method
# Chunk 3: execute method
# Chunk 4: helper_function
```

### Markdown Files (.md)

**Chunks by headers:**
```markdown
# Main Title

## Section 1
Content for section 1...

## Section 2
Content for section 2...

### Subsection 2.1
More content...

# Chunked into:
# Chunk 0: Main Title + Section 1
# Chunk 1: Section 2 (without subsections)
# Chunk 2: Subsection 2.1
```

### Other Files (.json, .txt, etc.)

**Word-based with overlap:**
```
Chunk size: 512 words
Overlap: 50 words

Original: 1500 words
Chunk 0: words 0-512
Chunk 1: words 462-974 (50 word overlap with chunk 0)
Chunk 2: words 924-1436
Chunk 3: words 1386-1500
```

**Why overlap?**
- Prevents breaking context at chunk boundaries
- Ensures no information loss between chunks
- Helps with questions that span chunk boundaries

---

## Context Management

### The LLM Context Window Problem

Most LLMs have context limits:
- Qwen 2.5 32B: 32K tokens (~100KB text)
- Qwen 2.5 14B: 32K tokens
- Qwen 2.5 72B: 128K tokens

But including everything wastes context:
- 100 files × 2000 chars = 200KB
- Exceeds context limit!

### Smart Summarization

The advanced system uses hierarchical summarization:

```python
def summarize_chunks(chunks, max_length=8000):
    """
    Given 100 chunks, create 8KB summary:
    1. Group chunks by file
    2. Take most relevant chunks first
    3. Truncate individual chunks if needed
    4. Stop at max_length limit
    """
```

**Example:**
```
Input: 100 chunks, ~150KB total
Output: Smart summary, 8KB

=== gateway.py ===
[Chunk 15] def execute_operation(interface, operation, **params):
    """Route operation through gateway to appropriate module"""
    module = self._get_module(interface)
    return module.execute(operation, **params)

[Chunk 16] def _get_module(self, interface):
    if interface not in self._modules:
        self._lazy_load_module(interface)
    return self._modules[interface]

=== cache.py ===
[Chunk 8] def get_cached(key):
    """Retrieve from cache via gateway"""
    return execute_operation(GatewayInterface.CACHE, 'get', key=key)

... (additional relevant chunks up to 8KB limit)
... (additional content truncated due to size)
```

---

## Configuration Guide

### Choosing Search Strategy

```python
# For broad questions:
search_mode = "hierarchical"
num_results = 100
# "How does the system handle errors?"
# "What's the architecture of the gateway?"

# For specific questions:
search_mode = "chunk"
num_results = 150
# "Where is the operation dispatch implemented?"
# "Show me the exact cache invalidation logic"

# For exploration:
search_mode = "file"
num_results = 20
# "What files handle authentication?"
# "Where is logging configured?"
```

### Context Length Tuning

```python
# Small models (7-14B):
max_context_length = 6000  # ~6KB, leaves room for response

# Medium models (32B):
max_context_length = 8000  # Default, good balance

# Large models (70B+):
max_context_length = 15000  # Can handle more context

# Models with large context windows (128K+):
max_context_length = 20000  # Maximum precision
```

### Web UI Settings

**Search Strategy Dropdown:**
- Hierarchical (20 files, 5 chunks each)
- Chunk-Level (100+ precise chunks)
- File-Level (Fast overview, 10-20 files)

**Number of Results:**
- 10-50: Fast, focused
- 100: Default, good coverage
- 200-500: Maximum recall (slower)

**Max Context Chars:**
- 2000-4000: Small models
- 6000-8000: Medium models (default)
- 10000-15000: Large models
- 15000-20000: Models with huge context

---

## Performance Comparison

| Strategy | Files | Chunks | Context | Speed | Precision | Recall |
|----------|-------|--------|---------|-------|-----------|--------|
| **File-Level** | 20 | 60 | ~6KB | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | ⭐⭐⭐ Good |
| **Hierarchical** | 20 | 100 | ~8KB | ⚡⚡ Medium | ⭐⭐⭐⭐ Great | ⭐⭐⭐⭐⭐ Excellent |
| **Chunk-Level** | Varies | 100-200 | ~8-15KB | ⚡ Slower | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Great |

**Speed benchmarks (NPU/GPU):**
- File search: ~100-200ms
- Hierarchical: ~300-500ms
- Chunk search (100): ~500-800ms
- Chunk search (500): ~2-3s

---

## Real-World Examples

### Example 1: Architecture Question

**Question:** "Explain the SUGA architecture pattern"

**Best Strategy:** Hierarchical
```
Why: SUGA is implemented across multiple files
Result: Gets relevant chunks from:
  - UNIFIED_PROJECT_INSTRUCTIONS.MD (architecture docs)
  - gateway.py (implementation)
  - All interface files (usage examples)
  - Test files (behavior examples)

20 files × 5 chunks = 100 chunks covering entire pattern
```

### Example 2: Specific Implementation

**Question:** "Show me the exact code for lazy module loading in gateway"

**Best Strategy:** Chunk-level
```
Why: Looking for specific function implementation
Result: Top 10 chunks all from gateway.py:
  1. _lazy_load_module() definition (98%)
  2. Module import logic (96%)
  3. Error handling (94%)
  4. Related helper functions (92%)
  etc.

Precision > diversity
```

### Example 3: File Discovery

**Question:** "What files implement caching?"

**Best Strategy:** File-level
```
Why: Want overview, not implementation details
Result: List of 15 files:
  - cache.py
  - gateway.py (uses caching)
  - test_cache.py
  - config files
  - Documentation

Fast response, clear structure
```

### Example 4: Cross-Cutting Concern

**Question:** "How is logging used throughout the project?"

**Best Strategy:** Hierarchical
```
Why: Logging appears in many files
Result: 100 chunks from 20 files showing:
  - logging.py implementation
  - gateway.py logging calls
  - Error handler logging
  - Each interface's logging
  - Test logging

Coverage across entire codebase
```

---

## Migration from Basic System

To upgrade to 100+ file support:

```bash
# 1. Run upgrade script
python upgrade_to_chunked.py

# 2. Verify new index
python -c "from lambda_indexer_chunked import LambdaProjectIndexer; i = LambdaProjectIndexer(); i.load_index('lambda_index'); print(f'Files: {len(i.file_embeddings)}, Chunks: {len(i.chunk_embeddings)}')"

# 3. Use advanced server
python rag_server_advanced.py
```

**What changes:**
- Old index → Backed up automatically
- New chunked index → Built from scratch
- Startup scripts → Updated to use advanced server
- Web UI → New search strategy options

**Backward compatible:**
- Can still do file-level search
- Same API endpoints
- All old scripts still work

---

## Tips & Best Practices

### ✅ DO:

1. **Use hierarchical as default** - Best all-around performance
2. **Increase results for broad questions** - 150-200 for "how does X work"
3. **Use chunk-level for debugging** - Finding exact code sections
4. **Tune context length to your model** - Larger models = more context
5. **Check file overview first** - See what files are relevant before LLM query

### ❌ DON'T:

1. **Don't use 500 chunks with small models** - Wastes context window
2. **Don't always use chunk-level** - Hierarchical is faster and often better
3. **Don't set context too high** - Leaves less room for LLM response
4. **Don't ignore search strategy choice** - Different questions need different approaches

---

## Troubleshooting

### "Getting irrelevant results"

**Try:**
1. Switch from file-level to chunk-level search
2. Increase number of results (100 → 150)
3. Rephrase question to be more specific
4. Use hierarchical to get better coverage

### "LLM response is cut off"

**Try:**
1. Reduce context length (8000 → 6000)
2. Reduce number of results (100 → 50)
3. Increase max_tokens in LLM settings
4. Use model with larger context window

### "Search is too slow"

**Try:**
1. Switch to file-level search for exploration
2. Reduce number of results
3. Use GPU instead of CPU for embeddings
4. Consider smaller embedding model

### "Missing important files"

**Try:**
1. Use hierarchical instead of chunk-level
2. Increase file_top_k (20 → 30)
3. Check if files are indexed: `python -c "...print files..."`
4. Rebuild index if files were recently added

---

## Future Enhancements

Planned improvements:
- [ ] Adaptive chunking based on code complexity
- [ ] Cross-chunk context linking
- [ ] Query expansion for better recall
- [ ] Re-ranking with cross-encoder
- [ ] Caching of frequent queries
- [ ] Multi-query decomposition for complex questions

---

**The advanced system gives you the power to ask questions that span 100+ files and get precise, relevant answers. Choose the right strategy for your question type and enjoy context-aware AI assistance!**
