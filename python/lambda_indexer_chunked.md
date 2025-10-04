# Copyright 2025 Joseph Hersey
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lambda Project Advanced RAG Server
Handles 100+ file results with smart summarization and context management
"""

from flask import Flask, request, jsonify, render_template_string
from lambda_indexer_chunked import LambdaProjectIndexer
import requests
import os

app = Flask(__name__)

# Initialize indexer with chunking support
print("Loading chunked embeddings...")
indexer = LambdaProjectIndexer(device="NPU")

# Load pre-built index or create new one
if os.path.exists("lambda_index_file_embeddings.npz"):
    indexer.load_index("lambda_index")
    print("✓ Loaded existing index")
else:
    print("Building new chunked index...")
    indexer.index_project("./")
    indexer.save_index("lambda_index")
    print("✓ Index created")

# LM Studio API endpoints
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

def summarize_chunks(chunks: list, max_length: int = 8000) -> str:
    """
    Summarize multiple chunks into concise context
    Groups by file and creates hierarchical summary
    """
    # Group chunks by file
    files = {}
    for chunk in chunks:
        filepath = chunk['file']
        if filepath not in files:
            files[filepath] = []
        files[filepath].append(chunk)
    
    summary = ""
    current_length = 0
    
    for filepath, file_chunks in files.items():
        # Sort chunks by chunk number for coherent reading
        file_chunks.sort(key=lambda x: x['chunk_num'])
        
        file_summary = f"\n=== {filepath} ===\n"
        
        # Add relevant chunks from this file
        for chunk in file_chunks:
            chunk_text = chunk['content'][:500]  # Limit chunk size
            chunk_summary = f"[Chunk {chunk['chunk_num']}] {chunk_text}\n"
            
            # Check if adding this would exceed limit
            if current_length + len(chunk_summary) > max_length:
                summary += "\n... (additional content truncated due to size)\n"
                break
            
            file_summary += chunk_summary
            current_length += len(chunk_summary)
        
        summary += file_summary
        
        if current_length >= max_length:
            break
    
    return summary

def create_file_overview(chunks: list) -> str:
    """Create overview of which files are relevant"""
    files = {}
    for chunk in chunks:
        filepath = chunk['file']
        if filepath not in files:
            files[filepath] = {
                'score': chunk['score'],
                'chunks': 0
            }
        files[filepath]['chunks'] += 1
        files[filepath]['score'] = max(files[filepath]['score'], chunk['score'])
    
    # Sort by score
    sorted_files = sorted(files.items(), key=lambda x: x[1]['score'], reverse=True)
    
    overview = "Relevant files (ordered by relevance):\n"
    for i, (filepath, data) in enumerate(sorted_files[:20], 1):
        overview += f"{i}. {filepath} ({data['chunks']} relevant chunks, {data['score']:.2%} match)\n"
    
    return overview

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Lambda RAG Assistant (Advanced)</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1400px;
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
            box-sizing: border-box;
        }
        input[type="number"], select {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-right: 10px;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-right: 10px;
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
            font-size: 13px;
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
        .advanced {
            background: #e7f3ff;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .stats {
            background: #d4edda;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Lambda Project RAG Assistant (Advanced)</h1>
        
        <div class="settings">
            <strong>System Status:</strong><br>
            📊 Embedder: NPU (Chunked Index)<br>
            🤖 LLM: LM Studio (Port 1234)<br>
            📁 Files: <span id="file-count">Loading...</span><br>
            🧩 Chunks: <span id="chunk-count">Loading...</span>
        </div>

        <div class="input-group">
            <label for="question">Ask a question about the Lambda Project:</label>
            <textarea id="question" rows="3" placeholder="e.g., How does the gateway pattern work in this project?"></textarea>
        </div>

        <div class="advanced">
            <label><strong>Search Strategy:</strong></label>
            <select id="search-mode">
                <option value="hierarchical">Hierarchical (20 files, 5 chunks each)</option>
                <option value="chunk">Chunk-Level (100+ precise chunks)</option>
                <option value="file">File-Level (Fast overview, 10-20 files)</option>
            </select>
            
            <br><br>
            
            <label for="num-results">Number of results:</label>
            <input type="number" id="num-results" value="100" min="10" max="500">
            
            <label for="context-length">Max context chars:</label>
            <input type="number" id="context-length" value="8000" min="2000" max="20000" step="1000">
        </div>

        <button onclick="askQuestion()">🔍 Search & Ask LLM</button>
        <button onclick="quickSearch()">⚡ Quick File Search</button>
        
        <div class="loading" id="loading">⏳ Searching project files and querying LLM...</div>

        <div class="results" id="results"></div>
    </div>

    <script>
        // Get stats
        fetch('/api/stats')
            .then(r => r.json())
            .then(data => {
                document.getElementById('file-count').textContent = data.indexed_files;
                document.getElementById('chunk-count').textContent = data.indexed_chunks;
            });

        async function askQuestion() {
            const question = document.getElementById('question').value;
            const searchMode = document.getElementById('search-mode').value;
            const numResults = document.getElementById('num-results').value;
            const contextLength = document.getElementById('context-length').value;
            
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
                        search_mode: searchMode,
                        num_results: parseInt(numResults),
                        max_context_length: parseInt(contextLength)
                    })
                });

                const data = await response.json();

                let html = '<div class="stats">';
                html += `<strong>Search Results:</strong><br>`;
                html += `Found ${data.total_results} results<br>`;
                html += `Using ${data.results_used} in context<br>`;
                html += `Context size: ${data.context_length} chars`;
                html += '</div>';

                html += '<div class="relevant-files"><h3>📂 Relevant Files:</h3>';
                html += '<pre>' + data.file_overview + '</pre>';
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

        async function quickSearch() {
            const question = document.getElementById('question').value;
            
            if (!question) {
                alert('Please enter a question');
                return;
            }

            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';

            try {
                const response = await fetch('/api/quick_search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: question,
                        top_k: 20
                    })
                });

                const data = await response.json();

                let html = '<div class="relevant-files"><h3>📂 Top Files (No LLM Query):</h3>';
                data.results.forEach(file => {
                    html += `<div class="file-item">
                        <strong>${file.file}</strong> (${(file.score * 100).toFixed(1)}% match)
                        <br><small>${file.chunks} chunks indexed</small>
                    </div>`;
                });
                html += '</div>';

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
        'indexed_files': len(indexer.file_embeddings),
        'indexed_chunks': len(indexer.chunk_embeddings),
        'status': 'ready'
    })

@app.route('/api/ask', methods=['POST'])
def ask():
    """Advanced RAG endpoint with 100+ file support"""
    data = request.json
    question = data.get('question', '')
    search_mode = data.get('search_mode', 'hierarchical')
    num_results = data.get('num_results', 100)
    max_context_length = data.get('max_context_length', 8000)
    
    # Step 1: Search based on mode
    if search_mode == 'hierarchical':
        results = indexer.search_hierarchical(
            question,
            file_top_k=20,
            chunks_per_file=5
        )
    elif search_mode == 'chunk':
        results = indexer.search_chunks(question, top_k=num_results)
    else:  # file mode
        results = indexer.search_files(question, top_k=min(num_results, 50))
        # Convert file results to chunk-like format
        chunk_results = []
        for r in results:
            chunk_ids = r['metadata'].get('chunk_ids', [])
            for chunk_id in chunk_ids[:3]:  # Take first 3 chunks per file
                chunk_meta = indexer.chunk_metadata.get(chunk_id, {})
                chunk_results.append({
                    'file': r['file'],
                    'score': r['score'],
                    'content': chunk_meta.get('content', ''),
                    'chunk_num': chunk_meta.get('chunk_num', 0),
                    'type': 'file_based'
                })
        results = chunk_results
    
    # Step 2: Create file overview
    file_overview = create_file_overview(results)
    
    # Step 3: Build context with smart summarization
    context = summarize_chunks(results, max_length=max_context_length)
    
    # Step 4: Build prompt for LLM
    system_prompt = """You are a helpful assistant with access to the Lambda Execution Engine project files. 
Answer questions based on the provided context from the project files. 
The context includes relevant code chunks from multiple files.
If the context doesn't contain the answer, say so clearly."""
    
    user_prompt = f"""Context from relevant project files:
{context}

---

Question: {question}

Answer based on the context above. Be specific and reference the files when relevant."""
    
    # Step 5: Query LM Studio
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
            timeout=90
        )
        
        llm_response.raise_for_status()
        llm_data = llm_response.json()
        llm_answer = llm_data['choices'][0]['message']['content']
        
    except requests.exceptions.ConnectionError:
        llm_answer = "❌ Error: Cannot connect to LM Studio. Make sure it's running on port 1234."
    except Exception as e:
        llm_answer = f"❌ Error querying LLM: {str(e)}"
    
    return jsonify({
        'total_results': len(results),
        'results_used': len([r for r in results if r.get('content', '')]),
        'context_length': len(context),
        'file_overview': file_overview,
        'llm_response': llm_answer,
        'search_mode': search_mode
    })

@app.route('/api/quick_search', methods=['POST'])
def quick_search():
    """Quick file search without LLM"""
    data = request.json
    question = data.get('question', '')
    top_k = data.get('top_k', 20)
    
    results = indexer.search_files(question, top_k=top_k)
    
    return jsonify({
        'results': [
            {
                'file': r['file'],
                'score': r['score'],
                'chunks': r['metadata'].get('chunk_count', 0)
            }
            for r in results
        ]
    })

@app.route('/api/reindex', methods=['POST'])
def reindex():
    """Rebuild the chunked index"""
    try:
        indexer.index_project("./")
        indexer.save_index("lambda_index")
        return jsonify({
            'status': 'success',
            'message': 'Index rebuilt',
            'files': len(indexer.file_embeddings),
            'chunks': len(indexer.chunk_embeddings)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Lambda Advanced RAG Server Starting")
    print("="*60)
    print(f"📊 Files: {len(indexer.file_embeddings)}")
    print(f"🧩 Chunks: {len(indexer.chunk_embeddings)}")
    print(f"🌐 Web UI: http://localhost:8000")
    print(f"🤖 LM Studio: {LM_STUDIO_URL}")
    print("="*60 + "\n")
    print("💡 Features:")
    print("  - Hierarchical search (20 files × 5 chunks = 100 results)")
    print("  - Chunk-level search (100+ precise chunks)")
    print("  - Smart summarization for large contexts")
    print("  - File overview before LLM query")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8000, debug=False)

# EOF
