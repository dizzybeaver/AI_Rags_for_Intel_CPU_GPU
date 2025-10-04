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
Lambda Project RAG Server
Web interface for semantic search + LM Studio integration
"""

from flask import Flask, request, jsonify, render_template_string
from lambda_indexer import LambdaProjectIndexer
import requests
import os

app = Flask(__name__)

# Initialize indexer (uses NPU by default, change to "GPU.1" for dual GPU setup)
print("Loading embeddings...")
indexer = LambdaProjectIndexer(device="NPU")

# Load pre-built index or create new one
if os.path.exists("lambda_index_embeddings.npz"):
    indexer.load_index("lambda_index")
    print("✓ Loaded existing index")
else:
    print("Building new index...")
    indexer.index_project("./")
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
            box-sizing: border-box;
        }
        input[type="number"] {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
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
            <input type="number" id="num-files" value="3" min="1" max="10">
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

# EOF
