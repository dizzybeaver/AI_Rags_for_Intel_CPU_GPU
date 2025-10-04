#!/bin/bash
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

# Lambda RAG System Startup Script (Linux/Mac)

echo "🚀 Starting Lambda RAG System..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

# Check if index exists
if [ ! -f "lambda_index_embeddings.npz" ]; then
    echo "📊 Building initial index..."
    python3 -c "from lambda_indexer import LambdaProjectIndexer; i = LambdaProjectIndexer(device='NPU'); i.index_project('./'); i.save_index('lambda_index')"
    echo ""
fi

# Start auto-indexer in background
echo "📁 Starting auto-indexer (background)..."
python3 auto_index.py > auto_index.log 2>&1 &
INDEXER_PID=$!
echo "   PID: $INDEXER_PID"

# Start RAG server
echo "🌐 Starting RAG web server..."
python3 rag_server.py > rag_server.log 2>&1 &
RAG_PID=$!
echo "   PID: $RAG_PID"

# Wait for server to start
sleep 2

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Lambda RAG System is Ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Web UI:       http://localhost:8000"
echo "🤖 LM Studio:    http://localhost:1234 (make sure it's running!)"
echo "📁 Auto-indexer: Running in background"
echo "📝 Logs:"
echo "   - auto_index.log"
echo "   - rag_server.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $INDEXER_PID 2>/dev/null
    kill $RAG_PID 2>/dev/null
    echo "✓ Services stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Wait indefinitely
while true; do
    sleep 1
    
    # Check if processes are still running
    if ! kill -0 $RAG_PID 2>/dev/null; then
        echo "❌ RAG server died. Check rag_server.log"
        cleanup
    fi
    
    if ! kill -0 $INDEXER_PID 2>/dev/null; then
        echo "❌ Auto-indexer died. Check auto_index.log"
        cleanup
    fi
done

# EOF
