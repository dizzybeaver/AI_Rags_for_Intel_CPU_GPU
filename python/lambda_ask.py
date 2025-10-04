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
        console.print("\n[cyan]Examples:[/cyan]")
        console.print('  python lambda_ask.py "How does the gateway pattern work?"')
        console.print('  python lambda_ask.py "What files handle logging?"')
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    ask_question(question)

# EOF
