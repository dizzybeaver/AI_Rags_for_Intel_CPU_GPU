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
Lambda RAG - Search Strategy Examples
Demonstrates different search modes for various use cases
"""

from lambda_indexer_chunked import LambdaProjectIndexer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def example_1_file_level_search():
    """Example 1: Fast file discovery"""
    console.print("\n[bold cyan]Example 1: File-Level Search (Fast Discovery)[/bold cyan]\n")
    
    indexer = LambdaProjectIndexer(device="NPU")
    indexer.load_index("lambda_index")
    
    question = "What files handle authentication and security?"
    console.print(f"Question: [yellow]{question}[/yellow]\n")
    
    results = indexer.search_files(question, top_k=10)
    
    table = Table(title="Top 10 Files")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("File", style="green")
    table.add_column("Score", style="magenta", width=8)
    table.add_column("Chunks", style="blue", width=8)
    
    for i, result in enumerate(results, 1):
        table.add_row(
            str(i),
            result['file'],
            f"{result['score']:.1%}",
            str(result['metadata']['chunk_count'])
        )
    
    console.print(table)
    console.print("\n[green]Use case:[/green] Quick exploration, understanding project structure")
    console.print("[green]Speed:[/green] ~100-200ms")


def example_2_chunk_level_search():
    """Example 2: Precise code finding"""
    console.print("\n[bold cyan]Example 2: Chunk-Level Search (Precise)[/bold cyan]\n")
    
    indexer = LambdaProjectIndexer(device="NPU")
    indexer.load_index("lambda_index")
    
    question = "Show me the exact implementation of operation routing in the gateway"
    console.print(f"Question: [yellow]{question}[/yellow]\n")
    
    results = indexer.search_chunks(question, top_k=100)
    
    # Show top 10
    table = Table(title="Top 10 Chunks (of 100 total)")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("File", style="green")
    table.add_column("Chunk", style="blue", width=8)
    table.add_column("Score", style="magenta", width=8)
    table.add_column("Preview", style="white")
    
    for i, result in enumerate(results[:10], 1):
        preview = result['content'][:60].replace('\n', ' ') + "..."
        table.add_row(
            str(i),
            result['file'].split('/')[-1],  # Just filename
            str(result['chunk_num']),
            f"{result['score']:.1%}",
            preview
        )
    
    console.print(table)
    
    # Show file distribution
    files = {}
    for r in results[:20]:
        files[r['file']] = files.get(r['file'], 0) + 1
    
    console.print(f"\n[green]File distribution in top 20:[/green]")
    for file, count in sorted(files.items(), key=lambda x: x[1], reverse=True):
        console.print(f"  {file}: {count} chunks")
    
    console.print("\n[green]Use case:[/green] Finding exact code, debugging, specific implementations")
    console.print("[green]Speed:[/green] ~500-800ms")


def example_3_hierarchical_search():
    """Example 3: Comprehensive coverage"""
    console.print("\n[bold cyan]Example 3: Hierarchical Search (Balanced)[/bold cyan]\n")
    
    indexer = LambdaProjectIndexer(device="NPU")
    indexer.load_index("lambda_index")
    
    question = "How is error handling implemented across the project?"
    console.print(f"Question: [yellow]{question}[/yellow]\n")
    
    results = indexer.search_hierarchical(
        question,
        file_top_k=15,
        chunks_per_file=5
    )
    
    # Group by file
    files = {}
    for result in results:
        filepath = result['file']
        if filepath not in files:
            files[filepath] = []
        files[filepath].append(result)
    
    # Show structure
    console.print("[green]Search Structure:[/green]")
    console.print(f"  Stage 1: Found {len(files)} relevant files")
    console.print(f"  Stage 2: Retrieved {len(results)} total chunks")
    console.print(f"  Distribution: ~{len(results)//len(files)} chunks per file\n")
    
    table = Table(title="Files and Their Top Chunks")
    table.add_column("File", style="green")
    table.add_column("Chunks", style="blue", width=8)
    table.add_column("Best Score", style="magenta", width=10)
    table.add_column("Avg Score", style="cyan", width=10)
    
    for filepath, chunks in list(files.items())[:10]:
        scores = [c['score'] for c in chunks]
        table.add_row(
            filepath.split('/')[-1],
            str(len(chunks)),
            f"{max(scores):.1%}",
            f"{sum(scores)/len(scores):.1%}"
        )
    
    console.print(table)
    console.print("\n[green]Use case:[/green] Broad questions, architecture understanding, patterns")
    console.print("[green]Speed:[/green] ~300-500ms")
    console.print("[green]Coverage:[/green] Balanced breadth and depth")


def example_4_comparison():
    """Example 4: Side-by-side comparison"""
    console.print("\n[bold cyan]Example 4: Strategy Comparison[/bold cyan]\n")
    
    indexer = LambdaProjectIndexer(device="NPU")
    indexer.load_index("lambda_index")
    
    question = "gateway architecture"
    console.print(f"Question: [yellow]{question}[/yellow]\n")
    
    import time
    
    # File-level
    start = time.time()
    file_results = indexer.search_files(question, top_k=10)
    file_time = (time.time() - start) * 1000
    
    # Chunk-level
    start = time.time()
    chunk_results = indexer.search_chunks(question, top_k=100)
    chunk_time = (time.time() - start) * 1000
    
    # Hierarchical
    start = time.time()
    hier_results = indexer.search_hierarchical(question, file_top_k=15, chunks_per_file=5)
    hier_time = (time.time() - start) * 1000
    
    # Comparison table
    table = Table(title="Performance Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Results", style="green")
    table.add_column("Time (ms)", style="magenta")
    table.add_column("Top Score", style="blue")
    table.add_column("Best For", style="yellow")
    
    table.add_row(
        "File-Level",
        str(len(file_results)),
        f"{file_time:.0f}",
        f"{file_results[0]['score']:.1%}",
        "Quick exploration"
    )
    
    table.add_row(
        "Chunk-Level",
        str(len(chunk_results)),
        f"{chunk_time:.0f}",
        f"{chunk_results[0]['score']:.1%}",
        "Precise code finding"
    )
    
    table.add_row(
        "Hierarchical",
        str(len(hier_results)),
        f"{hier_time:.0f}",
        f"{hier_results[0]['score']:.1%}",
        "Comprehensive coverage"
    )
    
    console.print(table)


def example_5_custom_pipeline():
    """Example 5: Custom search pipeline"""
    console.print("\n[bold cyan]Example 5: Custom Search Pipeline[/bold cyan]\n")
    
    indexer = LambdaProjectIndexer(device="NPU")
    indexer.load_index("lambda_index")
    
    question = "caching mechanisms and performance optimization"
    console.print(f"Question: [yellow]{question}[/yellow]\n")
    
    console.print("[green]Step 1:[/green] Quick file-level scan")
    file_results = indexer.search_files(question, top_k=5)
    console.print(f"  Found {len(file_results)} relevant files\n")
    
    console.print("[green]Step 2:[/green] Deep dive into top file")
    top_file = file_results[0]['file']
    console.print(f"  Deep diving into: {top_file}\n")
    
    # Get all chunks from top file
    all_chunks = indexer.search_chunks(question, top_k=200)
    file_chunks = [c for c in all_chunks if c['file'] == top_file]
    
    console.print(f"[green]Step 3:[/green] Analyze chunks from {top_file}")
    for i, chunk in enumerate(file_chunks[:5], 1):
        preview = chunk['content'][:80].replace('\n', ' ')
        console.print(f"  {i}. Chunk {chunk['chunk_num']}: {chunk['score']:.1%} - {preview}...")
    
    console.print("\n[green]Use case:[/green] Targeted deep dive after exploration")


def example_6_quality_thresholds():
    """Example 6: Using quality thresholds"""
    console.print("\n[bold cyan]Example 6: Quality Filtering[/bold cyan]\n")
    
    indexer = LambdaProjectIndexer(device="NPU")
    indexer.load_index("lambda_index")
    
    question = "database connection pooling"
    console.print(f"Question: [yellow]{question}[/yellow]\n")
    
    results = indexer.search_chunks(question, top_k=100)
    
    # Filter by different thresholds
    thresholds = {
        "Excellent (>90%)": [r for r in results if r['score'] > 0.9],
        "Great (80-90%)": [r for r in results if 0.8 <= r['score'] <= 0.9],
        "Good (70-80%)": [r for r in results if 0.7 <= r['score'] <= 0.8],
        "Fair (<70%)": [r for r in results if r['score'] < 0.7]
    }
    
    table = Table(title="Results by Quality")
    table.add_column("Quality", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("% of Total", style="magenta")
    table.add_column("Recommendation", style="yellow")
    
    for quality, chunks in thresholds.items():
        pct = (len(chunks) / len(results)) * 100 if results else 0
        
        if "Excellent" in quality:
            rec = "Use all - very relevant"
        elif "Great" in quality:
            rec = "Include in context"
        elif "Good" in quality:
            rec = "Consider if space allows"
        else:
            rec = "Skip unless desperate"
        
        table.add_row(quality, str(len(chunks)), f"{pct:.1f}%", rec)
    
    console.print(table)
    console.print("\n[green]Tip:[/green] Set minimum threshold based on model context size")
    console.print("  Small model (8K context): Use only >85% matches")
    console.print("  Large model (128K context): Can include >70% matches")


def main():
    """Run all examples"""
    console.print(Panel.fit(
        "[bold cyan]Lambda RAG - Search Strategy Examples[/bold cyan]\n"
        "Demonstrating different search modes with real queries",
        border_style="blue"
    ))
    
    examples = [
        ("File-Level Search", example_1_file_level_search),
        ("Chunk-Level Search", example_2_chunk_level_search),
        ("Hierarchical Search", example_3_hierarchical_search),
        ("Strategy Comparison", example_4_comparison),
        ("Custom Pipeline", example_5_custom_pipeline),
        ("Quality Filtering", example_6_quality_thresholds)
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        console.print(f"\n[bold]{'='*60}[/bold]")
        func()
        
        if i < len(examples):
            input("\nPress Enter to continue to next example...")
    
    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print("\n[bold green]✅ All examples completed![/bold green]\n")
    console.print("💡 Try these strategies with your own questions:")
    console.print("  • Use file-level for exploration")
    console.print("  • Use chunk-level for precision")
    console.print("  • Use hierarchical for comprehensive coverage")
    console.print("  • Combine strategies for complex questions")


if __name__ == "__main__":
    main()

# EOF
