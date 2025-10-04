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
Upgrade to Chunked Indexing System
Migrates from basic indexer to chunked indexer with 100+ file support
"""

import os
import shutil
from datetime import datetime

def backup_old_index():
    """Backup existing index files"""
    backup_dir = f"backup_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    files_to_backup = [
        'lambda_index_embeddings.npz',
        'lambda_index_metadata.json'
    ]
    
    backed_up = []
    for filename in files_to_backup:
        if os.path.exists(filename):
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copy2(filename, os.path.join(backup_dir, filename))
            backed_up.append(filename)
    
    if backed_up:
        print(f"✓ Backed up {len(backed_up)} files to {backup_dir}/")
        return backup_dir
    else:
        print("ℹ No existing index found - fresh installation")
        return None

def clean_old_index():
    """Remove old index files"""
    files_to_remove = [
        'lambda_index_embeddings.npz',
        'lambda_index_metadata.json'
    ]
    
    removed = []
    for filename in files_to_remove:
        if os.path.exists(filename):
            os.remove(filename)
            removed.append(filename)
    
    if removed:
        print(f"✓ Cleaned {len(removed)} old index files")

def build_new_index():
    """Build new chunked index"""
    print("\n🔨 Building new chunked index...")
    print("This may take a few minutes depending on project size...")
    
    from lambda_indexer_chunked import LambdaProjectIndexer
    
    indexer = LambdaProjectIndexer(device="NPU")
    indexer.index_project("./")
    indexer.save_index("lambda_index")
    
    print("\n✅ New chunked index built successfully!")
    print(f"   Files indexed: {len(indexer.file_embeddings)}")
    print(f"   Chunks created: {len(indexer.chunk_embeddings)}")
    
    return indexer

def update_startup_scripts():
    """Update startup scripts to use advanced server"""
    updates = {
        'start_lambda_rag.sh': {
            'old': 'python3 rag_server.py',
            'new': 'python3 rag_server_advanced.py'
        },
        'start_lambda_rag.bat': {
            'old': 'python rag_server.py',
            'new': 'python rag_server_advanced.py'
        }
    }
    
    updated = []
    for filename, changes in updates.items():
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
            
            if changes['old'] in content:
                content = content.replace(changes['old'], changes['new'])
                with open(filename, 'w') as f:
                    f.write(content)
                updated.append(filename)
    
    if updated:
        print(f"\n✓ Updated {len(updated)} startup scripts to use advanced server")

def show_comparison():
    """Show comparison between old and new systems"""
    print("\n" + "="*60)
    print("📊 UPGRADE COMPARISON")
    print("="*60)
    print("\nOLD SYSTEM (Basic):")
    print("  • File-level indexing only")
    print("  • Max 10-20 files per query")
    print("  • 2000 chars per file limit")
    print("  • Total context: ~20-40KB")
    print("  • No code structure awareness")
    
    print("\nNEW SYSTEM (Chunked):")
    print("  • File + chunk-level indexing")
    print("  • 100+ chunks per query")
    print("  • Smart code-aware chunking")
    print("  • Total context: configurable (8-20KB)")
    print("  • Hierarchical search strategies")
    print("  • Better precision and recall")
    
    print("\n" + "="*60)

def main():
    print("="*60)
    print("🚀 Lambda RAG System - Upgrade to Chunked Indexing")
    print("="*60)
    print("\nThis will upgrade your system to support 100+ file queries")
    print("with smart chunking and hierarchical search.\n")
    
    # Check if advanced files exist
    required_files = [
        'lambda_indexer_chunked.py',
        'rag_server_advanced.py'
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("❌ Error: Missing required files:")
        for f in missing:
            print(f"   - {f}")
        print("\nPlease ensure all advanced system files are present.")
        return
    
    print("✓ All required files found\n")
    
    # Ask for confirmation
    response = input("Continue with upgrade? (y/n): ").lower().strip()
    if response != 'y':
        print("Upgrade cancelled.")
        return
    
    print("\n" + "="*60)
    print("STEP 1: Backup existing index")
    print("="*60)
    backup_dir = backup_old_index()
    
    print("\n" + "="*60)
    print("STEP 2: Clean old index files")
    print("="*60)
    clean_old_index()
    
    print("\n" + "="*60)
    print("STEP 3: Build new chunked index")
    print("="*60)
    indexer = build_new_index()
    
    print("\n" + "="*60)
    print("STEP 4: Update startup scripts")
    print("="*60)
    update_startup_scripts()
    
    # Show comparison
    show_comparison()
    
    # Final instructions
    print("\n" + "="*60)
    print("✅ UPGRADE COMPLETE!")
    print("="*60)
    print("\n📌 Next Steps:")
    print("1. Restart your RAG server:")
    print("   ./start_lambda_rag.sh  (Linux/Mac)")
    print("   start_lambda_rag.bat    (Windows)")
    print("\n2. Open http://localhost:8000")
    print("\n3. Try the new search modes:")
    print("   • Hierarchical (default - 100 results)")
    print("   • Chunk-level (precise - 100+ chunks)")
    print("   • File-level (fast overview)")
    
    if backup_dir:
        print(f"\n💾 Backup saved to: {backup_dir}/")
        print("   (Can be deleted if upgrade works well)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()

# EOF
