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
Auto-indexer for Lambda Project
Watches for file changes and automatically re-indexes
"""

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from lambda_indexer import LambdaProjectIndexer
import time
import os
from pathlib import Path

class ProjectFileHandler(FileSystemEventHandler):
    def __init__(self, indexer, project_path):
        self.indexer = indexer
        self.project_path = project_path
        self.last_reindex = 0
        self.reindex_delay = 2  # Wait 2 seconds before reindexing
        
    def should_index(self, filepath):
        """Check if file should trigger reindexing"""
        extensions = {'.py', '.md', '.json', '.yaml', '.yml', '.txt'}
        path = Path(filepath)
        
        # Skip ignored directories
        ignored = {'venv', '__pycache__', '.git', 'node_modules', '.venv'}
        if any(part in ignored for part in path.parts):
            return False
        
        return path.suffix in extensions
    
    def trigger_reindex(self, event):
        """Trigger reindex with debouncing"""
        current_time = time.time()
        
        # Debounce: only reindex if 2 seconds have passed
        if current_time - self.last_reindex < self.reindex_delay:
            return
        
        if self.should_index(event.src_path):
            print(f"\n📝 File changed: {event.src_path}")
            print("🔄 Re-indexing project...")
            
            try:
                self.indexer.index_project(self.project_path)
                self.indexer.save_index("lambda_index")
                print("✅ Index updated successfully\n")
                self.last_reindex = current_time
            except Exception as e:
                print(f"❌ Error re-indexing: {e}\n")
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory:
            self.trigger_reindex(event)
    
    def on_created(self, event):
        """Handle file creation"""
        if not event.is_directory:
            self.trigger_reindex(event)
    
    def on_deleted(self, event):
        """Handle file deletion"""
        if not event.is_directory:
            self.trigger_reindex(event)

def main():
    # Get project path
    project_path = os.path.abspath("./")
    
    # Initialize indexer
    print("🚀 Starting Lambda Project Auto-Indexer")
    print("=" * 60)
    print(f"📁 Watching: {project_path}")
    print(f"📊 Device: NPU (change in code if needed)")
    print("=" * 60)
    
    indexer = LambdaProjectIndexer(device="NPU")
    
    # Load or create initial index
    if os.path.exists("lambda_index_embeddings.npz"):
        print("\n✓ Loading existing index...")
        indexer.load_index("lambda_index")
        print(f"✓ {len(indexer.embeddings)} files indexed")
    else:
        print("\n📊 Building initial index...")
        indexer.index_project(project_path)
        indexer.save_index("lambda_index")
        print(f"✓ {len(indexer.embeddings)} files indexed")
    
    # Set up file watcher
    event_handler = ProjectFileHandler(indexer, project_path)
    observer = Observer()
    observer.schedule(event_handler, project_path, recursive=True)
    observer.start()
    
    print("\n👀 Watching for file changes...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping auto-indexer...")
        observer.stop()
        observer.join()
        print("✓ Stopped successfully")

if __name__ == "__main__":
    main()

# EOF
