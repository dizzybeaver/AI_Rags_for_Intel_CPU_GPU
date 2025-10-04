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
Clipboard Context Enhancer
Automatically adds project context to questions copied to clipboard
"""

import pyperclip
import time
from lambda_indexer import LambdaProjectIndexer

def main():
    print("🚀 Lambda Clipboard Context Enhancer")
    print("=" * 60)
    print("📋 Watching clipboard for questions...")
    print("💡 Copy any question ending with '?' to auto-enhance it")
    print("Press Ctrl+C to stop\n")
    
    # Initialize indexer
    indexer = LambdaProjectIndexer(device="NPU")
    indexer.load_index("lambda_index")
    print(f"✓ Loaded index with {len(indexer.embeddings)} files\n")
    
    last_clipboard = ""
    
    while True:
        try:
            current = pyperclip.paste()
            
            # New question detected
            if current != last_clipboard and current.strip().endswith("?"):
                question = current.strip()
                print(f"\n📋 Detected: {question[:60]}{'...' if len(question) > 60 else ''}")
                
                # Search for context
                print("🔍 Searching for relevant context...")
                results = indexer.search(question, top_k=3)
                
                # Build enhanced prompt
                enhanced = f"{question}\n\n"
                enhanced += "📂 Relevant project files for context:\n"
                for i, r in enumerate(results, 1):
                    enhanced += f"{i}. {r['file']} ({r['score']:.1%} relevant)\n"
                
                # Put back in clipboard
                pyperclip.copy(enhanced)
                print("✅ Enhanced with context!")
                print("📋 Paste into LM Studio or your chat interface\n")
                
                last_clipboard = current
            
            time.sleep(0.5)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped watching clipboard")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()

# EOF
