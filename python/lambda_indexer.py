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
Lambda Project Indexer
Embeds project files using BGE-Small on NPU/GPU for semantic search
"""

import os
from pathlib import Path
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np
import json
from typing import List, Dict

class LambdaProjectIndexer:
    def __init__(self, device="NPU"):
        """
        Initialize embedder on specified device
        
        Args:
            device: One of "NPU", "GPU.0", "GPU.1", "CPU"
        """
        print(f"Initializing embedder on {device}...")
        self.device = device
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
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
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
                'venv', '__pycache__', '.git', 'node_modules', '.venv'
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
        print(f"✓ Index saved to {output_path}")
    
    def load_index(self, input_path: str):
        """Load pre-computed embeddings"""
        data = np.load(f"{input_path}_embeddings.npz")
        self.embeddings = {k: data[k] for k in data.files}
        
        with open(f"{input_path}_metadata.json", 'r') as f:
            self.metadata = json.load(f)
        print(f"✓ Index loaded from {input_path}")


if __name__ == "__main__":
    # Example usage
    indexer = LambdaProjectIndexer(device="NPU")
    
    # Index the entire project
    indexer.index_project("./")
    
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

# EOF
