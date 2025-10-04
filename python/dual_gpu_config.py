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
Dual GPU Configuration Helper
Automatically configures optimal device allocation for dual Arc B580 setup
"""

from openvino.runtime import Core
import json
import os

def detect_hardware():
    """Detect available Intel hardware"""
    ie = Core()
    devices = ie.available_devices
    
    gpus = [d for d in devices if d.startswith('GPU')]
    has_npu = 'NPU' in devices
    
    return {
        'gpus': gpus,
        'gpu_count': len(gpus),
        'has_npu': has_npu,
        'all_devices': devices
    }

def recommend_configuration(hardware):
    """Recommend optimal device allocation"""
    gpu_count = hardware['gpu_count']
    has_npu = hardware['has_npu']
    
    config = {
        'embedding_device': None,
        'lm_studio_gpu': None,
        'setup_type': None,
        'recommendations': []
    }
    
    if gpu_count >= 2:
        # Dual GPU setup (optimal)
        config['embedding_device'] = 'GPU.1'
        config['lm_studio_gpu'] = 'GPU.0'
        config['setup_type'] = 'dual_gpu_optimal'
        config['recommendations'] = [
            'Use GPU.0 for LM Studio (main LLM)',
            'Use GPU.1 for embeddings (background indexing)',
            'Load Qwen 2.5 32B on GPU.0 for best quality',
            'Or run two models: 32B on GPU.0, 14B Coder on GPU.1'
        ]
    elif gpu_count == 1:
        # Single GPU setup
        if has_npu:
            config['embedding_device'] = 'NPU'
            config['lm_studio_gpu'] = 'GPU.0'
            config['setup_type'] = 'single_gpu_with_npu'
            config['recommendations'] = [
                'Use GPU.0 for LM Studio (main LLM)',
                'Use NPU for embeddings (no GPU impact)',
                'Load Qwen 2.5 Coder 14B on GPU.0'
            ]
        else:
            config['embedding_device'] = 'CPU'
            config['lm_studio_gpu'] = 'GPU.0'
            config['setup_type'] = 'single_gpu_cpu_embeddings'
            config['recommendations'] = [
                'Use GPU.0 for LM Studio (main LLM)',
                'Use CPU for embeddings (slower but works)',
                'Load Qwen 2.5 Coder 14B on GPU.0'
            ]
    else:
        # No GPU (CPU/NPU only)
        config['embedding_device'] = 'NPU' if has_npu else 'CPU'
        config['lm_studio_gpu'] = 'CPU'
        config['setup_type'] = 'no_gpu'
        config['recommendations'] = [
            'No GPU detected - using CPU for everything',
            'Performance will be limited',
            'Consider using smaller models'
        ]
    
    return config

def save_configuration(config, filename='rag_config.json'):
    """Save configuration to file"""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration saved to {filename}")

def update_code_files(embedding_device):
    """Update Python files with optimal device"""
    files_to_update = [
        'lambda_indexer.py',
        'rag_server.py',
        'auto_index.py'
    ]
    
    print("\n📝 Updating code files...")
    
    for filename in files_to_update:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
            
            # Replace device parameter
            # Look for: LambdaProjectIndexer(device="...")
            import re
            pattern = r'LambdaProjectIndexer\(device="[^"]*"\)'
            replacement = f'LambdaProjectIndexer(device="{embedding_device}")'
            
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                
                with open(filename, 'w') as f:
                    f.write(content)
                
                print(f"✓ Updated {filename} to use {embedding_device}")
            else:
                print(f"⚠ Could not find device parameter in {filename}")
        else:
            print(f"⚠ File not found: {filename}")

def main():
    print("🚀 Lambda RAG - Dual GPU Configuration Helper")
    print("=" * 60)
    
    # Detect hardware
    print("\n🔍 Detecting hardware...")
    hardware = detect_hardware()
    
    print(f"\n📊 Detected Hardware:")
    print(f"   GPUs: {hardware['gpu_count']} ({', '.join(hardware['gpus']) if hardware['gpus'] else 'None'})")
    print(f"   NPU: {'Yes' if hardware['has_npu'] else 'No'}")
    
    # Get recommendations
    config = recommend_configuration(hardware)
    
    print(f"\n✅ Recommended Configuration:")
    print(f"   Setup Type: {config['setup_type']}")
    print(f"   Embedding Device: {config['embedding_device']}")
    print(f"   LM Studio GPU: {config['lm_studio_gpu']}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(config['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Save configuration
    save_configuration(config)
    
    # Ask if user wants to update code files
    print("\n" + "=" * 60)
    response = input("\n🔧 Update code files automatically? (y/n): ").lower().strip()
    
    if response == 'y':
        update_code_files(config['embedding_device'])
        print("\n✅ Configuration complete!")
        print("\n📌 Next Steps:")
        print("   1. Start LM Studio and load a model on GPU.0")
        print("   2. Run: ./start_lambda_rag.sh (Linux/Mac)")
        print("   2. Or: start_lambda_rag.bat (Windows)")
        print("   3. Open: http://localhost:8000")
    else:
        print("\n📝 Manual Configuration:")
        print(f"   Edit Python files to use: device=\"{config['embedding_device']}\"")
        print(f"   Configure LM Studio to use: {config['lm_studio_gpu']}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

# EOF
