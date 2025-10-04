# Dual Intel Arc B580 Setup Guide (24GB Total VRAM)

## Your New Hardware Capabilities

### With 2x Arc B580 (12GB each):
- **Total VRAM**: 24GB
- **Total Compute**: 2x XMX engines, 2x Ray Tracing units
- **Multi-GPU**: Can run models in parallel or split large models

**This changes EVERYTHING!** 🚀

---

## Major Changes vs Single B580

### Model Size Upgrades

| Single B580 (12GB) | Dual B580 (24GB) |
|-------------------|------------------|
| Qwen 2.5 14B (Q5) | **Qwen 2.5 32B (Unquantized)** |
| DeepSeek 16B (Q4) | **Qwen 2.5 72B (Q4)** |
| Max: ~20B params | **Max: ~70B+ params** |

### New Capabilities

✅ **Run TWO models simultaneously** (one per GPU)
✅ **Larger context windows** (more VRAM headroom)
✅ **Faster inference** (model splitting across GPUs)
✅ **Production-grade models** (70B class performance)

---

## LM Studio Configuration for Dual GPUs

### Method 1: Single Large Model Across Both GPUs

#### Step 1: Enable Multi-GPU in LM Studio

1. Open **LM Studio**
2. Go to **Settings** → **Inference**
3. Under **GPU Acceleration**:
   - Enable **GPU Offloading**
   - Select **Intel GPU (OpenVINO)**
   - **Important**: Set GPU layers to **999** (max)
   - Check **"Use multiple GPUs"** if available

#### Step 2: Verify Both GPUs Are Detected

In LM Studio console, look for:
```
Detected GPUs:
  GPU 0: Intel Arc B580 (12GB)
  GPU 1: Intel Arc B580 (12GB)
Total VRAM: 24GB
```

#### Step 3: Load Recommended Model

**Best for dual B580**:
```
Model: Qwen 2.5 72B Instruct (Q4_K_M)
VRAM Usage: ~20-22GB (split across both GPUs)
Performance: 15-25 tokens/sec
Quality: Exceptional (near GPT-4 class)
```

**Alternative - Maximum Speed**:
```
Model: Qwen 2.5 32B Instruct (FP16 - unquantized)
VRAM Usage: ~18GB
Performance: 30-40 tokens/sec
Quality: Better than 14B, faster than 72B
```

### LM Studio Settings for Dual GPU:
```
GPU Offload: 999 layers
Context Length: 32768 (you have VRAM for it now!)
Batch Size: 512
Temperature: 0.1-0.3 (for code)
```

---

## Method 2: TWO Models Running Simultaneously

### Strategy: Specialized Models on Each GPU

This is POWERFUL - run different models for different tasks!

#### Configuration A: Code + Chat
```
GPU 0 (Port 1234): Qwen 2.5 Coder 14B
  └─ Specialized code generation
  └─ Fast iteration cycles
  └─ ~10GB VRAM

GPU 1 (Port 1235): Qwen 2.5 32B Instruct  
  └─ General reasoning and planning
  └─ Architecture decisions
  └─ ~18GB VRAM
```

#### Configuration B: Large + Embedding
```
GPU 0 (Port 1234): Qwen 2.5 72B (Q4)
  └─ Main heavy-duty LLM
  └─ ~22GB VRAM

GPU 1: Keep free for embeddings/RAG
  └─ Run embedding models
  └─ Real-time document processing
  └─ OR run smaller specialized model
```

#### Configuration C: Development + Production
```
GPU 0: Development model (fast iterations)
  └─ Qwen 2.5 14B Coder
  └─ Quick tests and iterations

GPU 1: Quality model (final generation)
  └─ Qwen 2.5 72B
  └─ Final code review and refactoring
```

---

## How to Run Multiple LM Studio Instances

### Option 1: Two LM Studio Instances (Easiest)

**Instance 1 (GPU 0):**
```bash
# Start LM Studio normally
# Settings → GPU → Select GPU 0
# Server → Port 1234
# Load Qwen 2.5 Coder 14B
```

**Instance 2 (GPU 1):**
```bash
# Start second LM Studio instance
# Settings → GPU → Select GPU 1  
# Server → Port 1235
# Load Qwen 2.5 32B
```

### Option 2: llama.cpp with SYCL (More Control)

```bash
# Build llama.cpp with Intel GPU support
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release

# Run model split across BOTH GPUs
./build/bin/llama-server \
  -m models/qwen2.5-72b-q4.gguf \
  -ngl 99 \
  --split-mode row \
  -ts 0,1 \
  --port 1234

# This splits the model across GPU 0 and GPU 1
```

**Verify multi-GPU usage:**
```bash
# In another terminal, monitor both GPUs
intel_gpu_top
# Should show activity on both GPU 0 and GPU 1
```

---

## Updated Model Recommendations (24GB VRAM)

### 🥇 Tier 1: Best for Dual B580

#### **Qwen 2.5 72B Instruct (Q4_K_M)** - FLAGSHIP
```
Model: qwen2.5-72b-instruct-q4_k_m.gguf
VRAM: 20-22GB (across both GPUs)
Speed: 15-25 tokens/sec
Quality: ★★★★★ (Near GPT-4 level)
```
**Why**: This is the sweet spot. Massive model, excellent reasoning, fits perfectly.

#### **Qwen 2.5 32B Instruct (FP16)** - SPEED
```
Model: qwen2.5-32b-instruct-fp16.gguf
VRAM: 18GB
Speed: 30-40 tokens/sec
Quality: ★★★★★
```
**Why**: Unquantized quality, fast inference, leaves VRAM for large context.

#### **DeepSeek Coder V2 236B (Q2_K)** - MAXIMUM CAPABILITY
```
Model: deepseek-coder-v2-236b-q2_k.gguf
VRAM: 23GB
Speed: 5-10 tokens/sec
Quality: ★★★★★ (Best coding model available)
```
**Why**: The ultimate coding model. Slow but incredible quality. Only possible with 24GB.

---

### 🥈 Tier 2: Dual-Model Setups

#### Setup 1: Speed + Power
```
GPU 0: Qwen 2.5 14B Coder (Q5) - 10GB
GPU 1: Qwen 2.5 32B Instruct - 18GB
Total: 28GB (but they run independently)
```

#### Setup 2: Specialized Coding
```
GPU 0: DeepSeek Coder 16B - 9GB
GPU 1: Qwen 2.5 32B - 18GB
```

---

## OpenVINO Multi-GPU Configuration

### Update Your Python Scripts

```python
# Updated lambda_indexer.py for dual GPU

from openvino.runtime import Core

# Initialize with MULTI device plugin
ie = Core()

# Option 1: Use both GPUs automatically (load balancing)
device = "MULTI:GPU.0,GPU.1"

# Option 2: Heterogeneous (primary + fallback)
device = "HETERO:GPU.1,GPU.0,NPU,CPU"

# Option 3: Specific GPU selection
device = "GPU.1"  # Use second GPU

model = OVModelForFeatureExtraction.from_pretrained(
    "BAAI/bge-small-en-v1.5",
    export=True,
    device=device
)
```

### Multi-Device Strategy for Lambda Project

```python
# Optimal resource allocation

# GPU 0 (B580 #1): Main LLM
# - Qwen 72B or 32B for code generation
# - Port 1234

# GPU 1 (B580 #2): Support tasks
# - Embedding model (small, ~2GB)
# - Secondary LLM for chat/planning
# - Port 1235

# NPU: Background indexing
# - File watcher and auto-indexing
# - Zero impact on GPUs

# CPU: System tasks
# - Web server (RAG API)
# - File I/O
```

---

## Updated RAG Server for Dual GPU

### Enhanced `rag_server.py`

```python
# At the top of rag_server.py, update:

# Use GPU 1 for embeddings (GPU 0 busy with LLM)
indexer = LambdaProjectIndexer(device="GPU.1")  # Changed from NPU

# LLM endpoints for both models
LM_STUDIO_CODE = "http://localhost:1234/v1/chat/completions"  # GPU 0
LM_STUDIO_CHAT = "http://localhost:1235/v1/chat/completions"  # GPU 1 (optional)

# Add model selection in the API
@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    use_large_model = data.get('use_large_model', False)
    
    endpoint = LM_STUDIO_CHAT if use_large_model else LM_STUDIO_CODE
    
    # ... rest of code
```

---

## Performance Comparison

### Single B580 vs Dual B580

| Metric | Single B580 | Dual B580 |
|--------|-------------|-----------|
| **Max Model Size** | ~20B params | ~70B+ params |
| **Best Model** | Qwen 14B (Q5) | Qwen 72B (Q4) |
| **Context Length** | 8K-16K tokens | 32K-64K tokens |
| **Speed (14B)** | 25-35 tok/s | 25-35 tok/s |
| **Speed (32B)** | Won't fit | 30-40 tok/s |
| **Speed (72B)** | Won't fit | 15-25 tok/s |
| **Parallel Models** | ❌ No | ✅ Yes (2 models) |
| **Production Ready** | Limited | ✅ Yes |

---

## Monitoring Both GPUs

### Intel GPU Top

```bash
# Install Intel GPU tools
# Shows both GPUs separately

intel_gpu_top

# Output:
intel-gpu-top - 1234/ 1234 MHz;    0% RC6;  12.34/24.00 W;
     IMC reads:     1234 MiB/s
    IMC writes:     1234 MiB/s

      ENGINE      BUSY                                        
     Render/3D    95.00% |████████████████████████████████| (GPU 0)
     Render/3D    87.00% |███████████████████████████████ | (GPU 1)
         Video     0.00% |                                |
```

### Task Manager (Windows 11)

- Open Task Manager
- Performance Tab
- You'll see **two separate GPU entries**:
  - GPU 0 - Intel Arc B580
  - GPU 1 - Intel Arc B580
- Monitor VRAM usage separately for each

### Python Monitoring Script

```python
# gpu_monitor.py
from openvino.runtime import Core
import time

ie = Core()

while True:
    for device in ['GPU.0', 'GPU.1']:
        try:
            props = ie.get_property(device, 'FULL_DEVICE_NAME')
            print(f"{device}: {props}")
        except:
            print(f"{device}: Not available")
    print("-" * 40)
    time.sleep(2)
```

---

## Recommended Setup for Lambda Project

### 🎯 Optimal Configuration

```
┌─────────────────────────────────────────────────────────┐
│              Your Dual B580 Setup                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  GPU 0 (Arc B580 #1) - 12GB:                            │
│    └─ Qwen 2.5 32B Instruct (FP16)                      │
│       └─ General code generation                         │
│       └─ Architecture planning                           │
│       └─ Complex reasoning                               │
│       └─ Port 1234                                       │
│                                                          │
│  GPU 1 (Arc B580 #2) - 12GB:                            │
│    └─ Qwen 2.5 Coder 14B (Q5)                           │
│       └─ Fast code iterations                            │
│       └─ Quick debugging                                 │
│       └─ Real-time suggestions                           │
│       └─ Port 1235                                       │
│                                                          │
│  NPU (AI Boost):                                         │
│    └─ BGE-Small embeddings                              │
│       └─ Background file indexing                        │
│       └─ RAG context retrieval                           │
│                                                          │
│  CPU (20 cores):                                         │
│    └─ RAG server (port 8000)                            │
│    └─ File watchers                                      │
│    └─ System tasks                                       │
│                                                          │
└─────────────────────────────────────────────────────────┘

Workflow:
1. Quick question → GPU 1 (Fast 14B model)
2. Complex task → GPU 0 (Powerful 32B model)  
3. Background indexing → NPU (Always active)
4. File changes → Auto-reindex (NPU)
```

### Why This Works

1. **Two-tier model system**: Fast model for iterations, powerful model for complex work
2. **No resource conflicts**: Each component on separate hardware
3. **Always-on indexing**: NPU handles RAG without impacting GPUs
4. **Maximum flexibility**: Can run both models simultaneously or switch to single 72B model

---

## Alternative: Single Model Mode (Maximum Power)

If you want ONE incredibly powerful model instead of two:

```
┌─────────────────────────────────────────────────────────┐
│         Single Model - Maximum Performance               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  GPU 0 + GPU 1 (24GB combined):                         │
│    └─ Qwen 2.5 72B Instruct (Q4)                        │
│       └─ Split across both GPUs                          │
│       └─ Near GPT-4 level performance                    │
│       └─ 15-25 tokens/sec                                │
│       └─ 32K context window                              │
│                                                          │
│  NPU: Embeddings                                         │
│  CPU: RAG Server                                         │
│                                                          │
└─────────────────────────────────────────────────────────┘

Best for:
- Maximum quality on every query
- Complex architectural decisions  
- Production-grade code generation
- When speed is less important than quality
```

---

## Quick Start Commands (Dual GPU)

### Build Index on GPU 1 (Leave GPU 0 for LLM)
```bash
python -c "
from lambda_indexer import LambdaProjectIndexer
indexer = LambdaProjectIndexer(device='GPU.1')
indexer.index_project('./')
indexer.save_index('lambda_index')
print('✓ Index built on GPU 1')
"
```

### Start Dual Model Setup
```bash
# Terminal 1: LM Studio on GPU 0
# Load Qwen 32B, start server on port 1234

# Terminal 2: LM Studio on GPU 1  
# Load Qwen 14B Coder, start server on port 1235

# Terminal 3: RAG Server
python rag_server.py

# Terminal 4: Auto-indexer (NPU)
python auto_index.py
```

### Verify Both GPUs Working
```bash
# Monitor in real-time
intel_gpu_top

# Should show both GPUs with activity when querying
```

---

## Storage Considerations

With 24GB VRAM, you can now store larger models locally:

```bash
# Disk space needed for recommended models:
Qwen 2.5 14B Coder (Q5):    ~10 GB
Qwen 2.5 32B (FP16):        ~64 GB
Qwen 2.5 72B (Q4):          ~40 GB
DeepSeek Coder V2 236B (Q2): ~120 GB

# Recommended: 200GB free space for model storage
```

---

## What Changed From Single B580?

### Previous Limits (12GB):
- ❌ Couldn't run 32B+ models
- ❌ Limited context windows
- ❌ One model at a time only
- ❌ Aggressive quantization required

### New Capabilities (24GB):
- ✅ Run 72B models (Q4)
- ✅ Run 32B unquantized (full quality)
- ✅ Run TWO models simultaneously
- ✅ 32K+ context windows
- ✅ Production-grade performance
- ✅ Less quantization = better quality

---

## My Updated Recommendation

### Start Here:
```
Model: Qwen 2.5 32B Instruct (FP16)
Setup: Split across both GPUs
VRAM: ~18GB
Speed: 30-40 tokens/sec
Quality: Exceptional, unquantized

This gives you:
- Near-GPT-4 level quality
- Fast enough for interactive use  
- Leaves room for large context
- Best balance for Lambda project
```

### When You Need Maximum Quality:
```
Model: Qwen 2.5 72B Instruct (Q4)
Setup: Split across both GPUs
VRAM: ~22GB
Speed: 15-25 tokens/sec
Quality: Best available locally
```

### When You Need Maximum Speed:
```
GPU 0: Qwen 2.5 14B Coder (Q5) - Port 1234
GPU 1: Available for other tasks
Speed: 35-50 tokens/sec
Quality: Still very good
```

---

**Bottom line**: With dual B580s, you've jumped from "hobbyist" tier to "professional" tier local AI. You can now run models that compete with commercial APIs! 🚀
