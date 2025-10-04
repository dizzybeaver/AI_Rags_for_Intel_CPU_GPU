# Hybrid Inference Guide - Core Ultra 7 265K (NPU + CPU + GPU)

## Your Hardware Breakdown

### Intel Core Ultra 7 265K Features:
- **20 CPU Cores** (8 P-cores + 12 E-cores)
- **Intel Arc iGPU** (Xe-LPG graphics with 4 Xe-cores)
- **NPU** (AI Boost - 13 TOPS for AI workloads)
- **+ ARC B580** (Discrete GPU - 12GB VRAM)

**Total Compute**: 4 distinct AI-capable processors

---

## Can LM Studio Use All Simultaneously?

### Current LM Studio Limitations:
❌ **LM Studio GUI does NOT support true multi-device hybrid inference**
- You can only select ONE primary compute device at a time
- No built-in NPU support in current versions
- No automatic workload splitting across devices

### However, You CAN Use Them Through:
✅ **OpenVINO Runtime (Advanced Setup)**
✅ **llama.cpp with SYCL backend (Experimental)**
✅ **Intel AI PC Acceleration Toolkit**

---

## Method 1: OpenVINO Heterogeneous Plugin (RECOMMENDED)

### What This Enables:
- Use NPU for lightweight embedding/classification tasks
- Use discrete GPU (B580) for main LLM inference
- Use CPU as fallback/supplement
- Automatic workload distribution

### Setup Steps:

#### 1. Install OpenVINO Runtime
```bash
# Windows (PowerShell as Administrator)
pip install openvino openvino-dev

# Verify installation
python -c "from openvino.runtime import Core; print(Core().available_devices)"
# Should show: ['CPU', 'GPU.0', 'GPU.1', 'NPU']
```

#### 2. Install OpenVINO Model Server (Optional but Powerful)
```bash
pip install openvino-model-server
```

#### 3. Configure Heterogeneous Execution
Create `hetero_config.json`:
```json
{
  "device_priorities": ["GPU.1", "NPU", "GPU.0", "CPU"],
  "performance_hint": "LATENCY",
  "enable_profiling": true
}
```

**Device Mapping**:
- `GPU.1` = ARC B580 (discrete)
- `GPU.0` = Arc iGPU (integrated)
- `NPU` = AI Boost NPU
- `CPU` = All CPU cores

#### 4. Use with Python OpenVINO API
```python
from openvino.runtime import Core

ie = Core()

# Configure heterogeneous device
hetero_device = "HETERO:GPU.1,NPU,CPU"

# Load model with device priority
compiled_model = ie.compile_model(
    model=your_model,
    device_name=hetero_device,
    config={
        "PERFORMANCE_HINT": "LATENCY",
        "MULTI_DEVICE_PRIORITIES": "GPU.1,NPU,CPU"
    }
)
```

---

## Method 2: LM Studio + Separate NPU Workload

### Strategy: Division of Labor

#### Primary LLM (LM Studio on ARC B580):
- **Qwen 2.5 Coder 14B** running on discrete GPU
- Handles main code generation and reasoning

#### Embeddings/RAG (NPU):
- Run lightweight embedding model on NPU
- Handle document retrieval and context search
- Use for project file analysis

### Setup:

#### 1. LM Studio for Main LLM
```
Settings → GPU Acceleration → Intel GPU (OpenVINO) → GPU.1 (ARC B580)
Load: Qwen 2.5 Coder 14B
```

#### 2. Separate NPU Service for Embeddings
```bash
# Install optimum-intel for NPU support
pip install optimum-intel openvino-dev

# Run embedding model on NPU
python
>>> from optimum.intel import OVModelForFeatureExtraction
>>> model = OVModelForFeatureExtraction.from_pretrained(
...     "BAAI/bge-small-en-v1.5",
...     export=True,
...     device="NPU"
... )
```

---

## Method 3: llama.cpp with SYCL (Experimental)

### Enables:
- Multi-GPU support (both Arc iGPU + B580)
- CPU offloading
- NPU support (limited, experimental)

### Build llama.cpp with Intel Support:
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with SYCL for Intel GPUs
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release

# Run with multiple Intel GPUs
./build/bin/llama-cli \
  -m models/qwen2.5-coder-14b-q5.gguf \
  -ngl 99 \
  --split-mode row \
  -sm layer \
  -ts 0,1  # GPU 0 (iGPU) and GPU 1 (B580)
```

### To Use NPU with llama.cpp:
```bash
# Currently experimental - requires special build
export GGML_SYCL_DEVICE=0,1,NPU
./llama-cli -m model.gguf --device NPU,GPU
```

**Note**: NPU support in llama.cpp is still experimental as of early 2025.

---

## Practical Hybrid Setup Recommendation

### Configuration 1: Maximum Performance
```
ARC B580 (12GB): Main LLM inference (Qwen 2.5 Coder 14B)
NPU: Document embeddings + RAG retrieval
Arc iGPU: Parallel tasks or secondary model
CPU: System tasks + fallback
```

### Configuration 2: Development Workflow
```
ARC B580: Code generation model (LM Studio)
NPU: Continuous project file embedding/indexing
CPU: Running local embedding server
iGPU: Available for other tasks
```

### Configuration 3: Cost-Optimized
```
ARC B580: When you need maximum quality
NPU: Lightweight tasks (embeddings, classification)
CPU + iGPU: Medium-sized models when GPU is busy
```

---

## Tools That Support Multi-Device

### ✅ Native Intel Multi-Device Support:
1. **OpenVINO Toolkit** - Best for Intel hardware
2. **Intel Extension for PyTorch** - Good for training/fine-tuning
3. **OpenVINO Model Server** - Production inference server

### ⚠️ Limited Multi-Device Support:
1. **LM Studio** - Single device at a time (but you can run multiple instances)
2. **Ollama** - Single device, but lightweight
3. **llama.cpp** - Experimental multi-device

### ❌ No Multi-Device Support:
1. **text-generation-webui** - Single device
2. **koboldcpp** - Single device

---

## Performance Expectations

### Single LLM on ARC B580:
- **Qwen 2.5 Coder 14B**: 25-35 tokens/sec
- **Full GPU utilization**: ~10-11GB VRAM

### Hybrid Setup (B580 + NPU):
- **LLM on B580**: 25-35 tokens/sec (unchanged)
- **Embeddings on NPU**: 100-500 docs/sec
- **Total throughput**: 1.5-2x for RAG workloads

### NPU Alone (Lightweight Models):
- **Embedding generation**: Very fast (optimized for this)
- **Small LLMs (<1B params)**: Possible but slow
- **Best for**: Classification, embeddings, small transformer tasks

---

## Quick Start: Hybrid Setup

### Step 1: Install OpenVINO
```bash
pip install openvino openvino-dev optimum-intel
```

### Step 2: Configure LM Studio for B580
```
LM Studio → Settings → GPU → Select "Intel GPU (OpenVINO)" → GPU.1
```

### Step 3: Set Up NPU Embedding Service
```python
# save as embedding_server.py
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load on NPU
model = OVModelForFeatureExtraction.from_pretrained(
    "BAAI/bge-small-en-v1.5",
    export=True,
    device="NPU"
)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

@app.route('/embed', methods=['POST'])
def embed():
    text = request.json['text']
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return jsonify(embeddings.tolist())

if __name__ == '__main__':
    app.run(port=8001)
```

### Step 4: Use Both Together
```python
# Query LLM on B580 via LM Studio API
import requests

# Get embedding from NPU
embedding = requests.post('http://localhost:8001/embed', 
                         json={'text': 'your query'}).json()

# Use for RAG with LLM on B580
llm_response = requests.post('http://localhost:1234/v1/chat/completions',
                            json={
                                'messages': [...],
                                'context': embedding
                            })
```

---

## Monitoring Multi-Device Usage

### Intel GPU Tools:
```bash
# Install Intel GPU tools
# Download from: https://www.intel.com/content/www/us/en/download/785597/

# Monitor GPU usage
intel_gpu_top

# Check all device utilization
python
>>> from openvino.runtime import Core
>>> core = Core()
>>> for device in core.available_devices:
...     print(f"{device}: {core.get_property(device, 'FULL_DEVICE_NAME')}")
```

### Task Manager (Windows 11):
- Performance tab now shows NPU utilization
- Separate graphs for iGPU and discrete GPU
- Check "AI Acceleration" for NPU usage

---

## Limitations & Caveats

### NPU Limitations:
- ❌ Cannot run full 7B+ LLMs efficiently
- ❌ Limited model format support (INT8 primarily)
- ✅ Excellent for embeddings (< 1B params)
- ✅ Great for continuous background tasks

### Multi-Device Challenges:
- Memory copying overhead between devices
- Some speedup lost to coordination
- Best gains with truly parallel workloads (LLM + RAG)

### LM Studio Specific:
- No native multi-device in GUI
- Need to run separate processes for true parallelism
- Or use OpenVINO directly with Python

---

## My Recommendation for Lambda Project

### Optimal Setup:
```
ARC B580 (LM Studio):
  └─ Qwen 2.5 Coder 14B
     └─ Main code generation, debugging, refactoring

NPU (Python Service):
  └─ BGE-Small embedding model
     └─ Index all project files
     └─ Real-time code similarity search
     └─ RAG context retrieval

CPU/iGPU:
  └─ Available for other tasks
  └─ Run tests, build processes
```

**Why This Works**:
- LLM gets full B580 power (no sharing)
- NPU handles continuous indexing without blocking
- Can search project context while LLM generates code
- Each device optimized for its task

---

## Next Steps

1. **Install OpenVINO**: `pip install openvino openvino-dev`
2. **Verify devices**: Run device detection script
3. **Set up LM Studio on B580**: Configure GPU.1 in settings
4. **Optional**: Set up NPU embedding service for project files
5. **Test**: Run LLM and embedding tasks simultaneously

This setup will give you the best performance from your Core Ultra 7 265K + ARC B580 combo!
