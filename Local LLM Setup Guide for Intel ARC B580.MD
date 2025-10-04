# Local LLM Setup Guide for Intel ARC B580 (12GB VRAM)

## Quick Start: Enabling OpenVINO in LM Studio

### Step 1: Download LM Studio
1. Visit https://lmstudio.ai/
2. Download the latest version for your OS
3. Install and launch LM Studio

### Step 2: Enable Intel GPU Acceleration
1. Open LM Studio
2. Click **Settings** (gear icon in bottom left)
3. Navigate to **Inference** tab
4. Under **GPU Acceleration**:
   - Enable **GPU Offloading**
   - Select **Intel GPU (OpenVINO)** from the dropdown
   - Set **GPU Layers** to **Max** or **999** (offload everything)
5. Click **Save**

### Step 3: Verify OpenVINO is Active
- After loading a model, check the console output
- Look for: `Using OpenVINO backend` or `Intel GPU detected`
- You should see GPU memory usage in the model info panel

---

## Recommended Models for Coding (Arc B580 - 12GB VRAM)

### Tier 1: Best Balance (Recommended)

#### **Qwen 2.5 Coder 14B (Q5_K_M)**
- **Model ID**: `qwen2.5-coder-14b-instruct-q5_k_m.gguf`
- **VRAM Usage**: ~10GB
- **Why**: Best coding model for your hardware. Excellent at:
  - Following complex instructions
  - Multi-file codebases
  - Pattern recognition and reuse
  - Context retention (128K tokens)
- **Download in LM Studio**: Search "Qwen 2.5 Coder 14B"
- **Performance**: Fast inference on Intel XMX cores

#### **DeepSeek Coder V2 Lite 16B (Q4_K_M)**
- **Model ID**: `deepseek-coder-v2-lite-instruct-16b-q4_k_m.gguf`
- **VRAM Usage**: ~9GB
- **Why**: Specialized for code generation
  - Strong at debugging and refactoring
  - Good architectural understanding
  - Fill-in-middle support
- **Download**: Search "DeepSeek Coder V2 Lite"
- **Performance**: Excellent code quality, slightly slower than Qwen

---

### Tier 2: Fast & Efficient

#### **Qwen 2.5 Coder 7B (Unquantized)**
- **Model ID**: `qwen2.5-coder-7b-instruct-fp16.gguf`
- **VRAM Usage**: ~7GB
- **Why**: 
  - Fastest inference speed
  - Still highly capable for coding
  - Great for iterative development
- **Download**: Search "Qwen 2.5 Coder 7B"
- **Use Case**: Quick edits, rapid prototyping

#### **Llama 3.1 8B Instruct (Q6_K)**
- **Model ID**: `llama-3.1-8b-instruct-q6_k.gguf`
- **VRAM Usage**: ~7GB
- **Why**:
  - Strong general reasoning
  - Good instruction following
  - Reliable for standard coding tasks
- **Download**: Search "Llama 3.1 8B"
- **Use Case**: General purpose coding assistant

---

### Tier 3: Maximum Capability (If You Need More Power)

#### **Qwen 2.5 32B Instruct (Q4_K_M)**
- **Model ID**: `qwen2.5-32b-instruct-q4_k_m.gguf`
- **VRAM Usage**: ~11.5GB (tight fit)
- **Why**:
  - Highest reasoning capability that fits
  - Better architectural decisions
  - Complex problem solving
- **Download**: Search "Qwen 2.5 32B"
- **Warning**: May need to reduce context length to fit
- **Use Case**: Complex architectural decisions, large refactors

---

## Recommended Settings for Coding Tasks

### In LM Studio Inference Settings:
```
Temperature: 0.1-0.3 (lower for more deterministic code)
Top P: 0.9
Max Tokens: 4096-8192 (for code generation)
Context Length: 8192-16384 (Qwen supports up to 128K)
GPU Layers: 999 (offload everything)
```

### For Best Claude Compatibility:
These models generate code that Claude can work with effectively:
1. **Qwen 2.5 Coder 14B** - Follows architectural patterns well
2. **DeepSeek Coder V2** - Clean, well-structured code
3. **Qwen 2.5 Coder 7B** - Fast iterations with good quality

---

## Performance Expectations (Intel ARC B580)

| Model | Tokens/sec (Est.) | VRAM Usage | Best For |
|-------|-------------------|------------|----------|
| Qwen 2.5 Coder 7B | 40-60 | 7GB | Fast iterations |
| Qwen 2.5 Coder 14B | 25-35 | 10GB | **Best balance** |
| DeepSeek Coder V2 16B | 20-30 | 9GB | Code quality |
| Qwen 2.5 32B | 15-20 | 11.5GB | Complex tasks |

*Estimates based on OpenVINO with INT8 optimizations on Arc series GPUs*

---

## Troubleshooting

### OpenVINO Not Working?
1. Update to latest LM Studio version
2. Update Intel GPU drivers: https://www.intel.com/content/www/us/en/download/785597/
3. Check if model format is compatible (GGUF works best)
4. Try reducing GPU layers if model won't load

### Out of Memory?
1. Use more aggressive quantization (Q4 instead of Q5/Q6)
2. Reduce context length in settings
3. Close other GPU applications
4. Use smaller model variant

### Slow Performance?
1. Verify GPU is actually being used (check LM Studio console)
2. Enable **Metal** or **OpenVINO** acceleration explicitly
3. Update drivers
4. Try different quantization (sometimes Q4 is faster than Q5)

---

## My Top Recommendation

**Start with: Qwen 2.5 Coder 14B (Q5_K_M)**

Why:
- Perfect fit for 12GB VRAM
- Best coding performance in this size class
- Excellent at following the Lambda project's architectural patterns
- Fast enough for interactive use
- Well-tested with OpenVINO

You can always drop to 7B for speed or try 32B for complex architectural work.

---

## Additional Resources

- **LM Studio Discord**: https://discord.gg/aPQfnNkxGC
- **Qwen Models**: https://huggingface.co/Qwen
- **DeepSeek Models**: https://huggingface.co/deepseek-ai
- **Intel Arc GPU Optimization**: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html

---

*Generated for Lambda Execution Engine project development on Intel ARC B580*
