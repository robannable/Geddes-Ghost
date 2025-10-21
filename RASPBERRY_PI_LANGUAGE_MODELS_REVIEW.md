# Language Models Review for Raspberry Pi Deployment
**GeddesGhost Project - Patrick Geddes AI Chatbot**
**Date:** October 21, 2025
**Author:** Claude Code Analysis

## Executive Summary

This document reviews the latest language models available for running the GeddesGhost application on Raspberry Pi hardware. The current application uses Anthropic's Claude API or Ollama with local models. For Raspberry Pi deployment, we focus on lightweight models compatible with limited hardware resources.

---

## Current System Configuration

**Existing Models:**
- **Primary:** Anthropic Claude Sonnet 4 (`claude-sonnet-4-20250514`) - API-based
- **Alternative:** Ollama with `cogito:latest` model

**System Requirements:**
- TF-IDF-based RAG system with document retrieval
- Temperature-based cognitive mode switching (0.7-0.9)
- Max tokens: 4000
- Python dependencies: streamlit, scikit-learn, langchain, numpy, pandas

---

## Hardware Constraints

### Raspberry Pi Models
- **Raspberry Pi 4 (8GB RAM):** Can run models up to 7B parameters (quantized)
- **Raspberry Pi 5 (8GB RAM):** Improved performance, same size constraints
- **CPU Bottleneck:** All methods push CPU to ~100% during generation

### Critical Limitations
- Maximum practical model size: **7B parameters** (quantized)
- Models larger than 8GB cannot be loaded
- CPU is the primary bottleneck, not GPU
- Quantized GGUF format required for optimal performance

---

## Recommended Language Models for Raspberry Pi

### Tier 1: Best Performance (Highly Recommended)

#### 1. Gemma 2 (2B) / Gemma3 (1B)
**Model:** `gemma2:2b` or `gemma3:1b`

**Specifications:**
- Size: 1.4GB (Gemma3:1b) / 2GB (Gemma2:2b)
- Parameters: 1B-2B
- Developer: Google

**Performance:**
- Highest token throughput
- Lowest resource usage
- ~8-10 tokens/second on Pi 5

**Pros:**
- Excellent balance of speed and quality
- Very responsive for interactive chatbot use
- Low RAM footprint leaves room for your RAG system
- Good reasoning capabilities for its size

**Cons:**
- May struggle with very complex historical or philosophical reasoning
- Limited context window compared to larger models

**Recommended for:** Primary deployment option for responsive performance

**Ollama Installation:**
```bash
ollama pull gemma2:2b
# or
ollama pull gemma3:1b
```

---

#### 2. Microsoft Phi-3 (3.8B)
**Model:** `phi3:latest` or `phi3:3.8b`

**Specifications:**
- Size: 2.3GB
- Parameters: 3.8B
- Developer: Microsoft

**Performance:**
- ~4 tokens/second on Pi 4
- ~6-8 tokens/second on Pi 5
- Optimized for small hardware

**Pros:**
- Excellent quality-to-size ratio
- Strong reasoning and instruction-following
- Well-suited for educational/conversational use
- Good for nuanced historical discussions

**Cons:**
- Slightly slower than Gemma models
- Higher RAM usage (but still manageable)

**Recommended for:** Best balance of quality and performance for your use case

**Ollama Installation:**
```bash
ollama pull phi3:latest
```

---

#### 3. Qwen2.5 (0.5B-3B)
**Model:** `qwen2.5:0.5b` or `qwen2.5:3b`

**Specifications:**
- Size: 398MB (0.5B) / 1.9GB (3B)
- Parameters: 0.5B-3B
- Developer: Alibaba Cloud

**Performance:**
- Extremely fast on Pi hardware
- 10+ tokens/second for 0.5B version
- Good multilingual support

**Pros:**
- Ultra-lightweight (0.5B version)
- Very fast response times
- Recent model with modern architecture
- Good for testing and development

**Cons:**
- 0.5B may be too simple for complex philosophical discussions
- Less well-known than Phi or Gemma

**Recommended for:** Development/testing or ultra-fast responses

**Ollama Installation:**
```bash
ollama pull qwen2.5:0.5b
# or
ollama pull qwen2.5:3b
```

---

### Tier 2: Good Performance (Suitable Alternatives)

#### 4. TinyLlama (1.1B)
**Model:** `tinyllama:latest`

**Specifications:**
- Size: 637MB
- Parameters: 1.1B
- Developer: Community

**Performance:**
- Very fast inference
- Low memory footprint

**Pros:**
- Extremely lightweight
- Good for basic conversational AI
- Fast response times

**Cons:**
- Limited reasoning capabilities
- May struggle with historical context
- Basic language generation

**Recommended for:** Fallback option or resource-constrained scenarios

**Ollama Installation:**
```bash
ollama pull tinyllama:latest
```

---

#### 5. Deepseek-R1 Distilled (1.5B-7B)
**Model:** `deepseek-r1:1.5b` or `deepseek-r1:7b`

**Specifications:**
- Size: 1GB (1.5B) / 4.7GB (7B)
- Parameters: 1.5B-7B (distilled)
- Developer: Deepseek AI

**Performance:**
- Variable based on size
- 7B version may be slow on Pi 4

**Pros:**
- Strong reasoning capabilities
- Good for complex questions
- Multiple size options

**Cons:**
- 7B version may be too slow for interactive use
- Higher memory requirements for larger variants

**Recommended for:** If you need stronger reasoning and can accept slower responses

**Ollama Installation:**
```bash
ollama pull deepseek-r1:1.5b
# or
ollama pull deepseek-r1:7b
```

---

#### 6. BitNet b1.58 (2B)
**Model:** `bitnet:2b` (if available)

**Specifications:**
- Size: Very efficient (binary/ternary weights)
- Parameters: 2B
- Developer: Microsoft Research

**Performance:**
- Exceptional memory efficiency
- 8+ tokens/second
- Smallest RAM footprint

**Pros:**
- Cutting-edge quantization
- Very memory efficient
- Good speed

**Cons:**
- May require specific setup
- Newer model, less tested
- Availability may vary

**Recommended for:** Experimental deployment

---

### Tier 3: Marginal Performance (Use with Caution)

#### 7. Llama 3.2 (3B)
**Model:** `llama3.2:3b`

**Specifications:**
- Size: 2GB
- Parameters: 3B
- Developer: Meta

**Performance:**
- Moderate speed on Pi 5
- May struggle on Pi 4

**Pros:**
- Meta's latest small model
- Good general capabilities

**Cons:**
- Slower than comparable models
- Higher resource usage relative to performance

**Ollama Installation:**
```bash
ollama pull llama3.2:3b
```

---

## Performance Framework Comparison

### Ollama vs llama.cpp

**Ollama:**
- Easier setup and model management
- Better for production deployment
- Slightly slower (10-20%) than llama.cpp
- Current system already configured for Ollama

**llama.cpp:**
- 10-20% faster inference
- More configuration options
- Requires more manual setup
- Better for performance-critical applications

**Recommendation:** Stick with Ollama for ease of use unless performance is critical.

---

## Recommended Configuration for GeddesGhost

### Primary Recommendation: Phi-3 (3.8B)

**Rationale:**
- Best balance for historical/philosophical conversation
- Good reasoning capabilities for Patrick Geddes persona
- Manageable speed on Pi 4/5
- Sufficient quality for educational context

**Configuration in `geddesghost.py`:**
```python
"ollama": {
    "provider": "ollama",
    "model": "phi3:3.8b",
    "max_tokens": 2000,  # Reduced for Pi performance
    "temperature": 0.7,
    "top_p": 0.9,
    "api_endpoint": "http://localhost:11434/api/generate",
    "headers": {
        "Content-Type": "application/json"
    }
}
```

### Fallback Recommendation: Gemma2 (2B)

For faster responses with slight quality trade-off:

```python
"ollama": {
    "provider": "ollama",
    "model": "gemma2:2b",
    "max_tokens": 2000,
    "temperature": 0.7,
    "top_p": 0.9,
    "api_endpoint": "http://localhost:11434/api/generate",
    "headers": {
        "Content-Type": "application/json"
    }
}
```

---

## Deployment Considerations

### Memory Management
1. **Reduce max_tokens:** Lower from 4000 to 2000-2500 to speed generation
2. **Optimize RAG chunks:** Limit to 3-5 top chunks instead of current 5
3. **Clear caches regularly:** Streamlit caching can consume RAM

### Performance Optimization
1. **Use quantized models:** Q4_K_M or Q5_K_M quantization
2. **Consider swap file:** Add 4GB swap on Pi for stability
3. **Monitor temperature:** Ensure adequate cooling for Pi under load

### Response Time Expectations
- **Gemma2 (2B):** 30-60 seconds for typical response
- **Phi-3 (3.8B):** 60-120 seconds for typical response
- **Qwen 0.5B:** 15-30 seconds for typical response

### User Experience
- Add streaming responses if possible (Ollama supports streaming)
- Set clear expectations about response time
- Consider adding "thinking" animations

---

## Installation Instructions

### 1. Install Ollama on Raspberry Pi

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull Recommended Models

```bash
# Primary recommendation
ollama pull phi3:3.8b

# Fast alternative
ollama pull gemma2:2b

# Ultra-light fallback
ollama pull qwen2.5:0.5b
```

### 3. Test Model Performance

```bash
# Test with sample prompt
ollama run phi3:3.8b "Who was Patrick Geddes?"
```

### 4. Update Configuration

Edit `geddesghost.py` line 236 to use your chosen model:
```python
"model": "phi3:3.8b",  # or "gemma2:2b", etc.
```

### 5. Optimize for Pi

Edit `geddesghost.py` line 237:
```python
"max_tokens": 2000,  # Reduced from 4000
```

---

## Testing Matrix

| Model | Size | Expected Speed | Quality | RAM Usage | Recommendation |
|-------|------|----------------|---------|-----------|----------------|
| Gemma3:1b | 1.4GB | 8-10 tok/s | Good | Low | Excellent |
| Gemma2:2b | 2GB | 8-10 tok/s | Very Good | Low-Med | Excellent |
| Phi-3:3.8b | 2.3GB | 4-6 tok/s | Excellent | Medium | **Best Choice** |
| Qwen2.5:0.5b | 398MB | 10+ tok/s | Fair | Very Low | Testing Only |
| Qwen2.5:3b | 1.9GB | 6-8 tok/s | Very Good | Low-Med | Very Good |
| TinyLlama | 637MB | 10+ tok/s | Fair | Very Low | Fallback |
| Deepseek-R1:1.5b | 1GB | 6-8 tok/s | Good | Low | Good |
| Deepseek-R1:7b | 4.7GB | 2-3 tok/s | Excellent | High | Too Slow |

---

## Benchmarking Script

Create a simple benchmark to test models on your hardware:

```python
import time
import requests
import json

def benchmark_model(model_name, prompt="Who was Patrick Geddes?", tokens=100):
    start = time.time()

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": tokens}
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload
    )

    end = time.time()
    duration = end - start

    if response.status_code == 200:
        data = response.json()
        tokens_generated = len(data.get('response', '').split())
        tokens_per_sec = tokens_generated / duration

        print(f"\nModel: {model_name}")
        print(f"Time: {duration:.2f}s")
        print(f"Tokens: {tokens_generated}")
        print(f"Speed: {tokens_per_sec:.2f} tok/s")
        print(f"Response: {data.get('response', '')[:100]}...")
    else:
        print(f"Error testing {model_name}: {response.status_code}")

# Test models
models = ["phi3:3.8b", "gemma2:2b", "qwen2.5:3b"]
for model in models:
    benchmark_model(model)
```

---

## Next Steps

1. **Install Ollama** on your Raspberry Pi
2. **Pull Phi-3** and Gemma2 models for testing
3. **Run benchmarks** to compare performance on your hardware
4. **Update configuration** in `geddesghost.py`
5. **Test RAG performance** with the new models
6. **Optimize settings** based on response quality/speed trade-offs

---

## Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Raspberry Pi AI Guide](https://pimylifeup.com/raspberry-pi-ollama/)
- [Model Cards](https://ollama.com/library)
- [llama.cpp Alternative](https://github.com/ggerganov/llama.cpp)

---

## Conclusion

For the GeddesGhost Patrick Geddes chatbot on Raspberry Pi:

**Best Overall Choice:** **Phi-3 (3.8B)**
- Optimal balance of quality and performance
- Well-suited for educational/historical conversations
- Proven track record on Pi hardware

**Fastest Option:** **Gemma2 (2B)**
- Best for responsive user experience
- Good quality with excellent speed
- Lower resource usage

**Budget/Testing:** **Qwen2.5 (0.5B-3B)**
- Ultra-lightweight for development
- Fast iteration during testing

All recommended models can run effectively on Raspberry Pi 4/5 with 8GB RAM and will provide a significantly better user experience than attempting to run larger models.
