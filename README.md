# tiny-code-helper

A lightweight local coding assistant for the NVIDIA Jetson (Orin Nano 8GB) using **llama.cpp** + **Gemma 4 E2B** with **MTP speculative decoding** — no cloud, no GPU swap, no Ollama required.

Runs entirely on-device alongside your desktop GUI. Benchmarked at **42.7 tok/s on code** (vs 6.4 tok/s through Ollama with a 7B model).

## Why This Exists

The Jetson's 8GB shared CPU/GPU memory is tight. Running Ollama + a 7B model + the desktop GUI pushes you to the edge of OOM. This setup uses:

- **Gemma 4 E2B (3B effective, 2.9GB Q4_K_M)** — small enough to coexist with the GUI
- **MTP speculative decoding** — 72% draft acceptance on code, 37% faster than baseline
- **llama.cpp direct** — ~2x faster than Ollama's wrapper for the same model

Result: a fast coding helper that fits in ~4.1 GB RAM with 4096-token context, leaving room for your desktop.

## Prerequisites

### Hardware
- NVIDIA Jetson Orin Nano 8GB (or equivalent ARM64 with CUDA)
- JetPack 6 (L4T R36.x), CUDA 8.7+

### Software
- **llama.cpp** built with CUDA support:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="87" \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

Binaries land at `build/bin/llama-server`.

### Models

Download two files into `~/models/`:

```bash
# Main model (2.9 GB)
wget -O ~/models/gemma-4-E2B-it-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"

# MTP draft model (94 MB) — enables speculative decoding
mkdir -p ~/models/gemma-4-e2b-mtp
wget -O ~/models/gemma-4-e2b-mtp/mtp-gemma-4-E2B-it.gguf \
  "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mtp-gemma-4-E2B-it.gguf"
```

## Installation

```bash
# Clone this repo
git clone https://github.com/drwjkirkpatrick-web/tiny-code-helper.git ~/projects/tiny-code-helper

# Install the script to ~/bin (creates it if needed)
mkdir -p ~/bin
cp ~/projects/tiny-code-helper/scripts/code-llm ~/bin/code-llm
chmod +x ~/bin/code-llm

# Ensure ~/bin is on your PATH (add to ~/.bashrc if not)
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Usage

### Start the server

```bash
code-llm start
```

Output:
```
Starting llama.cpp server (Gemma 4 E2B + MTP)...
  PID: 31512
  Waiting for server......... ready!
  Endpoint: http://127.0.0.1:8080/v1/chat/completions
  Context: 4096 tokens
```

The script automatically:
- Stops Ollama if it's running (frees GPU/RAM)
- Sets max Jetson clock speeds (`jetson_clocks`)
- Launches llama-server with MTP speculative decoding
- Waits for the health endpoint to respond

### Commands

| Command | What it does |
|---|---|
| `code-llm start` | Launch server (auto-stops Ollama, max clocks) |
| `code-llm stop` | Stop server cleanly |
| `code-llm restart` | Stop then start |
| `code-llm status` | Show PID, RAM usage, endpoint |
| `code-llm test` | Run a code-gen test, show output + throughput |
| `code-llm chat` | Interactive coding chat |
| `code-llm chat --think` | Chat with reasoning shown |
| `code-llm chat --tokens 1200` | Chat with custom max_tokens |
| `code-llm log` | Tail last 50 lines of server log |

### Environment Overrides

| Variable | Default | Description |
|---|---|---|
| `CTX` | `4096` | Context window size in tokens |
| `PORT` | `8080` | Server port |
| `HOST` | `127.0.0.1` | Bind address |
| `LLAMA_BIN` | `~/llama.cpp/build/bin/llama-server` | Path to llama-server binary |
| `MODEL` | `~/models/gemma-4-E2B-it-Q4_K_M.gguf` | Main model path |
| `MTP_MODEL` | `~/models/gemma-4-e2b-mtp/mtp-gemma-4-E2B-it.gguf` | MTP draft model path |

Example:
```bash
CTX=8192 code-llm start    # larger context for long source files
PORT=8081 code-llm start    # different port
```

### Using with External Tools

The server exposes an OpenAI-compatible API at `http://127.0.0.1:8080`. Point any compatible tool at it:

**curl:**
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"write a quicksort in Python"}]}'
```

**VS Code Continue, Aider, etc.:**
Set the API base URL to `http://127.0.0.1:8080/v1` and any model name (it's ignored).

## Important: Gemma 4 E2B is a Thinking Model

Gemma 4 E2B generates internal reasoning in `reasoning_content` **before** producing the final answer in `content`. This means:

- **`max_tokens` must be high enough** for both thinking + answer. Use 800+ for code tasks. With low limits (e.g. 200), you get empty `content` because the model hasn't finished thinking.
- **OpenAI-compatible clients** that only read `content` will see blank responses if `max_tokens` is too low.
- **To get the answer:** read `choices[0].message.content`
- **To see the reasoning:** read `choices[0].message.reasoning_content`

The `code-llm chat` and `code-llm test` commands handle this automatically.

## Memory Budget (Jetson 8GB)

| Component | RAM |
|---|---|
| OS + overhead | ~1.5 GB |
| Desktop GUI (GNOME/GDM) | ~0.5–1 GB |
| llama-server + Gemma 4 E2B + MTP (4096 ctx) | ~4.1 GB |
| **Total** | **~6.1 GB** |
| **Remaining** | **~1.2 GB available** |

Tips:
- Stop Ollama before starting (`code-llm start` does this automatically)
- For maximum RAM, stop the GUI: `sudo systemctl stop gdm3` (frees ~600 MB)
- Monitor usage: `code-llm status` shows RSS, or `tegrastats` for GPU details
- Reduce context if memory is tight: `CTX=2048 code-llm start`

## Benchmark

Measured August 2026 on Jetson Orin Nano 8GB with `llama-bench` and `llama-server`.

| Config | Code tok/s | Notes |
|---|---|---|
| Ollama + Mistral 7B Q4 | 6.4 | Baseline comparison |
| llama.cpp + Qwen 7B Q4 (no MTP) | 12.55 | 2x faster than Ollama |
| llama.cpp + Gemma 4 E2B (no MTP) | 31.07 | Small model, fast |
| **llama.cpp + Gemma 4 E2B + MTP n-max=4** | **42.7** | **Best for code** |

MTP speculative decoding achieves 72% draft acceptance on code (vs 19% on prose). Use it for code generation and structured output only — for general chat, MTP hurts throughput.

## Baby LLM Nanny Integration (Optional)

**baby-llm-nanny** is a hallucination and quality screening tool for small local LLMs. It sends a curated bank of test prompts to your model and evaluates the responses against known-correct answers — catching hallucinations, bad math, logic errors, and instruction-following failures.

The nanny was designed for Ollama's API, but `code-llm` uses llama.cpp. The included **`nanny_bridge.py`** translates between them:

```
baby-llm-nanny  →  Ollama API (/api/generate)  →  nanny_bridge.py  →  llama.cpp (/v1/chat/completions)
                    port 11435                                          port 8080
```

The bridge strips Gemma 4 E2B's `reasoning_content` (thinking tokens) and passes only the final `content` through to the nanny — so screening measures actual answer quality, not thinking length.

### Real-Time Deterministic Code Review

When `--review` is enabled (default when using `code-llm nanny`), the bridge doesn't just translate — it **intercepts every response** and deterministically verifies any code it finds:

1. **Detects code** — extracts Python from markdown fences (```python ... ```), generic fences, or bare `def`/`class` blocks
2. **Syntax-checks** — compiles with `py_compile`, catches SyntaxError with line numbers
3. **Execution-tests** — runs the code in an isolated subprocess to catch runtime errors
4. **Auto-fixes** — if code fails, feeds the specific error message back to the model and re-queries
5. **Repeats** — up to `--max-fix-attempts` (default 3) until code passes or attempts are exhausted
6. **Returns** only the final verified code to the caller

This means any tool talking to the bridge (baby-llm-nanny, curl, VS Code Continue, etc.) automatically gets syntax-verified code. The caller never sees a response with broken syntax — the bridge fixes it before returning.

**Two layers of review:**
- **Bridge layer** (real-time): syntax + execution check + auto-fix — runs on every request
- **Nanny layer** (when using `--review`): functional test cases + generate→test→fix loop — runs on curated prompts

The bridge catches *structural* errors (syntax, import failures, runtime crashes). The nanny catches *functional* errors (wrong output, wrong arguments, logic bugs). Together they provide deterministic verification that the code actually works.

**Stats endpoint:** `curl http://127.0.0.1:11435/bridge/stats` shows:
- Total requests, code detected, passed first try, fixed via retry
- HTML detected, HTML fixed (autocorrect count)
- Whether review is enabled and max fix attempts

### HTML Adaptive Code Review

For students learning HTML, the bridge includes an **HTML adaptive error database** that automatically detects and fixes common HTML errors in real time — no model re-query needed, instant deterministic patching.

**Test suite:** 25 HTML prompts (12 profile pages + 13 game patterns) with 258 deterministic checks covering:
- Document structure (DOCTYPE, html, head, body, title)
- Semantic tags (nav, header, footer, main, section)
- Tag balance (div, p, ul/li, section)
- Accessibility (img alt attributes)
- Responsive design (viewport meta, charset)
- Game requirements (canvas, getContext, clearRect, requestAnimationFrame, event listeners)
- CSS styling (color, background, fillStyle)

**Test results (Gemma 4 E2B, August 2026):** 247/258 checks passed (95.7%). The 10 real errors were:

| Error | Count | Fix |
|---|---|---|
| Missing CSS color styling | 3 | Inject `<style>` block with base styles |
| Missing `<nav>` semantic tag | 1 | Replace `<div>` containing 3+ links with `<nav>` |
| Missing document structure (fragment) | 5 | Wrap in full `<!DOCTYPE html>` document, promote `<h2>` to `<h1>` |
| Missing `clearRect()` in canvas game | 1 | Inject `ctx.clearRect()` into game loop function |
| Markdown code fences wrapping HTML | 4 | Strip ```` ```html ```` fences |

**9 error patterns in the database**, each with `detect()` and `fix()` functions. After autocorrect: **10/10 fixable errors resolved** (the 1 remaining was a test-runner storage truncation, not a model error).

**How it works in the bridge:**
1. Bridge detects HTML in the response (DOCTYPE, `<html>`, 3+ structural tags, or ```` ```html ```` fence)
2. Runs all 9 error pattern detectors
3. Applies fixes in order (fences stripped → structure wrapped → content patched)
4. Returns the fixed HTML to the caller — broken HTML never reaches the student

**Three review layers now:**
- **HTML autocorrect** (real-time): deterministic pattern detection + instant fix
- **Python review** (real-time): syntax check + execution test + model re-query
- **Nanny review** (curated prompts): functional test cases + generate→test→fix loop

**Run the HTML test suite:**
```bash
code-llm start
cd ~/projects/tiny-code-helper/scripts
python3 html_test_runner.py --output results.json --verbose
# Or by category:
python3 html_test_runner.py --category profile --verbose
python3 html_test_runner.py --category game --verbose
```

### Quick Start

```bash
# 1. Start the LLM server
code-llm start

# 2. Run the nanny (auto-starts the bridge)
code-llm nanny

# Or run with custom args:
code-llm nanny-run --review --max-iterations 5
code-llm nanny-run --categories math,reasoning
code-llm nanny-run --verbose
```

### Nanny Commands

| Command | What it does |
|---|---|
| `code-llm nanny` | Auto-start bridge + run `--review --max-iterations 3` |
| `code-llm nanny-start` | Start just the Ollama-to-llama.cpp bridge |
| `code-llm nanny-stop` | Stop the bridge |
| `code-llm nanny-status` | Show bridge running state |
| `code-llm nanny-run ARGS` | Run nanny with custom args (passes ARGS to baby-llm-nanny) |

`code-llm stop` also stops the nanny bridge. `code-llm status` shows both.

### What the Nanny Tests

- **Live code review:** 9 built-in coding prompts (is-even, factorial, reverse-string, fizzbuzz, max-of-list, palindrome, count-vowels, binary-search, merge-sorted-lists). The nanny generates code, runs it against test cases in an isolated subprocess, and if tests fail, feeds specific error feedback back to the model for a retry — up to `--max-iterations` rounds.
- **Hallucination screening:** Factual questions with known-correct answers to catch made-up facts.
- **Math and logic:** Arithmetic and reasoning problems.
- **Instruction following:** Does the model follow output format instructions?

### Sample Results (August 2026, Gemma 4 E2B via llama.cpp)

```
🔬 Live Code Review Report — gemma-4-e2b
  Prompts reviewed: 9
  Final pass rate:  8/9
  Self-corrected:   2 (went from failing → passing)

  ✅ coding-is-even [1 iterations]        7.4s
  ✅ coding-factorial [1 iterations]     15.2s
  ✅ coding-reverse-string [1 iterations] 9.9s
  ✅ coding-fizzbuzz [1 iterations]      18.8s
  ✅ coding-max-of-list [2 iterations]   60.6s (self-corrected!)
  ✅ coding-palindrome [1 iterations]    23.7s
  ✅ coding-count-vowels [1 iterations]  19.3s
  ✅ coding-binary-search [2 iterations] 62.4s (self-corrected!)
  ❌ coding-merge-sorted-lists [2 iterations] — failed after 2 rounds
```

### Prerequisites

- **baby-llm-nanny** installed at `~/projects/baby-llm-nanny/` ([repo](https://github.com/drwjkirkpatrick-web/baby-llm-nanny))
- The LLM server must be running (`code-llm start`) before starting the nanny

### Files

| File | Purpose |
|---|---|
| `scripts/code-llm` | Main helper script (start/stop/status/chat/nanny) |
| `scripts/nanny_bridge.py` | Ollama-to-llama.cpp API bridge with Python + HTML review |
| `scripts/html_prompts.py` | 25 HTML test prompts (12 profile + 13 game) with deterministic checks |
| `scripts/html_test_runner.py` | HTML test runner — sends prompts to model, runs checks, records errors |
| `scripts/html_error_db.py` | Adaptive HTML error database — 9 detect/fix patterns for autocorrect |
| `scripts/html_test_results.json` | Baseline test results (25 prompts, 247/258 passed) |

## License

MIT