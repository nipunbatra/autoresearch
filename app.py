"""
Simple web app for NLQ -> Python code generation.
Uses MLX model checkpoint for inference on Apple Silicon.

Usage:
    pip install flask
    python app.py

Then open http://localhost:5000
"""

import os
import json
import pickle

import mlx.core as mx
import mlx.nn as nn

from train_mlx import GPT, GPTConfig, generate_text

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoint")
TOKENIZER_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "tokenizer")


def load_model():
    """Load trained MLX model from checkpoint."""
    config_path = os.path.join(CHECKPOINT_DIR, "config.json")
    weights_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_DIR}. Train the model first!\n"
            "  python download_code_data.py\n"
            "  python prepare.py --num-shards 5\n"
            "  python train_mlx.py"
        )

    with open(config_path) as f:
        meta = json.load(f)

    config = GPTConfig(**meta["config"])
    model = GPT(config)

    # Load weights
    model.load_weights(weights_path)
    mx.eval(model.parameters())

    return model, config


def load_tokenizer():
    """Load the trained tokenizer."""
    path = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No tokenizer at {path}. Run prepare.py first!")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Flask Web App
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLQ to Python Code</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
            background: #0d1117;
            color: #c9d1d9;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
        }
        h1 {
            color: #58a6ff;
            margin-bottom: 0.5rem;
            font-size: 1.5rem;
        }
        .subtitle {
            color: #8b949e;
            margin-bottom: 2rem;
            font-size: 0.85rem;
        }
        .container {
            width: 100%;
            max-width: 700px;
        }
        .input-group {
            margin-bottom: 1.5rem;
        }
        label {
            display: block;
            color: #8b949e;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
        }
        textarea {
            width: 100%;
            padding: 1rem;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-family: inherit;
            font-size: 0.9rem;
            resize: vertical;
            min-height: 80px;
        }
        textarea:focus {
            outline: none;
            border-color: #58a6ff;
        }
        .controls {
            display: flex;
            gap: 1rem;
            align-items: center;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }
        .control-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .control-item label {
            margin: 0;
            white-space: nowrap;
        }
        input[type="number"] {
            padding: 0.4rem;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 4px;
            color: #c9d1d9;
            font-family: inherit;
            width: 70px;
        }
        button {
            padding: 0.7rem 2rem;
            background: #238636;
            color: white;
            border: none;
            border-radius: 6px;
            font-family: inherit;
            font-size: 0.9rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover { background: #2ea043; }
        button:disabled {
            background: #21262d;
            color: #484f58;
            cursor: not-allowed;
        }
        .output {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 1rem;
            min-height: 120px;
            white-space: pre-wrap;
            font-size: 0.9rem;
            line-height: 1.5;
            position: relative;
        }
        .output .placeholder {
            color: #484f58;
            font-style: italic;
        }
        .copy-btn {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            padding: 0.3rem 0.6rem;
            background: #30363d;
            font-size: 0.75rem;
        }
        .copy-btn:hover { background: #484f58; }
        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #30363d;
            border-top-color: #58a6ff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 0.5rem;
            vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .model-info {
            margin-top: 2rem;
            padding: 0.8rem;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            font-size: 0.75rem;
            color: #8b949e;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>NLQ -> Python Code</h1>
        <p class="subtitle">~10M parameter GPT model trained on code instruction data (MLX)</p>

        <div class="input-group">
            <label>Describe what you want in natural language:</label>
            <textarea id="prompt" placeholder="e.g., Write a function to check if a number is prime">Write a function to reverse a string</textarea>
        </div>

        <div class="controls">
            <div class="control-item">
                <label>Temperature:</label>
                <input type="number" id="temperature" value="0.7" min="0" max="2" step="0.1">
            </div>
            <div class="control-item">
                <label>Max tokens:</label>
                <input type="number" id="max_tokens" value="256" min="32" max="512" step="32">
            </div>
            <button id="generate-btn" onclick="doGenerate()">Generate Code</button>
        </div>

        <label>Generated Python code:</label>
        <div class="output" id="output">
            <span class="placeholder">Generated code will appear here...</span>
        </div>

        <div style="margin-top: 0.8rem; display: flex; gap: 0.5rem;">
            <button id="run-btn" onclick="runCode()" style="background: #1f6feb; padding: 0.5rem 1.2rem; font-size: 0.8rem;">Run Code</button>
        </div>

        <div id="exec-output" class="output" style="margin-top: 0.8rem; min-height: 60px; display: none; border-color: #1f6feb;">
            <span class="placeholder">Output will appear here...</span>
        </div>

        <div class="model-info" id="model-info">Loading model info...</div>
    </div>

    <script>
        async function doGenerate() {
            const btn = document.getElementById('generate-btn');
            const output = document.getElementById('output');
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) return;

            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span>Generating...';
            output.innerHTML = '<span class="placeholder">Generating...</span>';

            try {
                const resp = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: prompt,
                        temperature: parseFloat(document.getElementById('temperature').value),
                        max_tokens: parseInt(document.getElementById('max_tokens').value),
                    })
                });
                const data = await resp.json();
                if (data.code) {
                    output.innerHTML = escapeHtml(data.code) +
                        '<button class="copy-btn" onclick="copyCode()">Copy</button>';
                } else {
                    output.innerHTML = '<span class="placeholder">Error: ' + (data.error || 'Unknown') + '</span>';
                }
            } catch (e) {
                output.innerHTML = '<span class="placeholder">Error: ' + e.message + '</span>';
            }
            btn.disabled = false;
            btn.textContent = 'Generate Code';
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function copyCode() {
            const output = document.getElementById('output');
            const text = output.textContent.replace('Copy', '').replace('Run Code', '').trim();
            navigator.clipboard.writeText(text);
        }

        async function runCode() {
            const output = document.getElementById('output');
            const execOutput = document.getElementById('exec-output');
            const code = output.textContent.replace('Copy', '').replace('Run Code', '').trim();
            if (!code || code.includes('will appear')) return;

            execOutput.style.display = 'block';
            execOutput.innerHTML = '<span class="placeholder">Running...</span>';

            try {
                const resp = await fetch('/execute', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({code: code})
                });
                const data = await resp.json();
                const result = (data.stdout || '') + (data.stderr ? ' STDERR: ' + data.stderr : '');
                execOutput.innerHTML = escapeHtml(result || '(no output)');
                if (data.error) {
                    execOutput.innerHTML += '<span style="color:#f85149;"> ' + escapeHtml(data.error) + '</span>';
                }
            } catch (e) {
                execOutput.innerHTML = '<span style="color:#f85149;">Error: ' + e.message + '</span>';
            }
        }

        fetch('/info').then(r => r.json()).then(data => {
            document.getElementById('model-info').textContent =
                `Model: ${data.n_params_M}M params | ${data.n_layer} layers | dim=${data.n_embd} | vocab=${data.vocab_size} | seq_len=${data.seq_len} | MLX on Apple Silicon`;
        }).catch(() => {});

        document.getElementById('prompt').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                doGenerate();
            }
        });
    </script>
</body>
</html>"""


def create_app():
    from flask import Flask, request, jsonify, Response

    app = Flask(__name__)

    print("Loading model...")
    model, config = load_model()
    print("Loading tokenizer...")
    enc = load_tokenizer()
    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Ready! Model: {n_params / 1e6:.1f}M params")

    @app.route("/")
    def index():
        return Response(HTML_TEMPLATE, mimetype="text/html")

    @app.route("/generate", methods=["POST"])
    def gen():
        data = request.get_json()
        prompt = data.get("prompt", "")
        temperature = float(data.get("temperature", 0.7))
        max_tokens = int(data.get("max_tokens", 256))

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        try:
            code = generate_text(model, enc, config, prompt,
                               max_tokens=max_tokens,
                               temperature=temperature)
            return jsonify({"code": code})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/execute", methods=["POST"])
    def execute():
        import subprocess
        data = request.get_json()
        code = data.get("code", "")
        if not code:
            return jsonify({"error": "No code provided"}), 400

        # Add a simple test call if the code defines a function
        test_code = code
        lines = code.strip().split("\n")
        # Find function definitions and add a test call
        for line in lines:
            if line.startswith("def "):
                func_name = line.split("(")[0].replace("def ", "").strip()
                # Auto-generate simple test based on function name
                test_calls = {
                    "reverse": f'print({func_name}("hello"))',
                    "palindrome": f'print({func_name}("racecar"))\nprint({func_name}("hello"))',
                    "prime": f'print({func_name}(7))\nprint({func_name}(10))',
                    "factorial": f'print({func_name}(5))',
                    "fibonacci": f'print({func_name}(10))',
                    "sort": f'print({func_name}([3,1,4,1,5,9,2,6]))',
                    "max": f'print({func_name}(3, 7))',
                    "min": f'print({func_name}(3, 7))',
                    "even": f'print({func_name}(4))\nprint({func_name}(7))',
                    "odd": f'print({func_name}(4))\nprint({func_name}(7))',
                    "sum": f'print({func_name}([1,2,3,4,5]))',
                    "average": f'print({func_name}([1,2,3,4,5]))',
                    "count": f'print({func_name}("hello world"))',
                    "upper": f'print({func_name}("hello"))',
                    "lower": f'print({func_name}("HELLO"))',
                    "length": f'print({func_name}([1,2,3]))',
                    "swap": f'print({func_name}(1, 2))',
                    "gcd": f'print({func_name}(12, 8))',
                    "binary": f'print({func_name}(42))',
                    "vowel": f'print({func_name}("hello world"))',
                    "square": f'print({func_name}(5))',
                    "cube": f'print({func_name}(3))',
                    "power": f'print({func_name}(2, 10))',
                    "add": f'print({func_name}(3, 5))',
                    "subtract": f'print({func_name}(10, 3))',
                    "multiply": f'print({func_name}(4, 7))',
                    "celsius": f'print({func_name}(100))',
                    "fahrenheit": f'print({func_name}(212))',
                    "leap": f'print({func_name}(2024))\nprint({func_name}(2023))',
                    "anagram": f'print({func_name}("listen", "silent"))',
                    "flatten": f'print({func_name}([[1,2],[3,[4,5]]]))',
                    "duplicates": f'print({func_name}([1,2,2,3,3,4]))',
                    "common": f'print({func_name}([1,2,3], [2,3,4]))',
                    "merge": f'print({func_name}({{"a":1}}, {{"b":2}}))',
                    "transpose": f'print({func_name}([[1,2],[3,4]]))',
                    "zip": f'print({func_name}([1,2,3], ["a","b","c"]))',
                    "capitalize": f'print({func_name}("hello world"))',
                    "word": f'print({func_name}("hello world foo"))',
                }
                for key, call in test_calls.items():
                    if key in func_name.lower():
                        test_code += "\n" + call
                        break
                else:
                    test_code += f'\nprint({func_name}.__doc__ or "Function defined: {func_name}")'
                break

        try:
            result = subprocess.run(
                ["python3", "-c", test_code],
                capture_output=True, text=True, timeout=5
            )
            return jsonify({
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            })
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Execution timed out (5s limit)"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/info")
    def info():
        return jsonify({
            "n_params_M": f"{n_params / 1e6:.1f}",
            "n_layer": config.n_layer,
            "n_embd": config.n_embd,
            "vocab_size": config.vocab_size,
            "seq_len": config.seq_len,
        })

    return app


if __name__ == "__main__":
    app = create_app()
    print("\nStarting server at http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
