"""
Download and prepare NLQ -> Python code dataset for training.
Uses CodeAlpaca-20k + MBPP datasets from HuggingFace.

Usage:
    uv run download_code_data.py

Saves formatted parquet shards to ~/.cache/autoresearch/data/
"""

import os
import json
import random
import pyarrow as pa
import pyarrow.parquet as pq
import requests

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

def download_json(url):
    """Download a JSON/JSONL file from HuggingFace."""
    print(f"Downloading {url}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    # Try JSON first, then JSONL
    text = resp.text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

def format_example(instruction, code):
    """Format a single NLQ->code example as training text."""
    instruction = instruction.strip()
    code = code.strip()
    return f"### Question\n{instruction}\n### Python Code\n{code}"

def augment_examples(examples):
    """Create augmented examples by rephrasing instructions."""
    rephrase_prefixes = [
        ("Write a function", "Create a function"),
        ("Write a function", "Implement a function"),
        ("Write a function", "Define a function"),
        ("Write a function", "Code a function"),
        ("Write a function", "Build a function"),
        ("Write a program", "Create a program"),
        ("Write a program", "Implement a program"),
        ("Write a Python", "Create a Python"),
        ("Write a Python", "Implement a Python"),
        ("Write code", "Create code"),
        ("Write code", "Implement code"),
    ]
    augmented = []
    random.seed(123)  # deterministic augmentation
    for ex in examples:
        # Extract instruction from format: "### Question\n{instruction}\n### Python Code\n{code}"
        parts = ex.split("### Python Code\n", 1)
        if len(parts) != 2:
            continue
        q_part = parts[0]  # "### Question\n{instruction}\n"
        code = parts[1]
        instruction = q_part.replace("### Question\n", "").strip()

        # Try each rephrase
        for old, new in rephrase_prefixes:
            if instruction.startswith(old):
                new_instruction = instruction.replace(old, new, 1)
                augmented.append(format_example(new_instruction, code))
                break  # one augmentation per example
        else:
            # For non-"Write" instructions, add "In Python, " prefix
            if random.random() < 0.3 and not instruction.startswith("In Python"):
                augmented.append(format_example(f"In Python, {instruction[0].lower()}{instruction[1:]}", code))

    return augmented


def collect_examples():
    """Collect examples from multiple sources."""
    examples = []

    # 1. CodeAlpaca-20k
    print("\n--- CodeAlpaca-20k ---")
    url = "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json"
    try:
        data = download_json(url)
        count = 0
        for item in data:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            inp = item.get("input", "")
            # Only keep Python-looking code (has def, import, print, =, etc.)
            if not output or len(output) < 10:
                continue
            # Combine instruction + input if input exists
            if inp:
                instruction = f"{instruction}\nInput: {inp}"
            # Filter for Python-ish examples
            python_markers = ["def ", "import ", "print(", "for ", "while ", "if ", "class ", "return ", " = ", "lambda "]
            if any(m in output for m in python_markers):
                examples.append(format_example(instruction, output))
                count += 1
        print(f"  Collected {count} Python examples from CodeAlpaca")
    except Exception as e:
        print(f"  Failed to download CodeAlpaca: {e}")

    # 2. MBPP (Mostly Basic Python Problems)
    print("\n--- MBPP ---")
    url = "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/main/mbpp.jsonl"
    try:
        data = download_json(url)
        count = 0
        for item in data:
            text = item.get("text", "")
            code = item.get("code", "")
            if text and code:
                examples.append(format_example(text, code))
                count += 1
        print(f"  Collected {count} examples from MBPP")
    except Exception as e:
        print(f"  Failed to download MBPP: {e}")

    # 3. Evol-Instruct Code (WizardCoder-style evolved instructions)
    print("\n--- Evol-Instruct-Code-80k (Python subset) ---")
    url = "https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1/resolve/main/EvolInstruct-Code-80k.json"
    try:
        data = download_json(url)
        count = 0
        for item in data:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            if not output or len(output) < 20:
                continue
            # Filter for Python
            python_markers = ["def ", "import ", "print(", "class ", "return ", " = ", "lambda ", "for ", "while "]
            # Must look like Python (not Java, C++, etc.)
            non_python = ["public static", "System.out", "#include", "cout <<", "console.log", "func ", "fn "]
            if any(m in output for m in python_markers) and not any(m in output for m in non_python):
                # Skip very long examples (> 1500 chars) to keep seq_len manageable
                formatted = format_example(instruction, output)
                if len(formatted) < 2000:
                    examples.append(formatted)
                    count += 1
        print(f"  Collected {count} Python examples from Evol-Instruct-Code")
    except Exception as e:
        print(f"  Failed to download Evol-Instruct-Code: {e}")

    # 4. Code Instructions from Alpaca (120k)
    print("\n--- Code-Instructions-120k (Python subset) ---")
    url = "https://huggingface.co/datasets/TokenBender/code_instructions_122k_alpaca_style/resolve/main/data.json"
    try:
        data = download_json(url)
        count = 0
        for item in data:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            inp = item.get("input", "")
            if not output or len(output) < 20:
                continue
            if inp:
                instruction = f"{instruction}\nInput: {inp}"
            python_markers = ["def ", "import ", "print(", "class ", "return ", " = ", "lambda "]
            non_python = ["public static", "System.out", "#include", "cout <<", "console.log"]
            if any(m in output for m in python_markers) and not any(m in output for m in non_python):
                formatted = format_example(instruction, output)
                if len(formatted) < 2000:
                    examples.append(formatted)
                    count += 1
        print(f"  Collected {count} Python examples from Code-Instructions-120k")
    except Exception as e:
        print(f"  Failed to download Code-Instructions-120k: {e}")

    # 5. Generate some synthetic simple examples for more data
    print("\n--- Synthetic examples ---")
    synthetic = generate_synthetic_examples()
    examples.extend(synthetic)
    print(f"  Generated {len(synthetic)} synthetic examples")

    # 6. Data augmentation — rephrase instructions
    print("\n--- Augmenting via instruction rephrasing ---")
    augmented = augment_examples(examples)
    examples.extend(augmented)
    print(f"  Generated {len(augmented)} augmented examples")

    return examples

def generate_synthetic_examples():
    """Generate simple synthetic NLQ -> Python code examples."""
    examples = []
    templates = [
        ("Write a function to add two numbers", "def add(a, b):\n    return a + b"),
        ("Write a function to subtract two numbers", "def subtract(a, b):\n    return a - b"),
        ("Write a function to multiply two numbers", "def multiply(a, b):\n    return a * b"),
        ("Write a function to find the maximum of two numbers", "def find_max(a, b):\n    return max(a, b)"),
        ("Write a function to find the minimum of two numbers", "def find_min(a, b):\n    return min(a, b)"),
        ("Write a function to check if a number is even", "def is_even(n):\n    return n % 2 == 0"),
        ("Write a function to check if a number is odd", "def is_odd(n):\n    return n % 2 != 0"),
        ("Write a function to check if a number is positive", "def is_positive(n):\n    return n > 0"),
        ("Write a function to check if a number is negative", "def is_negative(n):\n    return n < 0"),
        ("Write a function to calculate the square of a number", "def square(n):\n    return n ** 2"),
        ("Write a function to calculate the cube of a number", "def cube(n):\n    return n ** 3"),
        ("Write a function to calculate the absolute value", "def absolute(n):\n    return abs(n)"),
        ("Write a function to reverse a string", "def reverse_string(s):\n    return s[::-1]"),
        ("Write a function to check if a string is a palindrome", "def is_palindrome(s):\n    return s == s[::-1]"),
        ("Write a function to count characters in a string", "def count_chars(s):\n    return len(s)"),
        ("Write a function to convert a string to uppercase", "def to_upper(s):\n    return s.upper()"),
        ("Write a function to convert a string to lowercase", "def to_lower(s):\n    return s.lower()"),
        ("Write a function to find the length of a list", "def list_length(lst):\n    return len(lst)"),
        ("Write a function to sum all elements in a list", "def sum_list(lst):\n    return sum(lst)"),
        ("Write a function to find the average of a list", "def average(lst):\n    return sum(lst) / len(lst)"),
        ("Write a function to find the factorial of a number", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
        ("Write a function to generate fibonacci numbers up to n", "def fibonacci(n):\n    a, b = 0, 1\n    result = []\n    while a < n:\n        result.append(a)\n        a, b = b, a + b\n    return result"),
        ("Write a function to check if a number is prime", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
        ("Write a function to sort a list in ascending order", "def sort_ascending(lst):\n    return sorted(lst)"),
        ("Write a function to sort a list in descending order", "def sort_descending(lst):\n    return sorted(lst, reverse=True)"),
        ("Write a function to remove duplicates from a list", "def remove_duplicates(lst):\n    return list(set(lst))"),
        ("Write a function to flatten a nested list", "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"),
        ("Write a function to count words in a string", "def count_words(s):\n    return len(s.split())"),
        ("Write a function to find common elements in two lists", "def common_elements(lst1, lst2):\n    return list(set(lst1) & set(lst2))"),
        ("Write a function to merge two dictionaries", "def merge_dicts(d1, d2):\n    result = d1.copy()\n    result.update(d2)\n    return result"),
        ("Write a function to swap two variables", "def swap(a, b):\n    return b, a"),
        ("Write a function to find the GCD of two numbers", "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"),
        ("Write a function to find the LCM of two numbers", "def lcm(a, b):\n    from math import gcd\n    return abs(a * b) // gcd(a, b)"),
        ("Write a function to convert celsius to fahrenheit", "def celsius_to_fahrenheit(c):\n    return c * 9/5 + 32"),
        ("Write a function to convert fahrenheit to celsius", "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5/9"),
        ("Write a function to check if a year is a leap year", "def is_leap_year(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)"),
        ("Write a function to count vowels in a string", "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
        ("Write a function to capitalize the first letter of each word", "def capitalize_words(s):\n    return s.title()"),
        ("Write a function to find the second largest number in a list", "def second_largest(lst):\n    unique = list(set(lst))\n    unique.sort()\n    return unique[-2] if len(unique) >= 2 else None"),
        ("Write a function to check if two strings are anagrams", "def are_anagrams(s1, s2):\n    return sorted(s1.lower()) == sorted(s2.lower())"),
        ("Write a function to find the power of a number", "def power(base, exp):\n    return base ** exp"),
        ("Write a function to create a list of squares from 1 to n", "def squares(n):\n    return [i**2 for i in range(1, n+1)]"),
        ("Write a function to zip two lists together", "def zip_lists(lst1, lst2):\n    return list(zip(lst1, lst2))"),
        ("Write a function to find the index of an element in a list", "def find_index(lst, element):\n    try:\n        return lst.index(element)\n    except ValueError:\n        return -1"),
        ("Write a function to read a file and return its contents", "def read_file(path):\n    with open(path, 'r') as f:\n        return f.read()"),
        ("Write a function to write text to a file", "def write_file(path, text):\n    with open(path, 'w') as f:\n        f.write(text)"),
        ("Write a function to count occurrences of an element in a list", "def count_occurrences(lst, element):\n    return lst.count(element)"),
        ("Write a function to transpose a matrix", "def transpose(matrix):\n    return [list(row) for row in zip(*matrix)]"),
        ("Write a function to binary search in a sorted list", "def binary_search(lst, target):\n    low, high = 0, len(lst) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if lst[mid] == target:\n            return mid\n        elif lst[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1"),
        ("Write a function to implement bubble sort", "def bubble_sort(lst):\n    n = len(lst)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if lst[j] > lst[j+1]:\n                lst[j], lst[j+1] = lst[j+1], lst[j]\n    return lst"),
        ("Write a function to convert a number to binary string", "def to_binary(n):\n    return bin(n)[2:]"),
        ("Write a function to find all prime numbers up to n", "def primes_up_to(n):\n    sieve = [True] * (n + 1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if sieve[i]:\n            for j in range(i*i, n+1, i):\n                sieve[j] = False\n    return [i for i in range(n+1) if sieve[i]]"),
    ]
    for instruction, code in templates:
        examples.append(format_example(instruction, code))
    return examples


def save_as_shards(examples, original_examples=None, num_train_shards=5):
    """Save examples as parquet shards in the data directory.

    If original_examples is provided, the val set is derived from the original
    examples only (same seed=42 split) to keep val comparable across runs.
    New examples (in `examples` but not in original) go only to training.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Clear old data and tokenizer
    for f in os.listdir(DATA_DIR):
        if f.endswith(".parquet"):
            os.remove(os.path.join(DATA_DIR, f))
    if os.path.exists(TOKENIZER_DIR):
        for f in os.listdir(TOKENIZER_DIR):
            os.remove(os.path.join(TOKENIZER_DIR, f))

    if original_examples is not None:
        # Reproduce the original val split exactly
        orig = list(original_examples)
        random.seed(42)
        random.shuffle(orig)
        val_size = max(len(orig) // 10, 50)
        val_examples = orig[-val_size:]
        orig_train = orig[:-val_size]

        # Val set as a set for dedup
        val_set = set(val_examples)
        # All examples minus val = training (includes new data)
        train_examples = [e for e in examples if e not in val_set]
        print(f"  Val set: {len(val_examples)} (from original split, unchanged)")
        print(f"  Train set: {len(train_examples)} unique (original + new data)")
    else:
        random.seed(42)
        random.shuffle(examples)
        val_size = max(len(examples) // 10, 50)
        train_examples = examples[:-val_size]
        val_examples = examples[-val_size:]

    # Duplicate training data for more exposure
    repetitions = 20
    train_examples = train_examples * repetitions
    random.seed(42)
    random.shuffle(train_examples)

    # Save training shards
    shard_size = len(train_examples) // num_train_shards
    for i in range(num_train_shards):
        start = i * shard_size
        end = start + shard_size if i < num_train_shards - 1 else len(train_examples)
        shard_examples = train_examples[start:end]
        table = pa.table({"text": shard_examples})
        path = os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")
        pq.write_table(table, path)
        print(f"  Saved {path} ({len(shard_examples)} examples)")

    # Save validation shard
    val_table = pa.table({"text": val_examples})
    val_path = os.path.join(DATA_DIR, f"shard_{num_train_shards:05d}.parquet")
    pq.write_table(val_table, val_path)
    print(f"  Saved {val_path} ({len(val_examples)} validation examples)")

    print(f"\nTotal: {len(train_examples)} train, {len(val_examples)} val examples")
    print(f"Unique train examples: {len(train_examples) // repetitions}")


def collect_original_examples():
    """Collect only the original sources (CodeAlpaca + MBPP + synthetic).
    This reproduces the exact same dataset used for the original val split.
    """
    examples = []

    # 1. CodeAlpaca-20k (same as before)
    url = "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json"
    try:
        data = download_json(url)
        for item in data:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            inp = item.get("input", "")
            if not output or len(output) < 10:
                continue
            if inp:
                instruction = f"{instruction}\nInput: {inp}"
            python_markers = ["def ", "import ", "print(", "for ", "while ", "if ", "class ", "return ", " = ", "lambda "]
            if any(m in output for m in python_markers):
                examples.append(format_example(instruction, output))
    except Exception:
        pass

    # 2. Synthetic
    examples.extend(generate_synthetic_examples())
    return examples


if __name__ == "__main__":
    print("=== Downloading NLQ -> Python Code Dataset ===\n")

    # First collect original sources (for stable val split)
    print("--- Collecting original sources (for val split) ---")
    original_examples = collect_original_examples()
    print(f"  Original sources: {len(original_examples)} examples")

    # Then collect everything (original + new)
    print("\n--- Collecting all sources ---")
    all_examples = collect_examples()
    print(f"\nTotal collected: {len(all_examples)} examples")

    print("\n=== Saving as parquet shards ===\n")
    save_as_shards(all_examples, original_examples=original_examples, num_train_shards=5)

    print("\n=== Done! ===")
    print("Next steps:")
    print("  1. Run: uv run prepare.py --num-shards 5")
    print("  2. Run: uv run train.py")
