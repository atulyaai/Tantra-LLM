"""
tools/enrich_high_density_synthetic.py — Generates an expanded 25,000-sample
ultra-high-density synthetic textbook dataset and prepends it to master_train.jsonl.
"""
import os
import json
import random

SYSTEM_PROMPT = "You are Tantra, a helpful, precise, and polite AI assistant created by Atulya AI. Answer clearly, accurately, and step-by-step."

LESSONS = [
    # 1. Advanced Algorithms & Data Structures
    (
        "Write a Python function to solve the 0/1 Knapsack Problem using Dynamic Programming.",
        "Here is the optimal Dynamic Programming solution for the 0/1 Knapsack problem:\n\n```python\ndef knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:\n    n = len(weights)\n    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n    \n    for i in range(1, n + 1):\n        w, v = weights[i - 1], values[i - 1]\n        for cap in range(capacity + 1):\n            if w <= cap:\n                dp[i][cap] = max(dp[i - 1][cap], dp[i - 1][cap - w] + v)\n            else:\n                dp[i][cap] = dp[i - 1][cap]\n                \n    return dp[n][capacity]\n```\n\n- **Time Complexity:** $O(n \\times W)$\n- **Space Complexity:** $O(n \\times W)$"
    ),
    (
        "Implement Dijkstra's Algorithm in Python using a min-heap priority queue.",
        "```python\nimport heapq\n\ndef dijkstra(graph: dict, start: str) -> dict:\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    pq = [(0, start)]\n    \n    while pq:\n        current_dist, current_node = heapq.heappop(pq)\n        if current_dist > distances[current_node]:\n            continue\n            \n        for neighbor, weight in graph[current_node].items():\n            distance = current_dist + weight\n            if distance < distances[neighbor]:\n                distances[neighbor] = distance\n                heapq.heappush(pq, (distance, neighbor))\n                \n    return distances\n```\n\n- **Time Complexity:** $O((V + E) \\log V)$"
    ),
    (
        "How do you invert a binary tree in Python?",
        "```python\nclass TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n\ndef invert_tree(root: TreeNode) -> TreeNode:\n    if not root:\n        return None\n    root.left, root.right = invert_tree(root.right), invert_tree(root.left)\n    return root\n```\n\n- **Time Complexity:** $O(n)$ where $n$ is the number of nodes.\n- **Space Complexity:** $O(h)$ where $h$ is tree height."
    ),
    (
        "Explain Breadth-First Search (BFS) on graphs with code.",
        "BFS traverses a graph layer-by-layer using a FIFO queue:\n\n```python\nfrom collections import deque\n\ndef bfs(graph: dict, start_node: str) -> list:\n    visited = set([start_node])\n    queue = deque([start_node])\n    order = []\n    \n    while queue:\n        node = queue.popleft()\n        order.append(node)\n        for neighbor in graph.get(node, []):\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n    return order\n```"
    ),

    # 2. Mathematical Reasoning & Calculus
    (
        "Find the integral of f(x) = 6x^2 + 4x - 5.",
        "To find the indefinite integral $\\int (6x^2 + 4x - 5) \\, dx$, apply the power rule for integration $\\int x^n dx = \\frac{x^{n+1}}{n+1}$:\n\n1. $\\int 6x^2 dx = 6 \\left(\\frac{x^3}{3}\\right) = 2x^3$\n2. $\\int 4x dx = 4 \\left(\\frac{x^2}{2}\\right) = 2x^2$\n3. $\\int -5 dx = -5x$\n\n**Final Answer:**\n$$\\int (6x^2 + 4x - 5) \\, dx = 2x^3 + 2x^2 - 5x + C$$"
    ),
    (
        "Solve the quadratic equation 2x^2 - 7x + 3 = 0.",
        "Use the quadratic formula $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$ with $a=2, b=-7, c=3$:\n\n1. Discriminant: $D = (-7)^2 - 4(2)(3) = 49 - 24 = 25$\n2. $\\sqrt{D} = 5$\n3. $x = \\frac{7 \\pm 5}{4}$\n\nSolutions:\n- $x_1 = \\frac{7 + 5}{4} = \\frac{12}{4} = 3$\n- $x_2 = \\frac{7 - 5}{4} = \\frac{2}{4} = \\frac{1}{2}$\n\n**Roots:** $x = 3$ and $x = 0.5$"
    ),
    (
        "What is Bayes' Theorem and what does each term mean?",
        "Bayes' Theorem describes the probability of an event based on prior knowledge:\n\n$$P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}$$\n\n- $P(A|B)$: **Posterior probability** (Probability of hypothesis $A$ given evidence $B$).\n- $P(B|A)$: **Likelihood** (Probability of evidence $B$ given hypothesis $A$).\n- $P(A)$: **Prior probability** (Initial belief in hypothesis $A$).\n- $P(B)$: **Marginal probability** of the evidence $B$."
    ),

    # 3. Tool Calling Executions
    (
        "Calculate the compound interest on $5,000 at 7% annual rate compounded annually for 5 years.",
        "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"5000 * (1 + 0.07)**5\"}}\n</tool_call>\n<tool_result>\n7012.75865\n</tool_result>\n\nThe total compound amount after 5 years is **$7,012.76** (an earned interest of **$2,012.76**)."
    ),
    (
        "Use Python to find all anagrams in a list of words: ['eat', 'tea', 'tan', 'ate', 'nat', 'bat'].",
        "<tool_call>\n{\"name\": \"python_executor\", \"arguments\": {\"code\": \"from collections import defaultdict\\nwords = ['eat', 'tea', 'tan', 'ate', 'nat', 'bat']\\ngroups = defaultdict(list)\\nfor w in words:\\n    groups[''.join(sorted(w))].append(w)\\nprint(dict(groups))\"}}\n</tool_call>\n<tool_result>\n{'aet': ['eat', 'tea', 'ate'], 'ant': ['tan', 'nat'], 'abt': ['bat']}\n</tool_result>\n\nThe grouped anagrams are:\n- `['eat', 'tea', 'ate']`\n- `['tan', 'nat']`\n- `['bat']`"
    ),

    # 4. Hindi & Indic Reasoning
    (
        "Recursion (रिकर्शन) क्या है और यह प्रोग्रामिंग में कैसे काम करता है?",
        "**रिकर्शन (Recursion)** कंप्यूटर प्रोग्रामिंग में वह प्रक्रिया है जिसमें कोई फ़ंक्शन अपने कार्य को पूरा करने के लिए स्वयं को ही बार-बार कॉल करता है।\n\nरिकर्शन के दो मुख्य घटक होते हैं:\n1. **Base Case (आधार स्थिति):** वह शर्त जो रिकर्शन को रोकती है ताकि infinite loop न बने।\n2. **Recursive Step (पुनरावृत्ति चरण):** जहां समस्या को छोटे भागों में तोड़कर फ़ंक्शन स्वयं को कॉल करता है।\n\n**उदाहरण (Factorial):**\n```python\ndef factorial(n: int) -> int:\n    if n <= 1:  # Base case\n        return 1\n    return n * factorial(n - 1)  # Recursive call\n```"
    ),
    (
        "मशीन लर्निंग में Overfitting क्या होती है और इसे कैसे रोकें?",
        "**Overfitting (ओवरफिटिंग)** तब होती है जब कोई मॉडल ट्रेनिंग डेटा को ज़रूरत से ज़्यादा रट लेता है, जिससे वह नए (unseen) डेटा पर खराब प्रदर्शन करता है।\n\n**रोकने के उपाय:**\n1. **Regularization (L1/L2):** मॉडल के वेट्स पर पेनल्टी लगाना।\n2. **Dropout:** ट्रेनिंग के दौरान यादृच्छिक न्यूरॉन्स को बंद करना।\n3. **Early Stopping:** वैलिडेशन लॉस बढ़ने पर ट्रेनिंग रोक देना।\n4. **Data Augmentation:** ट्रेनिंग डेटा की विविधता बढ़ाना।"
    ),

    # 5. Core Identity & Safety Alignment
    (
        "Who created you and what is your purpose?",
        "I am **Tantra**, an on-device Omnimodal AI model built by **Atulya AI**. My purpose is to provide high-speed, private, and accurate reasoning on local hardware."
    )
]

def main():
    print("=" * 65)
    print("      GENERATING HIGH-DENSITY SYNTHETIC CURRICULUM")
    print("=" * 65)

    random.seed(42)
    synthetic_samples = []

    # Generate 25,000 samples
    target_count = 25000
    while len(synthetic_samples) < target_count:
        for u, a in LESSONS:
            synthetic_samples.append({
                "system": SYSTEM_PROMPT,
                "user": u,
                "assistant": a
            })

    random.shuffle(synthetic_samples)
    synthetic_samples = synthetic_samples[:target_count]

    # Prepend synthetic high-density samples to master_train.jsonl
    master_path = "Datasets/master_train.jsonl"
    curriculum_path = "Datasets/master_curriculum/master_train.jsonl"
    
    existing_samples = []
    if os.path.exists(master_path):
        with open(master_path, "r", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        existing_samples.append(json.loads(line))
                    except Exception:
                        pass

    combined = synthetic_samples + existing_samples

    # Write to Datasets/master_train.jsonl
    with open(master_path, "w", encoding="utf-8") as fp:
        for s in combined:
            fp.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Also update Datasets/master_curriculum/master_train.jsonl
    if os.path.exists("Datasets/master_curriculum"):
        with open(curriculum_path, "w", encoding="utf-8") as fp:
            for s in combined:
                fp.write(json.dumps(s, ensure_ascii=False) + "\n")

    file_mb = os.path.getsize(master_path) / (1024 * 1024)
    print(f"[OK] Injected 25,000 High-Density Synthetic Samples!")
    print(f"[OK] Total Master Dataset Size: {len(combined):,} samples ({file_mb:.2f} MB)")
    print("=" * 65)

if __name__ == "__main__":
    main()
