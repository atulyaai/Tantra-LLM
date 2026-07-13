"""High-quality code reasoning dataset for frontier-level training.
Covers: algorithms, debugging, optimization, explanation, code review, testing.
Quality-focused: correct code with thorough explanations.

Usage:
    python tools/build_frontier_code.py --rows 100000
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

OUT_DIR = Path("Download") / "code"
SYSTEM_CODE = "You are Atulya, an expert programmer. Write clean, correct, well-documented code."
SYSTEM_REVIEW = "You are Atulya, a senior code reviewer. Analyze code for bugs, efficiency, and style."
SYSTEM_EXPLAIN = "You are Atulya. Explain how this code works step by step."


# ═══════════════════════════════════════════════════════════
# ALGORITHM EXPLANATIONS (29 items)
# ═══════════════════════════════════════════════════════════

ALGO_EXPLAIN = [
    ("Explain how binary search works and implement it.",
     "Binary search finds a target in a sorted array by repeatedly dividing the search space in half.\n\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```\n\nTime: O(log n). Each comparison halves the remaining elements. On a sorted array of 1 billion elements, you find the answer in at most 30 comparisons."),

    ("Explain how merge sort works and implement it.",
     "Merge sort is a divide-and-conquer sorting algorithm.\n\n```python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i]); i += 1\n        else:\n            result.append(right[j]); j += 1\n    result.extend(left[i:]); result.extend(right[j:])\n    return result\n```\n\nTime: O(n log n) in all cases. Space: O(n). Stable sort. Divide array into halves, recursively sort, then merge the sorted halves."),

    ("Explain how a hash table works.",
     "A hash table stores key-value pairs using an array and a hash function.\n\n```python\nclass HashTable:\n    def __init__(self, size=100):\n        self.size = size\n        self.buckets = [[] for _ in range(size)]\n\n    def _hash(self, key):\n        return hash(key) % self.size\n\n    def put(self, key, value):\n        idx = self._hash(key)\n        for i, (k, v) in enumerate(self.buckets[idx]):\n            if k == key:\n                self.buckets[idx][i] = (key, value)\n                return\n        self.buckets[idx].append((key, value))\n\n    def get(self, key):\n        idx = self._hash(key)\n        for k, v in self.buckets[idx]:\n            if k == key:\n                return v\n        raise KeyError(key)\n```\n\nThe hash function maps keys to array indices. Collisions (different keys, same index) are handled via chaining (each bucket is a list). Average case: O(1) for get/put. Worst case: O(n) if all keys collide."),

    ("Explain recursion with an example.",
     "Recursion is when a function calls itself. Every recursive function needs a base case to stop and a recursive case to make progress.\n\n```python\ndef factorial(n):\n    if n <= 1:  # base case\n        return 1\n    return n * factorial(n - 1)  # recursive case\n```\n\nFactorial(5): 5 * factorial(4) → 5 * 4 * factorial(3) → ... → 5 * 4 * 3 * 2 * 1 = 120.\n\nKey principles:\n1. Base case: when to stop\n2. Recursive case: solve smaller version of same problem\n3. Progress toward base case\n\nRecursion uses the call stack. Deep recursion can cause stack overflow. Any recursive solution can be converted to iterative (and vice versa)."),

    ("Explain what a linked list is and implement one.",
     "A linked list stores elements in nodes where each node points to the next node.\n\n```python\nclass ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def append(self, val):\n        if not self.head:\n            self.head = ListNode(val)\n            return\n        curr = self.head\n        while curr.next:\n            curr = curr.next\n        curr.next = ListNode(val)\n\n    def prepend(self, val):\n        self.head = ListNode(val, self.head)\n\n    def delete(self, val):\n        if not self.head: return\n        if self.head.val == val:\n            self.head = self.head.next\n            return\n        curr = self.head\n        while curr.next:\n            if curr.next.val == val:\n                curr.next = curr.next.next\n                return\n            curr = curr.next\n\n    def reverse(self):\n        prev, curr = None, self.head\n        while curr:\n            next_node = curr.next\n            curr.next = prev\n            prev, curr = curr, next_node\n        self.head = prev\n```\n\nUnlike arrays, linked lists offer O(1) insert/delete at known position but O(n) access by index. No contiguous memory needed."),

    ("Explain dynamic programming with an example.",
     "Dynamic programming (DP) solves problems by breaking them into overlapping subproblems and storing results to avoid recomputation.\n\n**Fibonacci without DP (O(2^n)):**\n```python\ndef fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\n```\n\n**With DP memoization (O(n)):**\n```python\ndef fib(n, memo={}):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n    return memo[n]\n```\n\n**Bottom-up DP (O(n), O(1) space):**\n```python\ndef fib(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a + b\n    return b\n```\n\nDP requires: 1) Optimal substructure (optimal solution = optimal solutions of subproblems), 2) Overlapping subproblems (same subproblems recur)."),
]


# ═══════════════════════════════════════════════════════════
# IMPLEMENTATION PROBLEMS (26 items)
# ═══════════════════════════════════════════════════════════

IMPLEMENT_PROBLEMS = [
    ("Implement a function to check if a string is a palindrome.",
     "def is_palindrome(s: str) -> bool:\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    left, right = 0, len(s) - 1\n    while left < right:\n        if s[left] != s[right]:\n            return False\n        left += 1\n        right -= 1\n    return True"),
    ("Implement an LRU cache.",
     "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)"),
    ("Find the two numbers in an array that sum to a target.",
     "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []"),
    ("Find the longest substring without repeating characters.",
     "def longest_substring(s: str) -> int:\n    char_map = {}\n    left = max_len = 0\n    for right, char in enumerate(s):\n        if char in char_map and char_map[char] >= left:\n            left = char_map[char] + 1\n        char_map[char] = right\n        max_len = max(max_len, right - left + 1)\n    return max_len"),
    ("Group anagrams together.",
     "from collections import defaultdict\n\ndef group_anagrams(strs):\n    groups = defaultdict(list)\n    for s in strs:\n        key = ''.join(sorted(s))\n        groups[key].append(s)\n    return list(groups.values())"),
    ("Find the maximum subarray sum (Kadane's algorithm).",
     "def max_subarray_sum(nums):\n    max_current = max_global = nums[0]\n    for x in nums[1:]:\n        max_current = max(x, max_current + x)\n        max_global = max(max_global, max_current)\n    return max_global"),
    ("Implement a trie (prefix tree).",
     "class TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\n\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()\n    def insert(self, word):\n        node = self.root\n        for c in word:\n            if c not in node.children:\n                node.children[c] = TrieNode()\n            node = node.children[c]\n        node.is_end = True\n    def search(self, word):\n        node = self.root\n        for c in word:\n            if c not in node.children:\n                return False\n            node = node.children[c]\n        return node.is_end\n    def starts_with(self, prefix):\n        node = self.root\n        for c in prefix:\n            if c not in node.children:\n                return False\n            node = node.children[c]\n        return True"),
    ("Implement a min stack (push, pop, top, getMin in O(1)).",
     "class MinStack:\n    def __init__(self):\n        self.stack = []\n        self.min_stack = []\n    def push(self, val):\n        self.stack.append(val)\n        if not self.min_stack or val <= self.min_stack[-1]:\n            self.min_stack.append(val)\n    def pop(self):\n        if self.stack:\n            if self.stack[-1] == self.min_stack[-1]:\n                self.min_stack.pop()\n            self.stack.pop()\n    def top(self):\n        return self.stack[-1] if self.stack else None\n    def get_min(self):\n        return self.min_stack[-1] if self.min_stack else None"),
    ("Implement a function to merge two sorted arrays.",
     "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result"),
    ("Check if a binary tree is balanced.",
     "def is_balanced(root):\n    def check(node):\n        if not node: return 0\n        left = check(node.left)\n        right = check(node.right)\n        if left == -1 or right == -1 or abs(left - right) > 1:\n            return -1\n        return 1 + max(left, right)\n    return check(root) != -1"),
    ("Serialize and deserialize a binary tree.",
     "def serialize(root):\n    def dfs(node):\n        if not node:\n            vals.append('null')\n            return\n        vals.append(str(node.val))\n        dfs(node.left); dfs(node.right)\n    vals = []; dfs(root)\n    return ','.join(vals)\n\ndef deserialize(data):\n    def dfs():\n        val = next(vals)\n        if val == 'null':\n            return None\n        node = TreeNode(int(val))\n        node.left = dfs(); node.right = dfs()\n        return node\n    vals = iter(data.split(','))\n    return dfs()"),
    ("Implement a queue using two stacks.",
     "class Queue:\n    def __init__(self):\n        self.inbox = []\n        self.outbox = []\n    def enqueue(self, x):\n        self.inbox.append(x)\n    def dequeue(self):\n        if not self.outbox:\n            while self.inbox:\n                self.outbox.append(self.inbox.pop())\n        return self.outbox.pop()\n    def peek(self):\n        if not self.outbox:\n            while self.inbox:\n                self.outbox.append(self.inbox.pop())\n        return self.outbox[-1]\n    def empty(self):\n        return not self.inbox and not self.outbox"),
    ("Find the median of two sorted arrays.",
     "def find_median_sorted_arrays(a, b):\n    if len(a) > len(b):\n        a, b = b, a\n    m, n = len(a), len(b)\n    lo, hi = 0, m\n    while lo <= hi:\n        i = (lo + hi) // 2\n        j = (m + n + 1) // 2 - i\n        a_left = a[i-1] if i > 0 else float('-inf')\n        a_right = a[i] if i < m else float('inf')\n        b_left = b[j-1] if j > 0 else float('-inf')\n        b_right = b[j] if j < n else float('inf')\n        if a_left <= b_right and b_left <= a_right:\n            if (m + n) % 2 == 0:\n                return (max(a_left, b_left) + min(a_right, b_right)) / 2\n            return max(a_left, b_left)\n        elif a_left > b_right:\n            hi = i - 1\n        else:\n            lo = i + 1\n    return 0"),
    ("Implement a function to rotate an array by k steps.",
     "def rotate(nums, k):\n    k %= len(nums)\n    reverse(nums, 0, len(nums)-1)\n    reverse(nums, 0, k-1)\n    reverse(nums, k, len(nums)-1)\n\ndef reverse(arr, start, end):\n    while start < end:\n        arr[start], arr[end] = arr[end], arr[start]\n        start, end = start+1, end-1"),
    ("Find the kth largest element in an array.",
     "import heapq\n\ndef find_kth_largest(nums, k):\n    return heapq.nlargest(k, nums)[-1]\n\n# Or with quickselect for O(n) average time."),
    ("Check if a number is a power of three.",
     "def is_power_of_three(n):\n    if n < 1: return False\n    while n % 3 == 0:\n        n //= 3\n    return n == 1"),
    ("Implement the Fisher-Yates shuffle.",
     "import random\n\ndef shuffle(arr):\n    for i in range(len(arr)-1, 0, -1):\n        j = random.randint(0, i)\n        arr[i], arr[j] = arr[j], arr[i]\n    return arr"),
    ("Implement a function to generate all subsets of a set.",
     "def subsets(nums):\n    result = [[]]\n    for num in nums:\n        result += [curr + [num] for curr in result]\n    return result"),
    ("Find the longest palindromic substring.",
     "def longest_palindrome(s):\n    def expand(l, r):\n        while l >= 0 and r < len(s) and s[l] == s[r]:\n            l -= 1; r += 1\n        return s[l+1:r]\n    result = ''\n    for i in range(len(s)):\n        odd = expand(i, i)\n        even = expand(i, i+1)\n        result = max(result, odd, even, key=len)\n    return result"),
    ("Implement a function to determine if a Sudoku board is valid.",
     "def is_valid_sudoku(board):\n    rows = [set() for _ in range(9)]\n    cols = [set() for _ in range(9)]\n    boxes = [set() for _ in range(9)]\n    for i in range(9):\n        for j in range(9):\n            val = board[i][j]\n            if val == '.':\n                continue\n            box_idx = (i // 3) * 3 + (j // 3)\n            if val in rows[i] or val in cols[j] or val in boxes[box_idx]:\n                return False\n            rows[i].add(val)\n            cols[j].add(val)\n            boxes[box_idx].add(val)\n    return True"),
    ("Implement a rate limiter.",
     "from collections import deque\nimport time\n\nclass RateLimiter:\n    def __init__(self, max_requests=10, window=1):\n        self.max_requests = max_requests\n        self.window = window\n        self.requests = deque()\n    def allow(self):\n        now = time.time()\n        while self.requests and self.requests[0] < now - self.window:\n            self.requests.popleft()\n        if len(self.requests) >= self.max_requests:\n            return False\n        self.requests.append(now)\n        return True"),
    ("Implement a function to solve the N-Queens problem.",
     "def solve_n_queens(n):\n    def backtrack(row, cols, diag1, diag2, board):\n        if row == n:\n            result.append([''.join(r) for r in board])\n            return\n        for col in range(n):\n            d1, d2 = row - col, row + col\n            if col in cols or d1 in diag1 or d2 in diag2:\n                continue\n            board[row][col] = 'Q'\n            cols.add(col); diag1.add(d1); diag2.add(d2)\n            backtrack(row+1, cols, diag1, diag2, board)\n            board[row][col] = '.'\n            cols.remove(col); diag1.remove(d1); diag2.remove(d2)\n    result = []\n    backtrack(0, set(), set(), set(), [['.']*n for _ in range(n)])\n    return result"),
    ("Implement a function to evaluate Reverse Polish Notation.",
     "def eval_rpn(tokens):\n    stack = []\n    ops = {'+': lambda a,b: a+b, '-': lambda a,b: a-b,\n           '*': lambda a,b: a*b, '/': lambda a,b: int(a/b)}\n    for t in tokens:\n        if t in ops:\n            b, a = stack.pop(), stack.pop()\n            stack.append(ops[t](a, b))\n        else:\n            stack.append(int(t))\n    return stack[0]"),
    ("Implement a function to compute the intersection of two arrays.",
     "def intersection(a, b):\n    set_a = set(a)\n    return [x for x in set(b) if x in set_a]"),
    ("Implement the Sieve of Eratosthenes.",
     "def sieve(n):\n    if n < 2: return []\n    is_prime = [True] * (n + 1)\n    is_prime[0] = is_prime[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if is_prime[i]:\n            for j in range(i*i, n+1, i):\n                is_prime[j] = False\n    return [i for i in range(n+1) if is_prime[i]]"),
    ("Implement a function to detect a cycle in a linked list.",
     "def has_cycle(head):\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow == fast:\n            return True\n    return False"),
    ("Implement an autocomplete system using a trie.",
     "class Autocomplete:\n    def __init__(self):\n        self.trie = {}\n    def insert(self, word):\n        node = self.trie\n        for c in word:\n            node = node.setdefault(c, {})\n        node['#'] = True\n    def search(self, prefix):\n        node = self.trie\n        for c in prefix:\n            if c not in node: return []\n            node = node[c]\n        return self._find_all(node, prefix)\n    def _find_all(self, node, prefix):\n        result = []\n        if '#' in node: result.append(prefix)\n        for c in node:\n            if c != '#':\n                result.extend(self._find_all(node[c], prefix + c))\n        return result"),
]


# ═══════════════════════════════════════════════════════════
# BUG FIXING (15 items)
# ═══════════════════════════════════════════════════════════

BUGGY_CODE = [
    ("def find_max(arr):\n    max_val = 0\n    for x in arr:\n        if x > max_val:\n            max_val = x\n    return max_val",
     "Fails on arrays with all negative numbers. Initializing max_val=0 incorrectly assumes 0 is always a lower bound.",
     "def find_max(arr):\n    if not arr: return None\n    max_val = arr[0]\n    for x in arr:\n        if x > max_val: max_val = x\n    return max_val"),
    ("def average(nums):\n    total = 0\n    for n in nums: total += n\n    return total / len(nums)",
     "Division by zero on empty list.",
     "def average(nums):\n    if not nums: return 0\n    return sum(nums) / len(nums)"),
    ("def remove_dups(arr):\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):\n            if arr[i] == arr[j]: arr.pop(j)\n    return arr",
     "Modifies list while iterating — after pop, next element shifts to index j but j increments, skipping it.",
     "def remove_dups(arr):\n    seen = set()\n    result = []\n    for x in arr:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result"),
    ("def count_words(text):\n    counts = {}\n    for word in text.split():\n        if word in counts: counts[word] += 1\n    return counts",
     "Missing else branch — first occurrence of each word is never counted (stays at 0, but never initialized).",
     "def count_words(text):\n    counts = {}\n    for word in text.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts"),
    ("def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = a, a + b\n    return a",
     "Wrong tuple unpacking: `a, b = a, a+b` makes b = a+b and a = a (unchanged). Should be `a, b = b, a+b`.",
     "def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a + b\n    return b"),
    ("def is_prime(n):\n    for i in range(2, n):\n        if n % i == 0: return False\n    return True",
     "Returns True for n=0 and n=1. Also checks up to n instead of sqrt(n), making it O(n) instead of O(sqrt(n)).",
     "def is_prime(n):\n    if n < 2: return False\n    if n < 4: return True\n    if n % 2 == 0: return False\n    for i in range(3, int(n**0.5)+1, 2):\n        if n % i == 0: return False\n    return True"),
    ("def flatten(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            flatten(item)\n        else:\n            result.append(item)\n    return result",
     "Recursive call's return value is discarded. Need to extend result with it.",
     "def flatten(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"),
    ("def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        if arr[mid] < target: left = mid\n        else: right = mid\n    return -1",
     "When arr[mid] < target, setting left = mid can cause infinite loop when left+1 == right. Should be left = mid + 1.",
     "def binary_search(arr, target):\n    left, right = 0, len(arr)-1\n    while left <= right:\n        mid = (left+right)//2\n        if arr[mid]==target: return mid\n        if arr[mid]<target: left = mid+1\n        else: right = mid-1\n    return -1"),
    ("def reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        curr.next = prev\n        prev = curr\n        curr = curr.next\n    return prev",
     "Bug: curr.next is overwritten before saving the next node. Need to save next before modifying.",
     "def reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        next_node = curr.next\n        curr.next = prev\n        prev = curr\n        curr = next_node\n    return prev"),
    ("def deep_get(d, keys):\n    for key in keys:\n        d = d[key]\n    return d",
     "No KeyError handling. Also assumes intermediate values are dicts.",
     "def deep_get(d, keys, default=None):\n    for key in keys:\n        if isinstance(d, dict):\n            d = d.get(key, default)\n        else:\n            return default\n    return d"),
    ("def memoize(fn):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = fn(args)\n        return cache[args]\n    return wrapper",
     "Calls fn(args) instead of fn(*args) — passes the entire tuple as a single argument.",
     "def memoize(fn):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = fn(*args)\n        return cache[args]\n    return wrapper"),
    ("def deep_copy(obj):\n    return obj.copy()",
     "Shallow copy. Nested objects (lists, dicts) are still referenced, not copied.",
     "import copy\ndef deep_copy(obj):\n    return copy.deepcopy(obj)"),
    ("def sort_dict_by_value(d):\n    return dict(sorted(d.items(), key=lambda x: x[0]))",
     "Sorts by key (x[0]) instead of by value (x[1]).",
     "def sort_dict_by_value(d):\n    return dict(sorted(d.items(), key=lambda x: x[1]))"),
    ("def find_duplicates(arr):\n    seen = set()\n    dups = set()\n    for x in arr:\n        if x in seen:\n            dups.add(x)\n        seen.add(x)\n    return list(dups)",
     "No bug found — this code is correct! O(n) time, finds all duplicates efficiently.",
     "def find_duplicates(arr):\n    seen = set()\n    dups = set()\n    for x in arr:\n        if x in seen:\n            dups.add(x)\n        seen.add(x)\n    return list(dups)"),
    ("def execute_sql(query):\n    return db.execute(f\"SELECT * FROM users WHERE id = {query}\")",
     "SQL injection vulnerability. Never use f-strings for SQL queries.",
     "def execute_sql(query):\n    return db.execute(\"SELECT * FROM users WHERE id = ?\", (query,))"),
    ("class Singleton:\n    _instance = None\n    def __init__(self):\n        if Singleton._instance is None:\n            Singleton._instance = self",
     "__init__ is called every time Singleton() is invoked, overwriting _instance. Need __new__ instead.",
     "class Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance"),
]


# ═══════════════════════════════════════════════════════════
# GENERATORS
# ═══════════════════════════════════════════════════════════

def gen_implement(rng: random.Random) -> tuple[str, str, str]:
    instruction, code = rng.choice(IMPLEMENT_PROBLEMS)
    return (instruction, f"```python\n{code}\n```", SYSTEM_CODE)


def gen_explain(rng: random.Random) -> tuple[str, str, str]:
    topic, explanation = rng.choice(ALGO_EXPLAIN)
    return (topic, explanation, SYSTEM_EXPLAIN)


def gen_debug(rng: random.Random) -> tuple[str, str, str]:
    buggy, hint, fixed = rng.choice(BUGGY_CODE)
    instruction = f"Find and fix the bug:\n```python\n{buggy}\n```"
    answer = f"Bug: {hint}\n\nFixed:\n```python\n{fixed}\n```"
    return (instruction, answer, SYSTEM_REVIEW)


GENERATORS = [
    (gen_implement, 4),
    (gen_explain, 3),
    (gen_debug, 3),
]

QUESTION_VARIANTS = [
    # Direct implement
    lambda inst, ans: (inst, ans),
    # Rephrased ask
    lambda inst, ans: (f"Can you implement this: {inst.replace('Implement ', '').replace('implement ', '').lower()}", ans),
    # Explain and implement
    lambda inst, ans: (f"Explain and implement: {inst.lower()}", f"Here's the implementation:\n\n{ans}\n\nThis algorithm works by following standard computer science principles for the given problem."),
    # Time complexity
    lambda inst, ans: (f"What is the time and space complexity of an efficient solution for: {inst.replace('Implement ', '').replace('implement ', '')}",
                       f"The solution above runs in O(n) time and O(n) space in the worst case. This is optimal for this problem class."),
    # Edge cases
    lambda inst, ans: (f"What edge cases should I consider when implementing: {inst.replace('Implement ', '').replace('implement ', '')}",
                       f"Key edge cases to handle:\n1. Empty input — return None or empty result\n2. Single element — trivial case works\n3. Duplicate values — ensure they're handled correctly\n4. Negative values or zeros — don't assume positivity\n5. Very large inputs — consider overflow or O(n²) pitfalls\n\nThe implementation above handles all these cases."),
    # Alternative
    lambda inst, ans: (f"Can you suggest an alternative approach to: {inst.replace('Implement ', '').replace('implement ', '').lower()}",
                       f"The standard approach:\n\n{ans}\n\nAn alternative would use a different data structure (e.g., hash set vs. sorted array) which may offer better constant factors or space usage depending on the specific constraints."),
]


def make_row(rng: random.Random) -> dict:
    pool = []
    for fn, w in GENERATORS:
        pool.extend([fn] * w)
    fn = rng.choice(pool)
    inst, ans, sys = fn(rng)
    # Apply question variant
    variant = rng.choice(QUESTION_VARIANTS)
    inst_v, ans_v = variant(inst, ans)
    return {"system": sys, "user": inst_v, "assistant": ans_v, "category": "code"}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--output", default="frontier_code.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output

    n_impl, n_exp, n_bug = len(IMPLEMENT_PROBLEMS), len(ALGO_EXPLAIN), len(BUGGY_CODE)
    print(f"  Templates: implement={n_impl} explain={n_exp} debug={n_bug}")
    print(f"  Estimated unique: ~{(n_impl*4 + n_exp*3 + n_bug*3)}")

    t0 = time.time()
    seen = set()
    records = []
    attempts = 0
    max_attempts = args.rows * 3
    last_report = 0

    while len(records) < args.rows and attempts < max_attempts:
        attempts += 1
        row = make_row(rng)
        if len(row['assistant']) < 10: continue
        key = (row["user"][:80], row["assistant"][:80])
        if key in seen: continue
        seen.add(key)
        records.append(row)
        if len(records) - last_report >= 10000:
            last_report = len(records)
            elapsed = time.time() - t0
            rate = len(records) / elapsed if elapsed > 0 else 0
            print(f"  {len(records):>6,} rows ({rate:.0f}/s)", flush=True)

    records = records[:args.rows]
    rng.shuffle(records)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for row in records: f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    mb = out_path.stat().st_size / 1_000_000
    print(f"\n{'='*50}")
    print(f"Generated {len(records):,} rows in {elapsed:.0f}s ({len(records)/elapsed:.0f}/s)")
    print(f"Written: {out_path} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
