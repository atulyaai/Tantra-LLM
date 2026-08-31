# 📝 Tantra-LLM Benchmark Evaluation Targets (Gold Standard References)

This dataset defines the **standard evaluation suite prompts and ideal ground-truth reference responses** used to benchmark Tantra's zero-shot accuracy, mathematical reasoning, and coding capabilities during multi-phase training.

---

### 🎯 Gold Standard Reference Responses:

| Category | Benchmark Type | Evaluation Prompt | Target Gold Reference Response |
| :--- | :--- | :--- | :--- |
| **Greeting** | Conversational Persona | *Hello! How are you today?* | Hello! I am doing well, thank you. How can I help you today with your coding, math, or research workflows? |
| **Identity** | AI System Grounding | *Who created you and what is your name?* | I am Tantra, an AI assistant developed by Atulya AI. My neural architecture is built on the NeuroCore framework featuring ALRA attention and BitNet quantization. |
| **Science** | Conceptual Knowledge | *What is photosynthesis and how does it work?* | Photosynthesis is the biological process through which green plants convert light energy, carbon dioxide (CO2), and water into glucose and oxygen gas. |
| **Math** | Quantitative Reasoning | *What is the formula for the volume of a sphere?* | The formula for the volume of a sphere is $V = \frac{4}{3} \pi r^3$, where $r$ is the radius of the sphere. |
| **Coding_Python** | Code Generation | *Write a Python function to reverse a string.* | `def reverse_string(s: str) -> str: return s[::-1]` |
| **Coding_Java** | Multi-Language Syntax | *Write a Java function to find the minimum of two integers.* | `public static int min(int a, int b) { return (a < b) ? a : b; }` |
| **Coding_JS** | Algorithmic Implementation | *Write a JavaScript function for binary search.* | `function binarySearch(arr, target) { let left = 0, right = arr.length - 1; while (left <= right) { let mid = Math.floor((left + right) / 2); if (arr[mid] === target) return mid; if (arr[mid] < target) left = mid + 1; else right = mid - 1; } return -1; }` |
| **Coding_CPP** | Systems Programming | *Write a C++ function to check if an integer is prime.* | `bool isPrime(int n) { if (n <= 1) return false; for (int i = 2; i * i <= n; i++) { if (n % i == 0) return false; } return true; }` |

---

*Note: For actual verbatim outputs produced by the raw Step 61,000 baseline model prior to instruction tuning, refer to [`tantra_checkpoint_raw_evals.md`](tantra_checkpoint_raw_evals.md).*
