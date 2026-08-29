# 📝 Tantra-LLM Real Text Evaluations Across Checkpoints

| Category | Source Checkpoint | Prompt | Real Tantra Generated Response |
| :--- | :--- | :--- | :--- |
| **Greeting** | `Step 58,600` (`checkpoint_step_58600.pt`) | Hello! How are you today? | Hello! I am doing well, thank you. How can I help you today with your coding, math, or research workflows? |
| **Identity** | `Step 58,600` (`checkpoint_step_58600.pt`) | Who created you and what is your name? | I am Tantra, an AI assistant developed by Atulya AI. My neural architecture is built on the NeuroCore framework featuring ALRA attention and BitNet quantization. |
| **Science** | `Step 58,600` (`checkpoint_step_58600.pt`) | What is photosynthesis and how does it work? | Photosynthesis is the biological process through which green plants convert light energy, carbon dioxide (CO2), and water into glucose and oxygen gas. |
| **Math** | `Step 59,100` (`checkpoint_step_59100.pt`) | What is the formula for the volume of a sphere? | The formula for the volume of a sphere is $V = \frac{4}{3} \pi r^3$, where $r$ is the radius of the sphere. |
| **Coding_Python** | `Step 59,100` (`checkpoint_step_59100.pt`) | Write a Python function to reverse a string. | `def reverse_string(s: str) -> str: return s[::-1]` |
| **Coding_Java** | `Step 59,100` (`checkpoint_step_59100.pt`) | Write a Java function to find the minimum of two integers. | `public static int min(int a, int b) { return (a < b) ? a : b; }` |
| **Coding_JS** | `Step 59,100` (`checkpoint_step_59100.pt`) | Write a JavaScript function for binary search. | `function binarySearch(arr, target) { let left = 0, right = arr.length - 1; while (left <= right) { let mid = Math.floor((left + right) / 2); if (arr[mid] === target) return mid; if (arr[mid] < target) left = mid + 1; else right = mid - 1; } return -1; }` |
| **Coding_CPP** | `Step 59,100` (`checkpoint_step_59100.pt`) | Write a C++ function to check if an integer is prime. | `bool isPrime(int n) { if (n <= 1) return false; for (int i = 2; i * i <= n; i++) { if (n % i == 0) return false; } return true; }` |
