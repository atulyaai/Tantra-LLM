# Tantra-LLM Code Benchmark • Python Specialist
# Prompt: Write a Python function to reverse a string.

def reverse_string(s: str) -> str:
    """
    Reverses the given string.
    
    >>> reverse_string("hello")
    'olleh'
    >>> reverse_string("Tantra AI")
    'IA artnaT'
    """
    return s[::-1]

if __name__ == "__main__":
    test_str = "Tantra-LLM"
    print(f"Original: {test_str}")
    print(f"Reversed: {reverse_string(test_str)}")
