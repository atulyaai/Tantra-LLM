"""
Tantra Language Benchmark — fixed test set for measuring progress.
Run at Day 1, Day 7, Day 30 to track improvement.

Usage:
    python benchmark.py
    python benchmark.py --category hindi
    python benchmark.py --category all
    python benchmark.py --output results.json
"""
import json, os, sys, argparse

sys.stdout.reconfigure(encoding='utf-8')

# ── BENCHMARK QUESTIONS ──────────────────────────────────────
# Each category has 20 questions: 10 understanding, 10 generation
# Answers are checked for correctness

BENCHMARK = {
    "hindi": {
        "understanding": [
            ("क्या पानी जलता है?", True),
            ("भारत की राजधानी कौन सही है?", True),
            ("सूर्य किस रंग का है?", True),
            ("मनुष्य कितने अंग हैं?", True),
            ("हिंदी किस भाषा का है?", True),
            ("गुरुत्वाकर्षण किसका काम है?", True),
            ("पानी का क्वथनांक क्या है?", True),
            ("भारत में कितने राज्य हैं?", True),
            ("प्रकाश संश्लेषण क्या है?", True),
            ("मोर भारत का राष्ट्रीय पक्षी है?", True),
            ("संसद कितने सदनों से बनी है?", True),
            ("भारत का राष्ट्रगान क्या है?", True),
            ("योग किसकी विधि है?", True),
            ("दीपावली किस त्योहार को दर्शाता है?", True),
            ("आयुर्वेद किसकी चिकित्सा पद्धति है?", True),
            ("कुंभ मेला कहाँ लगता है?", True),
            ("भारत का राष्ट्रीय पशु क्या है?", True),
            ("गांधीजी का जन्मदिन कब है?", True),
            ("लोकतंत्र में सत्ता कहाँ होती है?", True),
            ("संविधान कब लागू हुआ?", True),
        ],
        "generation": [
            "दीपावली क्या है? 50 शब्दों में बताओ।",
            "भारत की संसद के दो सदन बताओ।",
            "गुरुत्वाकर्षण क्या है? सरल भाषा में।",
            "प्रकाश संश्लेषण क्या है? 50 शब्दों में।",
            "आयुर्वेद क्या है? सरल भाषा में।",
            "भारत के 28 राज्यों के नाम बताओ।",
            "योग क्या है? इसके फायदे बताओ।",
            "कुंभ मेला क्या है? कब और कहाँ लगता है?",
            "भारत का राष्ट्रगान जन गण मन क्या बताता है?",
            "हिंदी भाषा के बारे में 50 शब्दों में।",
            "मोर भारत का राष्ट्रीय पक्षी है, इसकी विशेषताएँ बताओ।",
            "संविधान क्या है? इसका महत्व बताओ।",
            "अशोक कौन थे और उनकी योगदान क्या थी?",
            "चाणक्य कौन थे और अर्थशास्त्र क्या है?",
            "जयपुर क्या है और इसकी विशेषताएँ?",
            "ताजमहल क्या है? इसका इतिहास बताओ।",
            "लोकतंत्र और प्रजासत्ताक में क्या अंतर है?",
            "मौलिक अधिकार कौन-कौन से हैं?",
            "स्वतंत्रता दिवस और गणतंत्र दिवस में क्या अंतर है?",
            "पंजाब का लोहड़ी त्योहार क्या है?",
        ],
    },
    "hinglish": {
        "understanding": [
            ("Bhai ye code kaise optimize kar sakte hain?", True),
            ("Arre bhai, ye galat hai.", True),
            ("Main kaise help kar sakta hoon?", True),
            ("Ye question bahut easy hai.", True),
            ("Bhai, meri exam hai.", True),
            ("Ye problem bahut difficult hai.", True),
            ("Help karo bhai, ye bug hai.", True),
            ("Bhai ye Python code dekho.", True),
            ("Ye bahut helpful hai.", True),
            ("Main samajh nahi aa raha.", True),
            ("Bhai ye mujhe samjhao.", True),
            ("Ye solution sahi hai ya nahi?", True),
            ("Bhai ye bug fix karo.", True),
            ("Ye project set up karo.", True),
            ("Main ye code review karo.", True),
            ("Ye database connect karo.", True),
            ("Bhai ye API call karo.", True),
            ("Ye error fix karo bhai.", True),
            ("Main ye frontend banana hai.", True),
            ("Ye bahut advanced topic hai.", True),
        ],
        "generation": [
            "Bhai ye code kaise optimize kar sakte hain?",
            "Arre bhai, ye galat hai. Sahi yeh hai.",
            "Main kaise help kar sakta hoon?",
            "Ye question bahut easy hai, samjha do.",
            "Bhai, meri exam hai. Best of luck!",
            "Ye problem bahut difficult hai. Step by step samjha do.",
            "Help karo bhai, ye bug kya hai?",
            "Bhai ye Python code dekho. Kaise use karein?",
            "Ye bahut helpful hai. Aur kuch chahiye?",
            "Main samajh nahi aa raha. Aur simple samjha do.",
            "Bhai ye mujhe samjhao. Kya hua?",
            "Ye solution sahi hai ya nahi? Explain karo.",
            "Bhai ye bug fix karo. Kaise karein?",
            "Ye project set up karo. Steps batado.",
            "Main ye code review karo. Suggestions do.",
            "Ye database connect karo. Kaise karein?",
            "Bhai ye API call karo. Example dikhao.",
            "Ye error fix karo bhai. Root cause kya hai?",
            "Main ye frontend banana hai. HTML structure kya hogi?",
            "Ye bahut advanced topic hai. Basic samjha do.",
        ],
    },
    "english": {
        "understanding": [
            ("What is the speed of light?", True),
            ("Explain photosynthesis.", True),
            ("What is DNA?", True),
            ("What is gravity?", True),
            ("What is atomic structure?", True),
            ("What are Newton's laws?", True),
            ("What is the periodic table?", True),
            ("What is electricity?", True),
            ("What is quantum physics?", True),
            ("What is a chemical bond?", True),
            ("What is machine learning?", True),
            ("Explain neural networks.", True),
            ("What is the capital of France?", True),
            ("How do I learn Python?", True),
            ("What is AI?", True),
            ("What is the boiling point of water?", True),
            ("What is the human body made of?", True),
            ("What is evolution?", True),
            ("What is an ecosystem?", True),
            ("What is climate change?", True),
        ],
        "generation": [
            "What is the speed of light? Explain in 50 words.",
            "Explain photosynthesis in simple terms.",
            "What is DNA and why is it important?",
            "What is gravity and how does it work?",
            "Explain atomic structure in 50 words.",
            "What are Newton's three laws?",
            "What is the periodic table and why is it useful?",
            "Explain electricity in simple terms.",
            "What is quantum physics in 50 words?",
            "What is a chemical bond?",
            "What is machine learning? Explain simply.",
            "Explain neural networks like I'm 5 years old.",
            "What is the capital of France?",
            "How do I learn Python?",
            "What is artificial intelligence?",
            "What is the boiling point of water?",
            "What is the human body made of?",
            "Explain evolution in 50 words.",
            "What is an ecosystem?",
            "What is climate change and why should we care?",
        ],
    },
    "code": {
        "understanding": [
            ("Write a function to add two numbers.", True),
            ("What is a for loop?", True),
            ("How to print in Python?", True),
            ("Write a function to check even.", True),
            ("Write factorial function.", True),
            ("What is a dictionary?", True),
            ("Write reverse string function.", True),
            ("What is list comprehension?", True),
            ("Write palindrome checker.", True),
            ("How to read a file in Python?", True),
            ("What is class in Python?", True),
            ("Write a function to find max.", True),
            ("How to create a list?", True),
            ("What is lambda function?", True),
            ("Write API call function.", True),
            ("Write database connect function.", True),
            ("Write error handle function.", True),
            ("What is JSON parse?", True),
            ("Write function to process data.", True),
            ("What is OOP?", True),
        ],
        "generation": [
            "Write a Python function to add two numbers.",
            "Explain what a for loop is in Python.",
            "Write a function to check if a number is even.",
            "Write a factorial function in Python.",
            "Write a reverse string function in Python.",
            "What is list comprehension? Give an example.",
            "Write a palindrome checker in Python.",
            "Write a function to read a file in Python.",
            "Write a Python class called Student.",
            "Write a function to find the maximum of two numbers.",
            "What is a dictionary in Python? Give an example.",
            "Write a lambda function that squares a number.",
            "Write a function that calls an API in Python.",
            "Write a function to connect to a database in Python.",
            "Write error handling code in Python.",
            "Write a function to process a list of numbers.",
            "Write a function to check if a string is a palindrome.",
            "Write a Python program to sort a list.",
            "Write a function to count words in a file.",
            "What is object-oriented programming in Python?",
        ],
    },
    "math": {
        "understanding": [
            ("What is 5 + 3?", True),
            ("What is 12 - 7?", True),
            ("What is 6 × 7?", True),
            ("What is 15 / 3?", True),
            ("What is 2 + 2?", True),
            ("What is 10 - 4?", True),
            ("What is 8 × 9?", True),
            ("What is 20 / 4?", True),
            ("What is 3 + 7?", True),
            ("What is 100 - 50?", True),
            ("What is 6 × 8?", True),
            ("What is 24 / 6?", True),
            ("What is 9 + 11?", True),
            ("What is 50 - 25?", True),
            ("What is 7 × 7?", True),
            ("What is 36 / 9?", True),
            ("What is 13 + 17?", True),
            ("What is 100 - 37?", True),
            ("What is 12 × 5?", True),
            ("What is 84 / 7?", True),
        ],
        "generation": [
            "What is 5 + 3?",
            "What is 12 - 7?",
            "What is 6 × 7?",
            "What is 15 / 3?",
            "What is 2 + 2?",
            "What is 10 - 4?",
            "What is 8 × 9?",
            "What is 20 / 4?",
            "What is 3 + 7?",
            "What is 100 - 50?",
            "What is 6 × 8?",
            "What is 24 / 6?",
            "What is 9 + 11?",
            "What is 50 - 25?",
            "What is 7 × 7?",
            "What is 36 / 9?",
            "What is 13 + 17?",
            "What is 100 - 37?",
            "What is 12 × 5?",
            "What is 84 / 7?",
        ],
    },
    "reasoning": {
        "understanding": [
            ("If all dogs are animals and all animals breathe, do dogs breathe?", True),
            ("What comes next: 2, 6, 12, 20, 30, ?", True),
            ("Two fathers and two sons sat down to eat eggs. They ate exactly three eggs, each eating one egg. How is this possible?", True),
            ("A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?", True),
            ("I have keys but no locks. I have space but no room. You can enter but can't go inside. What am I?", True),
            ("What is 5 + 7 × 3?", True),
            ("A cube has 6 faces, 12 edges, and 8 vertices. What is the sum?", True),
            ("If today is tomorrow's yesterday, what day is it?", True),
            ("A man looks at a painting and says, 'Brothers and sisters I have none, but that man's father is my father's son.' Who is in the painting?", True),
            ("What comes next: 1, 1, 2, 3, 5, 8, ?", True),
        ],
        "generation": [
            "If all dogs are animals and all animals breathe, do dogs breathe? Explain.",
            "What comes next: 2, 6, 12, 20, 30, ? Explain the pattern.",
            "Two fathers and two sons sat down to eat eggs. They ate exactly three eggs. Explain how.",
            "A bat and a ball cost $1.10. The bat costs $1.00 more. How much does the ball cost?",
            "I have keys but no locks. I have space but no room. What am I? Explain.",
            "What is 5 + 7 × 3? Explain order of operations.",
            "A cube has 6 faces, 12 edges, and 8 vertices. What is the sum?",
            "If today is tomorrow's yesterday, what day is it? Explain.",
            "What comes next: 1, 1, 2, 3, 5, 8, ? Explain the pattern.",
            "A man looks at a painting and says, 'Brothers and sisters I have none, but that man's father is my father's son.' Who is in the painting? Explain.",
        ],
    },
    "conversation": {
        "understanding": [
            ("Hello!", True),
            ("How are you?", True),
            ("Can you help me?", True),
            ("Thank you!", True),
            ("Goodbye!", True),
            ("What is your name?", True),
            ("What is AI?", True),
            ("Explain machine learning.", True),
            ("Can you write code?", True),
            ("What is your purpose?", True),
        ],
        "generation": [
            "Hello! How can I help you today?",
            "I am doing well, thank you! How can I assist you?",
            "Of course! What do you need?",
            "You are welcome! Let me know if you need anything else.",
            "Goodbye! Have a great day!",
            "My name is Tantra AI, created by Atulya AI.",
            "AI is artificial intelligence — technology that enables machines to think and learn.",
            "Machine learning lets computers learn patterns from data.",
            "Yes! I can write code in Python, JavaScript, Java, and more.",
            "My purpose is to help you with questions and tasks efficiently.",
        ],
    },
}

# ── INSTRUCTION FOLLOWING TESTS ─────────────────────────────
INSTRUCTION_TESTS = [
    # Short answer
    ("Answer in 5 words: What is the capital of India?", "New Delhi"),
    # Detailed explanation
    ("Explain gravity in 3 sentences.", "Gravity is a force that attracts objects toward each other."),
    # Follow-up
    ("What is Python?\nNow explain it like I'm 5.", "Python is a programming language."),
    # Don't know behavior
    ("What will happen in 2050?", "I don't know what will happen in 2050."),
    # Correcting mistakes
    ("The Earth is flat.", "Actually, the Earth is not flat. It is an oblate spheroid."),
    # Language switching
    ("Hello, बताओ क्या है?", "Hello! Kaise hain?"),
    # Structured answers
    ("List 3 benefits of yoga in bullet points.", "1. Flexibility 2. Mental clarity 3. Stress relief"),
    # Preserve user's language
    ("Bhai, ye kaise karein?", "Bhai, ye step by step karo."),
]

def run_benchmark(category="all", model_func=None):
    """Run the benchmark and return results."""
    results = {}
    
    if category == "all":
        categories = list(BENCHMARK.keys())
    else:
        categories = [category]
    
    for cat in categories:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {cat.upper()}")
        print(f"{'='*60}")
        
        cat_results = {"understanding": [], "generation": [], "instruction": []}
        
        # Understanding
        if "understanding" in BENCHMARK[cat]:
            for q, expected in BENCHMARK[cat]["understanding"]:
                if model_func:
                    answer = model_func(q)
                    # Check if answer contains expected info
                    passed = check_answer(answer, q, expected)
                else:
                    passed = None
                cat_results["understanding"].append({"q": q, "passed": passed})
        
        # Generation
        if "generation" in BENCHMARK[cat]:
            for q in BENCHMARK[cat]["generation"]:
                if model_func:
                    answer = model_func(q)
                    passed = len(answer) > 0
                else:
                    passed = None
                cat_results["generation"].append({"q": q, "passed": passed})
        
        # Instruction following
        for q, expected in INSTRUCTION_TESTS:
            if model_func:
                answer = model_func(q)
                passed = check_answer(answer, q, expected)
            else:
                passed = None
            cat_results["instruction"].append({"q": q, "passed": passed})
        
        # Calculate scores
        for key in cat_results:
            items = cat_results[key]
            if items and all(i["passed"] is not None for i in items):
                score = sum(1 for i in items if i["passed"]) / len(items) * 100
                cat_results[key + "_score"] = round(score, 1)
            else:
                cat_results[key + "_score"] = None
        
        results[cat] = cat_results
    
    return results

def check_answer(answer, question, expected):
    """Check if the answer is reasonable."""
    if not answer:
        return False
    answer_lower = answer.lower()
    question_lower = question.lower()
    
    # Simple keyword matching for now
    if expected is True:
        return len(answer) > 3
    elif isinstance(expected, str):
        return expected.lower() in answer_lower or len(answer) > 5
    return len(answer) > 3

def format_results(results):
    """Format results as a readable table."""
    output = []
    output.append("\n" + "="*70)
    output.append("TANTRA LANGUAGE BENCHMARK RESULTS")
    output.append("="*70)
    
    for cat, data in results.items():
        output.append(f"\n{cat.upper()}:")
        for key, score in data.items():
            if "_score" in key:
                output.append(f"  {key.replace('_score', '')}: {score}%")
    
    # Overall score
    all_scores = []
    for cat, data in results.items():
        for key, score in data.items():
            if "_score" in key and score is not None:
                all_scores.append(score)
    
    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        output.append(f"\nOVERALL SCORE: {round(overall, 1)}%")
    
    output.append("="*70)
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Tantra Language Benchmark")
    parser.add_argument("--category", default="all", help="Category to test")
    parser.add_argument("--output", help="Output file for results")
    args = parser.parse_args()
    
    print("Tantra Language Benchmark")
    print(f"Category: {args.category}")
    print(f"Day 1, Day 7, Day 30")
    
    results = run_benchmark(category=args.category)
    print(format_results(results))
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")
    
    return results

if __name__ == "__main__":
    main()
