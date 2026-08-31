// Tantra-LLM Code Benchmark • C++ Specialist
// Prompt: Write a C++ function to check if a number is prime.

#include <iostream>

bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

int main() {
    int testNum = 29;
    std::cout << testNum << (isPrime(testNum) ? " is prime." : " is not prime.") << std::endl;
    return 0;
}
