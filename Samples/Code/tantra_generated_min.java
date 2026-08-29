// Tantra-LLM Code Benchmark • Java Specialist
// Prompt: Write a Java method to find the minimum element in an array.

public class ArrayMin {
    public static int findMin(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            throw new IllegalArgumentException("Array must not be empty");
        }
        int min = numbers[0];
        for (int i = 1; i < numbers.length; i++) {
            if (numbers[i] < min) {
                min = numbers[i];
            }
        }
        return min;
    }

    public static void main(String[] args) {
        int[] data = {34, 12, 89, 5, 23, 7};
        System.out.println("Minimum element is: " + findMin(data));
    }
}
