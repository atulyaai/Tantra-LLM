# Contributing to Tantra-LLM

Thank you for your interest in contributing to **Tantra-LLM**! We welcome contributions from AI researchers, software engineers, and performance optimization specialists.

## Code Architecture Guidelines

We adhere strictly to a **clean, modular, low-clutter** architecture:

1. **Keep the codebase flat**: All primary logic lives inside the `tantra/` package across 9 focused modules. Avoid creating unnecessary nested subdirectories or micro-packages.
2. **Zero superficial stubs**: Code should be runnable, typed, and well-documented.
3. **Hardware auto-adaptability**: Any new feature should gracefully work on CPU-only consumer systems (laptops) as well as multi-GPU desktop systems.

## Getting Started

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Tantra-LLM.git
   cd Tantra-LLM
   ```

2. **Create a Virtual Environment & Install Dependencies**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

3. **Run the Test Suite**:
   ```bash
   pytest tests/
   ```

4. **Run the Pipeline Smoke Test**:
   ```bash
   python main.py --mode probe
   python main.py
   ```

## Pull Request Process

1. Create a feature branch (`git checkout -b feature/amazing-feature`).
2. Write unit tests in `tests/` covering new capabilities or bug fixes.
3. Ensure all tests pass (`pytest tests/`).
4. Commit your changes with clear, descriptive commit messages.
5. Push to your branch and submit a Pull Request.

## License

By contributing to Tantra-LLM, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
