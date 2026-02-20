# Contributing to LLM-Powered Automated FMEA Generator

Thank you for your interest in contributing to the **LLM-Powered Automated FMEA Generator**! 🚀

This document outlines the development workflow, contribution standards, and technical guidelines for maintaining high-quality contributions.

------------------------------------------------------------------------

## 🚨 Contribution Rules (Strict Enforcement)

> **Read this section carefully before contributing. PRs that do not follow these rules may be closed without review.**

-   ❌ Do NOT submit incomplete or untested features
-   ❌ Do NOT commit `.env` files or secrets
-   ❌ Do NOT hardcode API keys, tokens, or credentials
-   ❌ Do NOT push directly to the `main` branch
-   ❌ Do NOT introduce breaking changes without documentation
-   ✅ Create a new branch for every feature or fix
-   ✅ Ensure your code runs without errors
-   ✅ Follow the existing project structure
-   ✅ Add meaningful commit messages
-   ✅ Update documentation when necessary

------------------------------------------------------------------------

## 📌 Issue Policy

-   Check existing issues before creating a new one
-   Clearly describe bugs with reproduction steps
-   For feature requests, explain the use case and expected behavior
-   Keep issues focused and well-scoped

------------------------------------------------------------------------

## 🛠 Tech Stack

This project uses:

-   **Language**: Python 3.9+
-   **LLM Integration**: HuggingFace Transformers (Mistral / LLaMA / GPT-compatible models)
-   **NLP**: NLTK, spaCy
-   **Dashboard**: Streamlit
-   **Data Processing**: Pandas, NumPy
-   **Configuration**: YAML
-   **CLI Interface**: argparse
-   **Exports**: Excel, CSV, JSON

------------------------------------------------------------------------

## ✅ Prerequisites

Ensure you have:

-   Python 3.9 or higher
-   pip
-   Git
-   8GB RAM minimum (16GB recommended for LLM mode)
-   Optional: GPU with CUDA for faster inference

------------------------------------------------------------------------

## 🚀 Getting Started

### Step 1: Fork the Repository

Fork the repository to your GitHub account.

### Step 2: Clone Your Fork

``` bash
git clone https://github.com/YOUR-USERNAME/REPOSITORY-NAME.git
cd REPOSITORY-NAME
```

### Step 3: Create Virtual Environment

``` bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### Step 4: Install Dependencies

``` bash
pip install -r requirements.txt
```

### Step 5: Download NLP Resources

``` bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
python -m spacy download en_core_web_sm
```

### Step 6: Configure Environment

``` bash
cp .env.example .env
```

Edit `.env` only if required.

⚠️ **Never commit your `.env` file.**

------------------------------------------------------------------------

## 🌿 Branch Naming Convention

Always create a new branch before starting work:

``` bash
git checkout -b feature/your-feature-name
```

Branch prefixes:

-   `feature/` -- new feature
-   `fix/` -- bug fix
-   `docs/` -- documentation changes
-   `refactor/` -- structural improvements
-   `test/` -- adding or improving tests
-   `chore/` -- maintenance updates

------------------------------------------------------------------------

## 💻 Development Workflow

### 1. Understand the Codebase

Project structure:

    src/
      preprocessing.py
      llm_extractor.py
      risk_scoring.py
      fmea_generator.py
      utils.py

Follow existing modular design principles.

------------------------------------------------------------------------

### 2. Make Changes

-   Maintain modular architecture
-   Keep logic separated by responsibility
-   Avoid tight coupling
-   Ensure rule-based fallback remains functional

------------------------------------------------------------------------

### 3. Test Your Changes

Before submitting a PR:

``` bash
python examples.py
streamlit run app.py
python cli.py --text sample.csv --output test.xlsx
```

Verify:

-   No runtime errors
-   Outputs generate correctly
-   Export formats work
-   LLM and rule-based modes both function

------------------------------------------------------------------------

### 4. Commit Your Changes

``` bash
git add .
git commit -m "feat: add hybrid risk prioritization logic"
```

Commit format:

-   `feat:` -- new feature
-   `fix:` -- bug fix
-   `docs:` -- documentation
-   `refactor:` -- restructuring
-   `test:` -- testing improvements
-   `chore:` -- maintenance

------------------------------------------------------------------------

### 5. Push to Your Fork

``` bash
git push origin feature/your-feature-name
```

------------------------------------------------------------------------

### 6. Create a Pull Request

Your PR must include:

-   Clear title
-   Detailed description of changes
-   Screenshots (if dashboard UI changes)
-   Explanation of impact on architecture
-   Mention if config.yaml requires modification

------------------------------------------------------------------------

## 🔄 Pull Request Requirements

PRs must:

- Run without errors
- Not break CLI functionality
- Not break Streamlit dashboard
- Maintain compatibility with rule-based mode
- Avoid committing large model files
- Follow existing coding style

Maintainers may request revisions before merging.

------------------------------------------------------------------------

## 🧪 Testing Guidelines

When adding features:

-   Validate LLM mode
-   Validate rule-based fallback
-   Test structured input
-   Test unstructured input
-   Test hybrid mode
-   Confirm export integrity

------------------------------------------------------------------------

## 🔐 Security Guidelines

-   Never commit API keys
-   Never upload trained model weights
-   Do not expose local file paths
-   Avoid logging sensitive data

------------------------------------------------------------------------

## 🆘 Need Help?

-   Review README documentation
-   Run `examples.py` for reference
-   Check configuration in `config/config.yaml`
-   Open a GitHub issue with clear details

------------------------------------------------------------------------

## 🎯 Contribution Areas

We welcome contributions in:

-   LLM prompt engineering improvements
-   Risk scoring enhancements
-   Performance optimization
-   GPU optimization
-   Model quantization strategies
-   Additional export formats
-   Advanced analytics modules
-   Multi-language support

------------------------------------------------------------------------

## 📜 Code of Conduct

Be respectful and professional in all communications. Constructive feedback is encouraged.

------------------------------------------------------------------------

## 🚀 Future Enhancements

-   Domain-specific fine-tuned models
-   Real-time streaming inputs
-   Enterprise integration APIs
-   Advanced visualization modules
-   Benchmarking framework

------------------------------------------------------------------------

**Thank you for contributing to the LLM-Powered Automated FMEA Generator!**

Transforming failure analysis with AI.
