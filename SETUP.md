# Setup Guide for FMEA Generator

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [First Run](#first-run)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.9 or higher
- **RAM**: 8GB
- **Disk Space**: 5GB
- **Internet**: Required for model download (first time only)

### Recommended Requirements
- **RAM**: 16GB or higher
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for faster LLM inference)
- **Disk Space**: 10GB

---

## Installation Steps

### Step 1: Install Python

Download and install Python 3.9+ from [python.org](https://www.python.org/downloads/)

Verify installation:
```bash
python --version
```

### Step 2: Clone or Download Project

```bash
# If using Git
git clone <repository-url>
cd Symboisis

# Or download and extract ZIP file
```

### Step 3: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages. May take 5-10 minutes.

### Step 5: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

### Step 6: (Optional) Install spaCy Model

For enhanced NLP capabilities:
```bash
python -m spacy download en_core_web_sm
```

### Step 7: Setup Environment Variables

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Edit `.env` file if needed (optional for basic usage).

---

## Configuration

### Basic Configuration

The default configuration in `config/config.yaml` works out of the box.

### Model Selection

Edit `config/config.yaml` to change the LLM model:

```yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.2"  # Default
  # name: "meta-llama/Llama-2-7b-chat-hf"     # Alternative
  # name: null                                  # Rule-based (no LLM)
```

### For Limited Resources

If you have limited RAM or no GPU:

```yaml
model:
  name: null  # Use rule-based extraction (faster, less accurate)
  device: "cpu"
  quantization: true
```

---

## First Run

### Option 1: Run Examples (Recommended)

```bash
python examples.py
```

This will:
- Test all system components
- Generate sample FMEAs
- Create output files in `output/` folder

Press Enter to run each example.

### Option 2: Launch Dashboard

```bash
streamlit run app.py
```

Then open browser to: `http://localhost:8501`

### Option 3: Command Line Test

```bash
# Create a test file
echo "The brakes failed completely. Very dangerous situation." > test_review.txt

# Generate FMEA
python cli.py --text test_review.txt --output test_fmea.xlsx --no-model
```

---

## Troubleshooting

### Issue: "Python not found"

**Solution:**
- Verify Python is installed: `python --version`
- On Linux/Mac, try `python3` instead of `python`
- Ensure Python is added to PATH

### Issue: "pip install fails"

**Solution:**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install with verbose output to see errors
pip install -r requirements.txt -v
```

### Issue: "Out of memory when loading model"

**Solution:**

1. Use rule-based mode (no LLM):
   ```bash
   python cli.py --text input.csv --output result.xlsx --no-model
   ```

2. Or enable quantization in `config/config.yaml`:
   ```yaml
   model:
     quantization: true
     device: "cpu"
   ```

### Issue: "Model download is very slow"

**Solution:**

The first run downloads the model (~5-15GB). This only happens once.

Alternative: Use rule-based mode which doesn't require model download:
```yaml
model:
  name: null
```

### Issue: "CUDA/GPU not detected"

**Solution:**

The system will automatically fall back to CPU. To explicitly use CPU:
```yaml
model:
  device: "cpu"
```

### Issue: "Streamlit command not found"

**Solution:**
```bash
# Ensure virtual environment is activated
pip install streamlit

# Or run with full path
python -m streamlit run app.py
```

### Issue: "NLTK data not found"

**Solution:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Issue: "Import errors"

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

### Issue: "Port 8501 already in use"

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

## Verifying Installation

Run this verification script:

```bash
python -c "
import sys
print('Python version:', sys.version)

try:
    import pandas
    print('✓ pandas installed')
except ImportError:
    print('✗ pandas not found')

try:
    import streamlit
    print('✓ streamlit installed')
except ImportError:
    print('✗ streamlit not found')

try:
    import transformers
    print('✓ transformers installed')
except ImportError:
    print('✗ transformers not found')

try:
    import torch
    print('✓ torch installed')
    print('  CUDA available:', torch.cuda.is_available())
except ImportError:
    print('✗ torch not found')

print('\nSetup verification complete!')
"
```

Expected output:
```
Python version: 3.9.x
✓ pandas installed
✓ streamlit installed
✓ transformers installed
✓ torch installed
  CUDA available: True/False

Setup verification complete!
```

---

## Performance Optimization

### For CPU-Only Systems

```yaml
# config/config.yaml
model:
  device: "cpu"
  quantization: true
  
text_processing:
  max_reviews_per_batch: 50  # Reduce batch size
```

### For GPU Systems

```yaml
# config/config.yaml
model:
  device: "cuda"
  quantization: true  # Still recommended to save VRAM
  
text_processing:
  max_reviews_per_batch: 100
```

### For Fast Processing (Rule-Based)

```yaml
# config/config.yaml
model:
  name: null  # Disable LLM, use rule-based
```

Speed: ~50x faster than LLM
Accuracy: Medium (still useful for many cases)

---

## Next Steps

After successful installation:

1. **Run examples**: `python examples.py`
2. **Try the dashboard**: `streamlit run app.py`
3. **Read the README**: `README.md`
4. **Explore the code**: Start with `src/fmea_generator.py`
5. **Process your data**: Use CLI or dashboard

---

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review error messages carefully
3. Check Python and package versions
4. Try rule-based mode (`--no-model`)
5. Create an issue on GitHub with:
   - Error message
   - Python version
   - OS information
   - Steps to reproduce

---

## Uninstallation

To remove the project:

```bash
# Deactivate virtual environment
deactivate

# Delete project folder
# Windows: rmdir /s Symboisis
# Linux/Mac: rm -rf Symboisis
```

---

**Setup complete! You're ready to generate FMEAs.**

For usage instructions, see [README.md](README.md)
