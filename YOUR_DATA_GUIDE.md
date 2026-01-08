# Working with YOUR Datasets - Quick Guide

## 📊 Your Data Structure

### 1. Structured Data: `FMEA.csv`
**Location:** `d:/Symboisis/FMEA.csv`  
**Records:** 161 failure modes  
**Columns:**
- Component ID
- Component (Gearbox, Hydraulic cylinder, Bearing, etc.)
- Failure Mode (Gearbox oil leakage, etc.)
- Cause
- Effect
- Maintenance Strategy (existing controls)
- Severity, Occurrence, Detection (already scored!)
- RPN (pre-calculated)

**This is industrial machinery FMEA data** - perfect for demonstrating the system!

### 2. Unstructured Data: `archive (3)/`
**Location:** `d:/Symboisis/archive (3)/`  
**Files:** 50 CSV files with car reviews  
**Brands:** Ford, Toyota, Honda, BMW, Tesla, Mercedes, and 40+ more  
**Total Reviews:** ~100,000+ customer reviews  
**Columns:**
- Review_Date
- Author_Name
- Vehicle_Title
- Review_Title
- **Review** (main text we'll analyze)
- Rating

---

## 🚀 QUICK START - 3 Ways to Process YOUR Data

### ⚡ **FASTEST: Use the Custom Script**

```bash
python process_my_data.py
```

This automatically:
- ✅ Processes your FMEA.csv (structured data)
- ✅ Analyzes car reviews from archive (3)
- ✅ Creates hybrid analysis combining both
- ✅ Exports everything to `output/` folder

**Output files:**
- `YOUR_STRUCTURED_FMEA.xlsx` - Enhanced FMEA.csv
- `YOUR_FORD_FMEA.xlsx` - Ford car review analysis
- `YOUR_TOYOTA_FMEA.xlsx` - Toyota review analysis
- `YOUR_HONDA_FMEA.xlsx` - Honda review analysis
- `YOUR_HYBRID_FMEA.xlsx` - Combined analysis
- `YOUR_HYBRID_FMEA_SUMMARY.txt` - Text summary

---

### 🎨 **INTERACTIVE: Use the Dashboard**

```bash
streamlit run app.py
```

Then:
1. **For structured data:**
   - Click "Structured File (CSV/Excel)"
   - Upload `FMEA.csv`
   - Click "Generate FMEA"

2. **For car reviews:**
   - Click "Unstructured Text"
   - Upload any file from `archive (3)/`
   - Click "Generate FMEA"

3. **For hybrid:**
   - Click "Hybrid (Both)"
   - Upload `FMEA.csv` + any car review file
   - Click "Generate Hybrid FMEA"

---

### 💻 **PROGRAMMABLE: Use Python Code**

#### Process Your Structured FMEA.csv:
```python
from fmea_generator import FMEAGenerator
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize
generator = FMEAGenerator(config)

# Process YOUR FMEA.csv
fmea_df = generator.generate_from_structured('FMEA.csv')

# Export
generator.export_fmea(fmea_df, 'output/my_enhanced_fmea.xlsx')
```

#### Process Your Car Reviews:
```python
# Process Ford reviews
fmea_df = generator.generate_from_text(
    'archive (3)/Scraped_Car_Review_ford.csv', 
    is_file=True
)

# Export
generator.export_fmea(fmea_df, 'output/ford_failures.xlsx')
```

#### Process Multiple Brands:
```python
from pathlib import Path

# Process all car brands
for review_file in Path('archive (3)').glob('*.csv'):
    brand_name = review_file.stem.replace('Scraped_Car_Review_', '')
    
    print(f"Processing {brand_name}...")
    fmea_df = generator.generate_from_text(str(review_file), is_file=True)
    
    output_path = f'output/{brand_name}_FMEA.xlsx'
    generator.export_fmea(fmea_df, output_path)
    print(f"Saved to {output_path}")
```

---

## 📋 Specific Examples for YOUR Data

### Example 1: Find Critical Issues in Your Industrial FMEA

```python
import pandas as pd
from fmea_generator import FMEAGenerator
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

generator = FMEAGenerator(config)
fmea_df = generator.generate_from_structured('FMEA.csv')

# Find all critical items
critical = fmea_df[fmea_df['Action Priority'] == 'Critical']
print(f"Found {len(critical)} critical issues")

# Show highest RPN items
top_risks = fmea_df.nlargest(10, 'Rpn')
print(top_risks[['Component', 'Failure Mode', 'Rpn']])
```

### Example 2: Compare Failure Patterns Across Car Brands

```python
from pathlib import Path

brands_to_compare = ['ford', 'toyota', 'honda', 'bmw']
results = {}

for brand in brands_to_compare:
    file_pattern = f'*{brand}*.csv'
    files = list(Path('archive (3)').glob(file_pattern))
    
    if files:
        fmea_df = generator.generate_from_text(str(files[0]), is_file=True)
        results[brand] = {
            'total_failures': len(fmea_df),
            'critical_count': len(fmea_df[fmea_df['Action Priority'] == 'Critical']),
            'avg_rpn': fmea_df['Rpn'].mean()
        }

# Compare results
import pandas as pd
comparison_df = pd.DataFrame(results).T
print(comparison_df)
```

### Example 3: Extract Specific Component Failures

```python
# From your FMEA.csv, focus on specific components
fmea_df = generator.generate_from_structured('FMEA.csv')

# Filter by component
gearbox_issues = fmea_df[fmea_df['Component'].str.contains('Gearbox', case=False, na=False)]
print(f"\nGearbox-related failures: {len(gearbox_issues)}")
print(gearbox_issues[['Failure Mode', 'Cause', 'Rpn']])

# Filter high RPN items for specific component
high_risk_hydraulic = fmea_df[
    (fmea_df['Component'].str.contains('Hydraulic', case=False, na=False)) & 
    (fmea_df['Rpn'] > 100)
]
print(f"\nHigh-risk hydraulic failures: {len(high_risk_hydraulic)}")
```

### Example 4: Analyze Tesla Reviews for Safety Issues

```python
# Process Tesla reviews specifically
tesla_file = 'archive (3)/Scraped_Car_Review_tesla.csv'

fmea_df = generator.generate_from_text(tesla_file, is_file=True)

# Filter for safety-related failures
safety_keywords = ['brake', 'airbag', 'crash', 'accident', 'steering', 'safety']
safety_issues = fmea_df[
    fmea_df['Failure Mode'].str.contains('|'.join(safety_keywords), case=False, na=False)
]

print(f"Found {len(safety_issues)} safety-related issues in Tesla reviews")
print(safety_issues[['Failure Mode', 'Effect', 'Rpn', 'Action Priority']])
```

---

## 🎯 Understanding Your Results

### Your FMEA.csv Already Has Scores
Your structured data already includes Severity, Occurrence, Detection, and RPN scores. The system will:
- ✅ Validate these scores
- ✅ Add Action Priority classification
- ✅ Generate recommended actions
- ✅ Create visualizations

### Car Reviews Need Extraction
Your car reviews are raw text. The system will:
- 🔍 Extract failure modes from review text
- 📊 Calculate S, O, D scores automatically
- 🎯 Compute RPN and prioritize
- 💡 Generate recommendations

---

## 📊 What You'll Get

### From FMEA.csv Processing:
- Enhanced FMEA with recommendations
- Risk priority classification
- Component-wise analysis
- Visual risk matrices

### From Car Reviews Processing:
- Extracted failure modes from customer complaints
- Severity analysis of customer issues
- Frequency patterns across reviews
- Brand-specific failure trends

### From Hybrid Analysis:
- Comprehensive risk view (industrial + customer)
- Cross-validation of failure modes
- Holistic risk assessment
- Combined recommendations

---

## ⚙️ Customization for Your Data

### Adjust Processing Speed

Edit `config/config.yaml`:
```yaml
text_processing:
  max_reviews_per_batch: 100  # Increase for more reviews
  enable_sentiment_filter: true  # Focus on negative reviews
  negative_threshold: 0.3  # Lower = more reviews included
```

### Focus on Specific Components

```python
# Filter your FMEA.csv before processing
import pandas as pd

df = pd.read_csv('FMEA.csv')
critical_components = ['Gearbox', 'Hydraulic cylinder', 'Bearing']
filtered_df = df[df['Component'].isin(critical_components)]
filtered_df.to_csv('temp_filtered.csv', index=False)

# Then process
fmea_df = generator.generate_from_structured('temp_filtered.csv')
```

### Process Specific Car Brands

```python
# Create list of brands you want
brands = ['ford', 'toyota', 'honda', 'bmw', 'tesla']

for brand in brands:
    files = list(Path('archive (3)').glob(f'*{brand}*.csv'))
    if files:
        print(f"Processing {brand}...")
        fmea_df = generator.generate_from_text(str(files[0]), is_file=True)
        generator.export_fmea(fmea_df, f'output/{brand}_analysis.xlsx')
```

---

## 💡 Pro Tips

1. **Start Small**: Process 100 reviews first, then increase batch size
2. **Use Rule-Based Mode**: Faster processing, still accurate
3. **Filter Critical**: Focus on Critical/High priority items first
4. **Compare Brands**: Process multiple car brands to find patterns
5. **Combine Sources**: Hybrid analysis gives best insights

---

## 🐛 Troubleshooting

### "File not found" error
```bash
# Verify files exist
dir FMEA.csv
dir "archive (3)"
```

### Processing is slow
```python
# Use rule-based mode (faster)
config['model']['name'] = None

# Reduce batch size
config['text_processing']['max_reviews_per_batch'] = 50
```

### Out of memory
```python
# Process brands one at a time
# Reduce batch size in config
# Close other applications
```

---

## 🚀 Ready to Start?

**Recommended first command:**
```bash
python process_my_data.py
```

This will analyze YOUR data and create all outputs in ~5-10 minutes!

---

**Questions?** Check [README.md](README.md) for full documentation.
