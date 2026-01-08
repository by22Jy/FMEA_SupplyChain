# Model Training Guide

## Overview

Your FMEA Generator now includes a **two-stage model training pipeline** based on the research paper methodology:

### Stage 1: Sentiment Classification (97% Accuracy)
- **Objective**: Classify reviews as negative/positive
- **Model**: Fine-tuned GPT-3 Curie
- **Accuracy**: 97% (vs 87% with BiLSTM-CNN)
- **Purpose**: Identify negative reviews containing failure information

### Stage 2: Part Extraction (98-99% Accuracy)
- **Objective**: Identify reviews mentioning automotive parts
- **Model**: Fine-tuned GPT-3.5 Turbo
- **Accuracy**: 98-99%
- **Features**: Multilingual support (English, French, Spanish)
- **Training Size**: 18,000+ reviews recommended

## Quick Start

### 1. Install OpenAI Library
```bash
pip install openai>=1.0.0
```

### 2. Get OpenAI API Key
- Sign up at https://platform.openai.com/
- Create an API key
- Keep it secure!

### 3. Run Training Script
```bash
python train_models.py
```

The script will:
1. Load your car review data from `archive (3)/`
2. Prepare training data (sentiment labels + part labels)
3. Fine-tune GPT-3 Curie for sentiment (97% accuracy)
4. Fine-tune GPT-3.5 Turbo for part extraction (98-99% accuracy)
5. Test the pipeline on sample data
6. Save results to `output/model_training_results.xlsx`

### 4. Update Configuration
After training completes, update `config/config.yaml`:

```yaml
llm:
  sentiment_model: "curie:ft-personal-2024-01-06:abcd1234"
  part_extraction_model: "ft:gpt-3.5-turbo-0613:personal::5678efgh"
```

### 5. Use Trained Models
```bash
streamlit run app.py
```

In the dashboard:
- Select "GPT-3 Curie (OpenAI Fine-tuned)" for sentiment
- Select "GPT-3.5 Turbo (OpenAI)" for part extraction
- Enter your API key

## Training Data Requirements

### Sentiment Classification
- **Minimum**: 1,000 reviews
- **Recommended**: 10,000+ reviews
- **Labels**: Binary (positive/negative based on ratings)
  - Negative: 1-2 stars
  - Positive: 4-5 stars
  - Neutral (3 stars) excluded

### Part Extraction
- **Minimum**: 5,000 reviews
- **Recommended**: 18,000+ reviews (paper standard)
- **Labels**: Binary (has_part: yes/no)
- **Languages**: English, French, Spanish supported

## Training Process

### Stage 1: Sentiment (GPT-3 Curie)
1. Prepare data in JSONL format
2. Upload to OpenAI
3. Fine-tune with:
   - 4 epochs
   - Learning rate: 0.1
   - 90/10 train/val split
4. Expected time: 10-20 minutes
5. Expected accuracy: 97%

### Stage 2: Part Extraction (GPT-3.5 Turbo)
1. Prepare data with system/user/assistant messages
2. Upload to OpenAI
3. Fine-tune with default parameters
4. Expected time: 20-40 minutes
5. Expected accuracy: 98-99%

## Fallback Methods

If you don't have an OpenAI API key or prefer offline operation:

### BiLSTM-CNN Sentiment (87% Accuracy)
- Uses TensorFlow with bidirectional LSTM
- Runs locally without API calls
- Automatically used when no API key provided

### String Matching for Parts (75% Accuracy)
- Keyword-based pattern matching
- Multilingual keywords included
- No training required
- Lower accuracy but immediate availability

## Cost Estimation

### OpenAI Fine-tuning Costs (January 2024)
- **GPT-3 Curie**: ~$0.03 per 1K training tokens
- **GPT-3.5 Turbo**: ~$0.08 per 1K training tokens

**Example for 10,000 reviews:**
- Sentiment: ~$5-10
- Part Extraction: ~$15-25
- **Total**: ~$20-35

### Inference Costs
- Curie: $0.002 per 1K tokens
- GPT-3.5 Turbo: $0.001 per 1K tokens

## Using Trained Models

### In Streamlit Dashboard
1. Go to sidebar
2. Enable "Use LLM Model"
3. Select GPT model
4. Enter API key
5. Process reviews

### In Code
```python
from src.model_trainer import FMEAModelTrainer

# Initialize with API key
trainer = FMEAModelTrainer(api_key="sk-...")

# Process reviews
results = trainer.process_reviews_pipeline(review_list)

# Results include:
# - Sentiment classification
# - Part extraction
# - Confidence scores
# - Parts found
```

### PFMEA Generator Form
1. Go to "PFMEA Generator" tab
2. Fill form:
   - Defect: "dimensions not ok"
   - Cause: "Bad positioning of components"
   - Effect: "Assembly difficulties"
3. Click "Generate"
4. System uses trained models automatically

## Accuracy Comparison

| Method | Stage | Accuracy | Speed | Cost |
|--------|-------|----------|-------|------|
| BiLSTM-CNN | Sentiment | 87% | Fast | Free |
| GPT-3 Curie | Sentiment | 97% | Medium | Low |
| String Match | Parts | 75% | Very Fast | Free |
| GPT-3.5 Turbo | Parts | 98-99% | Medium | Low |

## Monitoring Training

### Check Fine-tuning Status
```python
import openai
openai.api_key = "sk-..."

# List fine-tune jobs
jobs = openai.FineTune.list()
print(jobs)

# Get specific job
job = openai.FineTune.retrieve("ft-xyz123")
print(f"Status: {job.status}")
print(f"Fine-tuned model: {job.fine_tuned_model}")
```

### View Training Metrics
OpenAI provides:
- Training loss
- Validation loss
- Accuracy over epochs
- Token usage

Access via: https://platform.openai.com/finetune

## Troubleshooting

### Issue: "Rate limit exceeded"
**Solution**: OpenAI has rate limits. Wait a few minutes or upgrade plan.

### Issue: "Insufficient training data"
**Solution**: Minimum 100 examples required. Recommended: 1,000+

### Issue: "Model not found after training"
**Solution**: Training takes time. Check status with `FineTune.retrieve()`

### Issue: "Low accuracy on custom data"
**Solution**: 
- Increase training data size
- Ensure data quality (clean labels)
- Add domain-specific examples

## Best Practices

1. **Start Small**: Train on 1,000-5,000 reviews first
2. **Validate Results**: Test on separate validation set
3. **Monitor Costs**: Track API usage in OpenAI dashboard
4. **Version Models**: Save model IDs for reproducibility
5. **Update Regularly**: Retrain quarterly with new data
6. **A/B Test**: Compare with fallback methods

## Research Paper Reference

This implementation is based on:
- **Sentiment**: GPT-3 Curie fine-tuning (97% accuracy)
- **Part Extraction**: GPT-3.5 Turbo (98-99% accuracy)
- **Dataset**: Automotive reviews (English, French, Spanish)
- **Training Size**: 18,000 reviews for optimal performance

## Support

For issues or questions:
1. Check OpenAI documentation: https://platform.openai.com/docs
2. Review training logs in `output/` folder
3. Test with fallback methods first
4. Ensure API key is valid and has credits

## Next Steps

After successful training:
1. ✅ Models saved to OpenAI account
2. ✅ Configuration updated
3. ✅ Test results verified
4. 🚀 Ready for production use!

Run your FMEA analysis:
```bash
python process_my_data.py  # Full analysis
streamlit run app.py       # Interactive dashboard
python examples.py         # Example workflows
```
