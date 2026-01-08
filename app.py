"""
Streamlit Dashboard for FMEA Generator
Interactive web interface for generating and analyzing FMEA
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import yaml
import sys
from datetime import datetime
import logging
from PIL import Image
import io

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from fmea_generator import FMEAGenerator
from preprocessing import DataPreprocessor
from llm_extractor import LLMExtractor
from risk_scoring import RiskScoringEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LLM-Powered FMEA Generator",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stDataFrame {
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load configuration from YAML file"""
    config_path = Path('config/config.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        st.error("Configuration file not found!")
        return {}


@st.cache_resource
def initialize_generator(_config):
    """Initialize FMEA Generator (cached to avoid reloading model)"""
    return FMEAGenerator(_config)


def display_metrics(fmea_df):
    """Display key metrics from FMEA"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Failure Modes",
            value=len(fmea_df)
        )
    
    with col2:
        critical_count = len(fmea_df[fmea_df['Action Priority'] == 'Critical'])
        st.metric(
            label="Critical Issues",
            value=critical_count,
            delta="Needs Immediate Action" if critical_count > 0 else None
        )
    
    with col3:
        avg_rpn = fmea_df['Rpn'].mean()
        st.metric(
            label="Average RPN",
            value=f"{avg_rpn:.1f}"
        )
    
    with col4:
        max_rpn = fmea_df['Rpn'].max()
        st.metric(
            label="Maximum RPN",
            value=int(max_rpn)
        )


def plot_rpn_distribution(fmea_df):
    """Plot RPN distribution"""
    fig = px.histogram(
        fmea_df,
        x='Rpn',
        nbins=30,
        title='RPN Distribution',
        labels={'Rpn': 'Risk Priority Number', 'count': 'Frequency'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_priority_distribution(fmea_df):
    """Plot action priority distribution"""
    priority_counts = fmea_df['Action Priority'].value_counts()
    
    colors = {
        'Critical': '#d62728',
        'High': '#ff7f0e',
        'Medium': '#ffbb78',
        'Low': '#2ca02c'
    }
    
    fig = go.Figure(data=[
        go.Pie(
            labels=priority_counts.index,
            values=priority_counts.values,
            marker=dict(colors=[colors.get(p, '#1f77b4') for p in priority_counts.index])
        )
    ])
    fig.update_layout(title='Action Priority Distribution')
    return fig


def plot_risk_matrix(fmea_df):
    """Plot risk matrix (Severity vs Occurrence)"""
    fig = px.scatter(
        fmea_df,
        x='Occurrence',
        y='Severity',
        size='Rpn',
        color='Action Priority',
        hover_data=['Failure Mode', 'Effect'],
        title='Risk Matrix (Severity vs Occurrence)',
        color_discrete_map={
            'Critical': '#d62728',
            'High': '#ff7f0e',
            'Medium': '#ffbb78',
            'Low': '#2ca02c'
        }
    )
    fig.update_xaxes(range=[0, 11])
    fig.update_yaxes(range=[0, 11])
    return fig


def plot_top_risks(fmea_df, top_n=10):
    """Plot top N risks by RPN"""
    top_risks = fmea_df.nlargest(top_n, 'Rpn')
    
    fig = px.bar(
        top_risks,
        x='Rpn',
        y='Failure Mode',
        orientation='h',
        title=f'Top {top_n} Risks by RPN',
        color='Action Priority',
        color_discrete_map={
            'Critical': '#d62728',
            'High': '#ff7f0e',
            'Medium': '#ffbb78',
            'Low': '#2ca02c'
        }
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def extract_text_from_image(image_file):
    """Extract text from image using OCR"""
    try:
        # Try EasyOCR first (no external dependencies needed)
        try:
            import easyocr
            from PIL import Image as PILImage
            
            # Create reader (cached for performance)
            if 'easyocr_reader' not in st.session_state:
                with st.spinner("Initializing OCR engine (first time only)..."):
                    st.session_state.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            
            reader = st.session_state.easyocr_reader
            
            # Save image temporarily
            img = PILImage.open(image_file)
            temp_img_path = "temp_ocr_image.png"
            img.save(temp_img_path)
            
            # Extract text
            results = reader.readtext(temp_img_path)
            text = '\n'.join([result[1] for result in results])
            
            # Clean up
            Path(temp_img_path).unlink(missing_ok=True)
            
            return text if text.strip() else "No text found in image"
            
        except ImportError:
            # Fallback to pytesseract if easyocr not available
            try:
                import pytesseract
                from PIL import Image as PILImage
                
                # Open image
                img = PILImage.open(image_file)
                
                # Extract text
                text = pytesseract.image_to_string(img)
                
                if text.strip():
                    return text
                else:
                    return "No text found in image"
                    
            except Exception as e:
                return f"OCR libraries not properly configured. Error: {str(e)}\n\nPlease install: pip install easyocr"
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error extracting text from image: {str(e)}\n\nDetails: {error_details}"


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">⚠️ LLM-Powered FMEA Generator</div>', 
                unsafe_allow_html=True)
    st.markdown("### Automated Failure Mode and Effects Analysis from Structured & Unstructured Data")
    
    # Load configuration
    config = load_config()
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=FMEA+Generator", 
                use_column_width=True)
        st.markdown("---")
        
        st.markdown("### 📊 Input Options")
        input_type = st.radio(
            "Select Input Type:",
            ["Unstructured Text", "Structured File (CSV/Excel)", "Hybrid (Both)"]
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        model_options = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
            "Rule-based (No LLM)"
        ]
        selected_model = st.selectbox("Model Selection:", model_options)
        
        if selected_model == "Rule-based (No LLM)":
            config['model']['name'] = None
        else:
            config['model']['name'] = selected_model
        
        # Output format
        output_format = st.selectbox("Export Format:", ["Excel", "CSV"])
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.info("""
        This system uses LLMs to automatically generate FMEA from:
        - Customer reviews
        - Complaint reports
        - Structured failure data
        
        **Features:**
        - Intelligent extraction
        - Automated risk scoring
        - Visual analytics
        - Export capabilities
        """)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Generate FMEA", "🎯 PFMEA Generator", "📊 Analytics", "ℹ️ Help"])
    
    with tab1:
        st.markdown('<div class="sub-header">Generate FMEA</div>', unsafe_allow_html=True)
        
        if input_type == "Unstructured Text":
            text_input_method = st.radio(
                "Input Method:",
                ["Upload File", "Enter Text Manually"]
            )
            
            if text_input_method == "Upload File":
                uploaded_file = st.file_uploader(
                    "Upload image file (PNG, JPEG) - OCR will extract text",
                    type=['png', 'jpg', 'jpeg']
                )
                
                if uploaded_file:
                    # Display uploaded image
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                    
                    with col2:
                        if st.button("🚀 Extract Text & Generate FMEA", type="primary"):
                            with st.spinner("Extracting text from image..."):
                                # Extract text using OCR
                                extracted_text = extract_text_from_image(uploaded_file)
                                
                                # Show extracted text
                                st.markdown("**Extracted Text:**")
                                st.text_area("", extracted_text, height=150, key="extracted", disabled=True)
                                
                                if "Error" not in extracted_text and "No text found" not in extracted_text:
                                    with st.spinner("Generating FMEA from extracted text..."):
                                        generator = initialize_generator(config)
                                        # Split text into lines
                                        texts = [line.strip() for line in extracted_text.split('\n') if line.strip()]
                                        fmea_df = generator.generate_from_text(texts, is_file=False)
                                        st.session_state['fmea_df'] = fmea_df
                                else:
                                    st.error(extracted_text)
            else:
                text_input = st.text_area(
                    "Enter text (reviews, reports, complaints):",
                    height=200,
                    placeholder="Paste customer reviews, failure reports, or complaint text here..."
                )
                
                if text_input and st.button("🚀 Generate FMEA", type="primary"):
                    with st.spinner("Analyzing text and generating FMEA..."):
                        generator = initialize_generator(config)
                        texts = [line.strip() for line in text_input.split('\n') if line.strip()]
                        fmea_df = generator.generate_from_text(texts, is_file=False)
                        st.session_state['fmea_df'] = fmea_df
        
        elif input_type == "Structured File (CSV/Excel)":
            uploaded_file = st.file_uploader(
                "Upload structured FMEA file (CSV or Excel)",
                type=['csv', 'xlsx', 'xls']
            )
            
            if uploaded_file:
                temp_path = Path(f"temp_{uploaded_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if st.button("🚀 Generate FMEA", type="primary"):
                    with st.spinner("Processing structured data..."):
                        generator = initialize_generator(config)
                        fmea_df = generator.generate_from_structured(str(temp_path))
                        st.session_state['fmea_df'] = fmea_df
                    
                    temp_path.unlink()
        
        else:  # Hybrid
            st.markdown("**Upload both structured and unstructured data:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Structured Data:**")
                structured_file = st.file_uploader(
                    "Upload CSV/Excel",
                    type=['csv', 'xlsx', 'xls'],
                    key='structured'
                )
            
            with col2:
                st.markdown("**Unstructured Data:**")
                unstructured_text = st.text_area(
                    "Enter text manually (reviews, reports, complaints):",
                    height=200,
                    placeholder="Paste customer reviews, failure reports, or complaint text here...",
                    key='hybrid_text'
                )
            
            if (structured_file or unstructured_text) and st.button("🚀 Generate Hybrid FMEA", type="primary"):
                with st.spinner("Processing hybrid data..."):
                    generator = initialize_generator(config)
                    
                    structured_path = None
                    text_data = None
                    
                    if structured_file:
                        structured_path = Path(f"temp_structured_{structured_file.name}")
                        with open(structured_path, "wb") as f:
                            f.write(structured_file.getbuffer())
                    
                    if unstructured_text:
                        # Convert text to list of lines
                        text_data = [line.strip() for line in unstructured_text.split('\n') if line.strip()]
                    
                    fmea_df = generator.generate_hybrid(
                        structured_file=str(structured_path) if structured_path else None,
                        text_input=text_data if text_data else None
                    )
                    st.session_state['fmea_df'] = fmea_df
                    
                    # Cleanup
                    if structured_path:
                        structured_path.unlink()
        
        # Display results
        if 'fmea_df' in st.session_state:
            st.success("✅ FMEA Generated Successfully!")
            
            fmea_df = st.session_state['fmea_df']
            
            # Display metrics
            st.markdown("---")
            st.markdown("### 📈 Key Metrics")
            display_metrics(fmea_df)
            
            # Display FMEA table
            st.markdown("---")
            st.markdown("### 📋 FMEA Table")
            
            # Add filtering options
            col1, col2 = st.columns(2)
            with col1:
                priority_filter = st.multiselect(
                    "Filter by Priority:",
                    options=['Critical', 'High', 'Medium', 'Low'],
                    default=['Critical', 'High', 'Medium', 'Low']
                )
            
            with col2:
                rpn_threshold = st.slider("Minimum RPN:", 0, 1000, 0)
            
            filtered_df = fmea_df[
                (fmea_df['Action Priority'].isin(priority_filter)) &
                (fmea_df['Rpn'] >= rpn_threshold)
            ]
            
            st.dataframe(filtered_df, use_container_width=True, height=400)
            
            # Export options
            st.markdown("---")
            st.markdown("### 💾 Export FMEA")
            
            col1, col2 = st.columns(2)
            
            with col1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"FMEA_{timestamp}"
                
                if output_format == "Excel":
                    # Create Excel file in memory
                    output_path = Path(f"output/{filename}.xlsx")
                    output_path.parent.mkdir(exist_ok=True)
                    
                    generator = initialize_generator(config)
                    generator.export_fmea(filtered_df, str(output_path), format='excel')
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Excel",
                            data=f,
                            file_name=f"{filename}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{filename}.csv",
                        mime="text/csv"
                    )
    
    with tab2:
        st.markdown('<div class="sub-header">PFMEA LLM Generator</div>', unsafe_allow_html=True)
        st.markdown("Generate PFMEA records using a form-based approach with automatic prompt generation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Input Form")
            
            # Process context (required)
            process_type = st.text_input(
                "Process Step",
                value="Sealing",
                help="Specify the process step (e.g., Sealing, Welding, Assembly, Painting)"
            )
            
            # Components (allow multiple)
            components_input = st.text_area(
                "Components (one per line)",
                value="Heat Sealer\nPackaging Material",
                help="Enter components involved in this process, one per line",
                height=80
            )
            
            # General failure information
            defect = st.text_input(
                "General Defect/Failure Type",
                value="Improper Seal",
                help="Enter the general defect or fault type"
            )
            
            cause = st.text_area(
                "Potential Causes (one per line)",
                value="Temperature fluctuation\nMaterial variation",
                help="Describe potential causes, one per line",
                height=80
            )
            
            effect = st.text_area(
                "Potential Effects (one per line)",
                value="Product leakage\nReduced shelf life",
                help="Describe potential effects, one per line",
                height=80
            )
            
            generate_btn = st.button("🚀 Generate Multiple Failure Modes", type="primary", use_container_width=True)
            
            st.info("💡 Tip: Enter multiple components, causes, and effects (one per line) to generate comprehensive PFMEA table")
        
        with col2:
            st.markdown("#### Generated Prompt & Results")
            
            if generate_btn:
                # Parse inputs
                components = [c.strip() for c in components_input.split('\n') if c.strip()]
                causes = [c.strip() for c in cause.split('\n') if c.strip()]
                effects = [e.strip() for e in effect.split('\n') if e.strip()]
                
                # Build the prompt dynamically
                prompt_parts = ["Generate PFMEA records"]
                
                if process_type:
                    prompt_parts.append(f"for the {process_type} process")
                
                if defect:
                    prompt_parts.append(f"with failure type '{defect}'")
                
                if components:
                    prompt_parts.append(f"analyzing components: {', '.join(components)}")
                
                prompt_text = " ".join(prompt_parts) + "."
                
                # Display generated prompt
                st.markdown("**Generated prompt:**")
                st.info(prompt_text)
                
                # Generate multiple FMEA records
                with st.spinner("Generating PFMEA records..."):
                    try:
                        generator = initialize_generator(config)
                        
                        # Create FMEA records directly without LLM extraction (use form data as-is)
                        all_fmea_records = []
                        
                        # Strategy: Generate one record per component, pairing with causes/effects
                        for i, component in enumerate(components):
                            # Pair with corresponding cause/effect or cycle through them
                            cause_text = causes[i % len(causes)] if causes else "Not specified"
                            effect_text = effects[i % len(effects)] if effects else "Not specified"
                            
                            # Create FMEA record directly (bypass extraction to preserve exact text)
                            record = {
                                'component': component,
                                'failure_mode': defect,
                                'cause': cause_text,
                                'effect': effect_text,
                                'existing_controls': 'Not specified'
                            }
                            
                            # Score the record
                            scored_record = generator.scorer.score_fmea_row(record)
                            
                            # Create DataFrame
                            record_df = pd.DataFrame([scored_record])
                            
                            # Standardize column names
                            record_df.columns = [col.replace('_', ' ').title() for col in record_df.columns]
                            record_df.rename(columns={
                                'Severity': 'Severity',
                                'Occurrence': 'Occurrence', 
                                'Detection': 'Detection',
                                'Rpn': 'Rpn',
                                'Action Priority': 'Action Priority'
                            }, inplace=True)
                            
                            all_fmea_records.append(record_df)
                        
                        # Combine all records
                        if all_fmea_records:
                            combined_df = pd.concat(all_fmea_records, ignore_index=True)
                            
                            # Add Process Step column if specified
                            if process_type:
                                combined_df.insert(0, 'Process Step', process_type)
                            
                            # Store in session state for Analytics tab
                            st.session_state['fmea_df'] = combined_df
                            
                            st.success(f"✅ Generated {len(combined_df)} PFMEA record(s)")
                            
                            # Display results in a clean table format
                            display_columns = ['Process Step', 'Component', 'Failure Mode', 'Cause', 'Effect', 
                                             'Severity', 'Occurrence', 'Detection', 'Rpn', 'Action Priority']
                            # Only include Process Step if it exists
                            display_columns = [col for col in display_columns if col in combined_df.columns]
                            
                            st.dataframe(
                                combined_df[display_columns],
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Download button
                            output_file = 'output/pfmea_form_generated.xlsx'
                            Path('output').mkdir(exist_ok=True)
                            generator.export_fmea(combined_df, output_file, format='excel')
                            
                            with open(output_file, 'rb') as f:
                                st.download_button(
                                    "📥 Download PFMEA Report",
                                    f,
                                    file_name="pfmea_generated.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        else:
                            st.warning("No PFMEA records generated. Please check your inputs.")
                            
                    except Exception as e:
                        st.error(f"Error generating PFMEA: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.markdown("*Fill the form and click Generate to see results*")
    
    with tab3:
        st.markdown('<div class="sub-header">Analytics & Visualization</div>', unsafe_allow_html=True)
        
        if 'fmea_df' in st.session_state:
            fmea_df = st.session_state['fmea_df']
            
            # RPN Distribution
            st.plotly_chart(plot_rpn_distribution(fmea_df), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Priority Distribution
                st.plotly_chart(plot_priority_distribution(fmea_df), use_container_width=True)
            
            with col2:
                # Risk Matrix
                st.plotly_chart(plot_risk_matrix(fmea_df), use_container_width=True)
            
            # Top Risks
            st.plotly_chart(plot_top_risks(fmea_df), use_container_width=True)
            
            # Statistics
            st.markdown("### 📊 Statistical Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Severity**")
                st.write(fmea_df['Severity'].describe())
            
            with col2:
                st.markdown("**Occurrence**")
                st.write(fmea_df['Occurrence'].describe())
            
            with col3:
                st.markdown("**Detection**")
                st.write(fmea_df['Detection'].describe())
        else:
            st.info("Generate an FMEA first to see analytics.")
    
    with tab4:
        st.markdown('<div class="sub-header">Help & Documentation</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ## 🎯 How to Use This System
        
        ### Input Types:
        
        #### 1. Unstructured Text
        - Upload CSV files with customer reviews
        - Paste complaint text directly
        - System will extract failure modes automatically using LLM
        
        #### 2. Structured Files
        - Upload CSV/Excel with columns:
          - failure_mode (required)
          - effect (required)
          - cause (required)
          - component (optional)
          - existing_controls (optional)
        
        #### 3. Hybrid Mode
        - Combine both structured and unstructured data
        - System intelligently merges and deduplicates
        
        ### Risk Scoring:
        
        - **Severity (S)**: Impact of failure (1-10)
          - 1-3: Minor
          - 4-6: Moderate
          - 7-9: Major
          - 10: Critical/Catastrophic
        
        - **Occurrence (O)**: Likelihood of occurrence (1-10)
          - 1-3: Rare
          - 4-6: Moderate
          - 7-9: Frequent
          - 10: Almost certain
        
        - **Detection (D)**: Likelihood of detection (1-10)
          - 1-3: Almost certain to detect
          - 4-6: Moderate chance
          - 7-9: Low chance
          - 10: Almost impossible
        
        - **RPN = S × O × D** (1-1000)
        
        ### Action Priority:
        - **Critical**: RPN ≥ 500 or S ≥ 9
        - **High**: RPN ≥ 250
        - **Medium**: RPN ≥ 100
        - **Low**: RPN < 100
        
        ### Tips:
        - Focus on Critical and High priority items first
        - Use filters to focus on specific risk levels
        - Export results for team reviews
        - Monitor RPN trends over time
        """)


if __name__ == "__main__":
    main()
