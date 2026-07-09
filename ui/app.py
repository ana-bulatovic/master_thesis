#!/usr/bin/env python3
"""
Streamlit Web UI for NLP Pipeline
Sarcasm Detection, Sentiment Classification & Summarization
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import subprocess

from pipeline.pipeline import NLPPipeline
from llm.ollama_client import OllamaClient
from config import ModelConfig, PipelineConfig
from evaluation.evaluator import Evaluator

CONFIG_PATH = str(PROJECT_ROOT / "config" / "config.yaml")

st.set_page_config(
    page_title="NLP Pipeline - Master's Thesis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    """Streamlit UI for NLP Pipeline"""

    def __init__(self):
        self.available_models = []
        self.setup_session_state()
        self.initialize_pipeline()

    def setup_session_state(self):
        """Initialize session state variables"""
        if 'results' not in st.session_state:
            st.session_state.results = []
        if 'metrics' not in st.session_state:
            st.session_state.metrics = {}
        if 'config_changed' not in st.session_state:
            st.session_state.config_changed = False
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = None

    def initialize_pipeline(self):
        """Initialize pipeline with default config"""
        try:
            st.session_state.pipeline = NLPPipeline(CONFIG_PATH)
        except Exception as e:
            st.warning(f"Could not initialize pipeline: {e}")
            st.session_state.pipeline = None

    @property
    def pipeline(self):
        """Get pipeline from session state"""
        return st.session_state.pipeline
    
    @pipeline.setter
    def pipeline(self, value):
        """Set pipeline in session state"""
        st.session_state.pipeline = value

    def get_available_models(self):
        """Get list of available Ollama models"""
        fallback_models = ["llama2:latest", "mistral", "neural-chat", "tinyllama"]
        try:
            client = OllamaClient(ModelConfig(name="llama2"))
            if client.check_connection():
                models = client.list_models()
                if models:
                    combined = list(dict.fromkeys(models + fallback_models))
                    return combined
                return fallback_models
            else:
                return fallback_models
        except:
            return fallback_models

    def create_sidebar(self):
        """Create sidebar with configuration options"""
        st.sidebar.title("⚙️ Configuration")

        # Model Selection
        st.sidebar.subheader("🤖 Model Selection")
        self.available_models = self.get_available_models()

        selected_model = st.sidebar.selectbox(
            "LLM Model",
            options=self.available_models,
            index=0 if self.available_models else 0,
            help="Select the language model to use for all tasks"
        )

        # Model Parameters
        st.sidebar.subheader("🔧 Model Parameters")
        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more creative, lower values more deterministic"
        )

        max_tokens = st.sidebar.slider(
            "Max Tokens",
            min_value=64,
            max_value=512,
            value=128,
            step=32,
            help="Maximum number of tokens to generate"
        )

        # Prompting Technique
        st.sidebar.subheader("📝 Prompting Technique")
        technique = st.sidebar.selectbox(
            "Technique",
            options=["zero-shot", "few-shot", "chain-of-thought"],
            index=1,  # Default to few-shot
            help="Zero-shot: No examples, Few-shot: With examples, Chain-of-thought: Step-by-step reasoning"
        )

        # Pipeline Components
        st.sidebar.subheader("🔄 Pipeline Components")
        use_sarcasm = st.sidebar.checkbox(
            "Sarcasm Detection",
            value=True,
            help="Detect if reviews contain sarcasm"
        )

        use_sentiment = st.sidebar.checkbox(
            "Sentiment Classification",
            value=True,
            help="Classify sentiment as positive/negative/neutral"
        )

        use_sentiment_for_summary = st.sidebar.checkbox(
            "Sentiment-Aware Summarization",
            value=True,
            help="Use sentiment information in summarization"
        )

        use_sarcasm_for_summary = st.sidebar.checkbox(
            "Sarcasm-Aware Summarization",
            value=True,
            help="Use sarcasm detection information in summarization"
        )

        # Ollama Status
        st.sidebar.subheader("🔗 Ollama Status")
        if self.check_ollama_status():
            st.sidebar.success("✅ Connected")
        else:
            st.sidebar.error("❌ Not Connected")
            st.sidebar.info("Make sure Ollama is running: `ollama serve`")

        # Apply Configuration Button
        if st.sidebar.button("🔄 Apply Configuration", type="primary"):
            self.update_pipeline_config(
                selected_model, technique, temperature, max_tokens,
                use_sarcasm, use_sentiment, use_sentiment_for_summary, use_sarcasm_for_summary
            )
            st.session_state.config_changed = True
            st.rerun()

        # Download selected model on demand
        if st.sidebar.button(f"📥 Download {selected_model}"):
            output = self.download_model(selected_model)
            st.sidebar.text_area(
                "Download Output",
                output,
                height=180,
                help="Ollama output from the model pull command"
            )

        return {
            'model': selected_model,
            'technique': technique,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'use_sarcasm': use_sarcasm,
            'use_sentiment': use_sentiment,
            'use_sentiment_for_summary': use_sentiment_for_summary,
            'use_sarcasm_for_summary': use_sarcasm_for_summary,
        }

    def run_ollama_command(self, args):
        """Run an Ollama CLI command and return output"""
        try:
            result = subprocess.run(
                ["C:\\Users\\ANA\\AppData\\Local\\Programs\\Ollama\\ollama.exe"] + args,
                capture_output=True,
                text=True,
                timeout=600
            )
            output = result.stdout + "\n" + result.stderr
            return output.strip()
        except subprocess.TimeoutExpired as e:
            return f"Command timed out: {e}"
        except Exception as e:
            return f"Failed to run Ollama command: {e}"

    def download_model(self, model: str):
        """Download a model using Ollama CLI"""
        output = self.run_ollama_command(["pull", model])
        return output

    def check_ollama_status(self):
        """Check if Ollama server is running"""
        try:
            client = OllamaClient(ModelConfig(name="llama2"))
            return client.check_connection()
        except:
            return False

    def update_pipeline_config(self, model, technique, temperature, max_tokens,
                             use_sarcasm, use_sentiment, use_sentiment_for_summary,
                             use_sarcasm_for_summary):
        """Update pipeline configuration"""
        # Update config.yaml
        config = {
            'model': {
                'name': model,
                'temperature': temperature,
                'num_predict': max_tokens,
            },
            'pipeline': {
                'sarcasm_model': model,
                'sentiment_model': model,
                'summarization_model': model,
                'prompting_technique': technique,
                'use_sarcasm_detection': use_sarcasm,
                'use_sentiment_for_summarization': use_sentiment_for_summary,
                'use_sarcasm_for_summarization': use_sarcasm_for_summary,
            }
        }

        import yaml
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        # Reinitialize pipeline
        self.pipeline = NLPPipeline(CONFIG_PATH)

    def create_input_section(self):
        """Create input section for reviews"""
        st.markdown('<div class="section-header">📝 Input Reviews</div>', unsafe_allow_html=True)

        input_method = st.radio(
            "Input Method",
            ["Single Review", "Batch Reviews", "Upload File"],
            horizontal=True
        )

        reviews = []

        if input_method == "Single Review":
            review_text = st.text_area(
                "Enter a review:",
                height=100,
                placeholder="Type your review here...",
                help="Enter a single product review to analyze"
            )
            if review_text.strip():
                reviews = [review_text.strip()]

        elif input_method == "Batch Reviews":
            batch_text = st.text_area(
                "Enter multiple reviews (one per line):",
                height=200,
                placeholder="Review 1\nReview 2\nReview 3",
                help="Enter multiple reviews, one per line"
            )
            if batch_text.strip():
                reviews = [line.strip() for line in batch_text.split('\n') if line.strip()]

        elif input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload reviews file",
                type=['txt', 'csv', 'json'],
                help="Upload a file containing reviews"
            )

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.txt'):
                        content = uploaded_file.getvalue().decode('utf-8')
                        reviews = [line.strip() for line in content.split('\n') if line.strip()]

                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        text_column = st.selectbox(
                            "Select text column:",
                            df.columns.tolist(),
                            index=0 if 'review' in df.columns else 0
                        )
                        reviews = df[text_column].fillna("").astype(str).tolist()

                    elif uploaded_file.name.endswith('.json'):
                        data = json.load(uploaded_file)
                        if isinstance(data, list):
                            reviews = [str(item.get('text', item)) for item in data if item]

                    st.success(f"Loaded {len(reviews)} reviews from file")

                except Exception as e:
                    st.error(f"Error loading file: {e}")

        # Process Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_button = st.button(
                "🚀 Process Reviews",
                type="primary",
                use_container_width=True,
                disabled=len(reviews) == 0
            )

        if process_button and reviews:
            with st.spinner("Processing reviews... This may take a few minutes."):
                self.process_reviews(reviews)

        return reviews

    def process_reviews(self, reviews):
        """Process reviews through the pipeline"""
        if not self.pipeline:
            st.error("Pipeline not initialized. Please configure first.")
            return

        try:
            # Process reviews
            results = self.pipeline.process_batch(reviews)

            # Store results
            st.session_state.results = results

            # Calculate metrics if we have ground truth (for demonstration)
            self.calculate_metrics(results)

            st.success(f"✅ Processed {len(reviews)} reviews successfully!")

        except Exception as e:
            st.error(f"Error processing reviews: {e}")
            st.exception(e)

    def calculate_metrics(self, results):
        """Calculate evaluation metrics"""
        if not results:
            return

        # For demonstration, we'll create mock ground truth
        # In real usage, you'd load actual ground truth labels
        mock_sentiments = ["positive", "negative", "neutral"] * (len(results) // 3 + 1)
        mock_sentiments = mock_sentiments[:len(results)]

        # Extract predictions
        predicted_sentiments = [r['sentiment_classification']['sentiment'] for r in results]

        # Calculate classification metrics
        if len(predicted_sentiments) == len(mock_sentiments):
            classification_metrics = Evaluator.evaluate_classification(
                predicted_sentiments, mock_sentiments
            )
            st.session_state.metrics['classification'] = classification_metrics

        # Calculate summarization metrics (mock references)
        if results:
            mock_references = [
                "Good product with excellent features.",
                "Poor quality and disappointing experience.",
                "Average product with mixed results."
            ] * (len(results) // 3 + 1)
            mock_references = mock_references[:len(results)]

            predicted_summaries = []
            for r in results:
                summary = r.get('summarization_without_sentiment', {}).get('summary', '')
                if summary:
                    predicted_summaries.append(summary)

            if predicted_summaries and len(predicted_summaries) == len(mock_references[:len(predicted_summaries)]):
                summarization_metrics = Evaluator.evaluate_summarization(
                    predicted_summaries,
                    mock_references[:len(predicted_summaries)],
                    use_rouge=True,
                    use_bertscore=False
                )
                st.session_state.metrics['summarization'] = summarization_metrics

    def display_results(self):
        """Display processing results"""
        if not st.session_state.results:
            return

        st.markdown('<div class="section-header">📊 Results</div>', unsafe_allow_html=True)

        results = st.session_state.results

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Reviews", len(results))

        with col2:
            sentiments = [r['sentiment_classification']['sentiment'] for r in results]
            positive_count = sentiments.count('positive')
            st.metric("Positive", positive_count)

        with col3:
            negative_count = sentiments.count('negative')
            st.metric("Negative", negative_count)

        with col4:
            neutral_count = sentiments.count('neutral')
            st.metric("Neutral", neutral_count)

        # Results display
        st.subheader("📋 Detailed Results")

        for i, result in enumerate(results, 1):
            with st.expander(f"Review {i}", expanded=(i <= 3)):
                self.display_single_result(result, i)

        # Export results
        st.subheader("💾 Export Results")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Export as JSON"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results_{timestamp}.json"

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                st.success(f"Results exported to {filename}")

        with col2:
            if st.button("📊 Export as CSV"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results_{timestamp}.csv"

                # Flatten results for CSV
                flattened = []
                for result in results:
                    row = {
                        'review': result['original_review'][:100] + '...' if len(result['original_review']) > 100 else result['original_review'],
                        'sarcasm': result.get('sarcasm_detection', {}).get('is_sarcastic', 'N/A'),
                        'sentiment': result['sentiment_classification']['sentiment'],
                        'summary_standard': result['summarization_without_sentiment']['summary'],
                        'summary_sentiment_aware': result.get('summarization_with_sentiment', {}).get('summary', 'N/A'),
                        'summary_sarcasm_aware': result.get('summarization_with_sarcasm', {}).get('summary', 'N/A'),
                        'summary_full_context': result.get('summarization_with_sentiment_and_sarcasm', {}).get('summary', 'N/A'),
                    }
                    flattened.append(row)

                df = pd.DataFrame(flattened)
                df.to_csv(filename, index=False, encoding='utf-8')

                st.success(f"Results exported to {filename}")

    def display_single_result(self, result, index):
        """Display a single result"""
        st.markdown(f"**Original Review:** {result['original_review']}")

        col1, col2 = st.columns(2)

        with col1:
            # Sarcasm detection
            if result.get('sarcasm_detection'):
                sarcasm = result['sarcasm_detection']
                sarcasm_status = "Yes" if sarcasm['is_sarcastic'] else "No"
                st.markdown(f"**Sarcasm:** {sarcasm_status}")
                if len(sarcasm.get('response', '')) < 100:
                    st.text(f"Response: {sarcasm['response']}")

            # Sentiment
            sentiment = result['sentiment_classification']['sentiment'].upper()
            color_map = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'orange'}
            color = color_map.get(sentiment, 'blue')
            st.markdown(f"**Sentiment:** <span style='color:{color};font-weight:bold'>{sentiment}</span>", unsafe_allow_html=True)

        with col2:
            # Summaries
            st.markdown("**Standard Summary:**")
            st.info(result['summarization_without_sentiment']['summary'])

            if result.get('summarization_with_sentiment'):
                st.markdown("**Sentiment-Aware Summary:**")
                st.success(result['summarization_with_sentiment']['summary'])

            if result.get('summarization_with_sarcasm'):
                st.markdown("**Sarcasm-Aware Summary:**")
                st.warning(result['summarization_with_sarcasm']['summary'])

            if result.get('summarization_with_sentiment_and_sarcasm'):
                st.markdown("**Sentiment + Sarcasm Summary:**")
                st.success(result['summarization_with_sentiment_and_sarcasm']['summary'])

    def display_metrics(self):
        """Display evaluation metrics"""
        if not st.session_state.metrics:
            return

        st.markdown('<div class="section-header">📈 Evaluation Metrics</div>', unsafe_allow_html=True)

        metrics = st.session_state.metrics

        if 'classification' in metrics:
            st.subheader("🎯 Classification Metrics")

            class_metrics = metrics['classification']

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Accuracy", f"{class_metrics['accuracy']:.3f}")

            with col2:
                st.metric("Precision", f"{class_metrics['precision']:.3f}")

            with col3:
                st.metric("Recall", f"{class_metrics['recall']:.3f}")

            with col4:
                st.metric("F1-Score", f"{class_metrics['f1']:.3f}")

            # Sentiment distribution chart
            if st.session_state.results:
                sentiments = [r['sentiment_classification']['sentiment'] for r in st.session_state.results]
                sentiment_counts = pd.Series(sentiments).value_counts()

                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_sequence=['#ff9999', '#66b3ff', '#99ff99']
                )
                st.plotly_chart(fig, use_container_width=True)

        if 'summarization' in metrics:
            st.subheader("📝 Summarization Metrics")

            sum_metrics = metrics['summarization']

            if 'rouge' in sum_metrics:
                rouge = sum_metrics['rouge']

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("ROUGE-1 F1", f"{rouge['rouge1_f']:.3f}")

                with col2:
                    st.metric("ROUGE-2 F1", f"{rouge['rouge2_f']:.3f}")

                with col3:
                    st.metric("ROUGE-L F1", f"{rouge['rougeL_f']:.3f}")

    def run(self):
        """Run the Streamlit app"""
        # Header
        st.markdown('<div class="main-header">🤖 NLP Pipeline Dashboard</div>', unsafe_allow_html=True)
        st.markdown("*Master's Thesis - Sarcasm Detection, Sentiment Classification & Summarization*")

        # Configuration sidebar
        config = self.create_sidebar()

        # Main content
        tab1, tab2, tab3 = st.tabs(["📝 Input & Process", "📊 Results", "📈 Metrics"])

        with tab1:
            self.create_input_section()

        with tab2:
            self.display_results()

        with tab3:
            self.display_metrics()

        # Footer
        st.markdown("---")
        st.markdown("*Built with Streamlit • Master's Thesis Project • 2024*")

def main():
    """Main function"""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
