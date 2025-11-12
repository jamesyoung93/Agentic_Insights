# Kosmos AI Scientist - Ultimate Edition

Autonomous data-driven discovery system using real statistical analysis (correlations, regressions, ANOVA, t-tests) combined with optional LLM-powered research question generation and synthesis. Built with Python, Streamlit, and SciPy.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run streamlit_app_ULTIMATE.py
```

The app will launch at `http://localhost:8501`

## How It Works

1. **Statistical Analysis** - Real statistical tests on your data (Pearson correlations, linear regression, ANOVA, t-tests)
2. **World Model** - Tracks all discoveries and maintains context across research cycles
3. **Enhanced Reports** - Generates publication-quality reports with full statistical evidence
4. **Optional LLM** - Add your OpenAI API key in the sidebar for intelligent question generation and discovery synthesis

**Key Features:**
- 📊 6+ statistical analysis methods with effect sizes
- 🧠 World model for contextual discovery tracking
- 📝 Publication-ready report generation
- 🤖 Optional LLM integration (GPT-4, GPT-3.5-turbo)
- 📁 Automatic sample data generation if no CSV provided
- 💾 Persistent state across sessions

## File Structure

```
├── streamlit_app_ULTIMATE.py    # Main application
├── world_model_builder.py       # Discovery tracking and state management
├── auto_enhanced_report.py      # Report generation
├── config.py                    # Configuration (API keys)
├── requirements.txt             # Dependencies
├── data/                        # Sample datasets
├── agents/                      # Optional agent modules
├── knowledge/                   # Reference materials
└── prompts/                     # LLM prompt templates
```

## Usage

### Basic Usage (No API Key Required)

1. Open the app: `streamlit run streamlit_app_ULTIMATE.py`
2. Configure your research objective in the sidebar
3. Click "Start Discovery"
4. View results in the Results and Discoveries tabs
5. Download the enhanced report

### With LLM Features

1. Get an OpenAI API key from https://platform.openai.com/api-keys
2. Paste it in the sidebar under "OpenAI API Key"
3. The app will use GPT for intelligent question generation and discovery synthesis

## Requirements

- Python 3.8+
- Streamlit
- Pandas, NumPy, SciPy
- OpenAI API (optional, for LLM features)

See `requirements.txt` for exact versions.

## Documentation

- **START_HERE.md** - Quick start guide and feature overview
- **SETUP_INSTRUCTIONS.md** - Detailed setup and troubleshooting

## Scientific Foundation

Based on the Kosmos framework emphasizing:
- Statistical rigor in all analysis
- Traceability of claims to evidence
- Explicit methodology documentation
- Proper causal assessment

See arXiv:2511.02824v2 for details.

## License

See LICENSE file for details.
