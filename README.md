# Agentic Insights

Autonomous data-driven discovery system that generates rigorous, evidence-based reports through iterative analysis cycles. Built with Python and Streamlit.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Web Interface

```bash
streamlit run streamlit_app_enhanced.py
```

### Run from Command Line

```bash
python main.py
```

## What's Included

- **streamlit_app_enhanced.py** - Interactive web UI
- **auto_enhanced_report.py** - Report generation with statistical evidence extraction
- **world_model_builder.py** - Context management and state tracking
- **main.py** - Command-line interface
- **agents/** - Discovery agent modules (world model, data analyst, literature searcher)
- **data/** - Sample datasets
- **prompts/** - Agent prompt templates
- **knowledge/** - Reference literature

## Core Features

- 📊 Automated statistical analysis and evidence extraction
- 🧠 Context-aware discovery through world models
- 📝 Evidence-based report generation
- 🖥️ Interactive Streamlit interface with real-time progress
- 💾 Persistent state and cycle tracking
- 🔍 Integrates multiple discovery agents

## Usage Example

```python
from world_model_builder import WorldModel
from auto_enhanced_report import AutoEnhancedReportGenerator

# Initialize world model
wm = WorldModel()
wm.set_objective("Analyze customer behavior patterns")

# Run discovery cycle
wm.increment_cycle()
trajectory = wm.add_trajectory(
    trajectory_type="data_analysis",
    objective="Find correlations",
    outputs={"correlation": 0.85, "p_value": 0.001}
)

# Record discovery
wm.add_discovery(
    title="Age-Product Correlation",
    summary="Strong correlation detected",
    evidence=["p < 0.001", "r = 0.85"],
    trajectory_ids=[trajectory.id]
)

# Generate report
generator = AutoEnhancedReportGenerator()
report_path = generator.generate_enhanced_report(
    discoveries=[d.to_dict() for d in wm.discoveries],
    trajectories=[t.to_dict() for t in wm.trajectories],
    world_model=wm.to_dict()
)

# Save state
wm.save()
```

## Documentation

- **START_HERE.md** - Quick start guide
- **SETUP_INSTRUCTIONS.md** - Detailed setup and troubleshooting

## Requirements

- Python 3.8+
- Streamlit
- Pandas, NumPy, SciPy
- OpenAI API (optional, for LLM features)

See `requirements.txt` for full dependencies.

## Scientific Foundation

Inspired by the Kosmos framework (arXiv:2511.02824v2) emphasizing:
- Statistical rigor in analysis
- Traceability of all claims
- Explicit methodology documentation
- Causal assessment standards

## License

See LICENSE file for details.
