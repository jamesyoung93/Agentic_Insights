# Architecture Overview Slide - Generation Notes

## Method
- **Library**: python-pptx 0.6.21+
- **Diagram Engine**: Programmatic shapes (no external graph tools)
- **Validation**: ZIP check + reopen test passed

## Architecture Summary
The Agentic Insights system follows a multi-agent architecture:

1. **Sources Layer**: CSV data files, OpenAI API
2. **Orchestration Layer**: Streamlit UI, Kosmos main orchestrator
3. **Processing Layer**: Data Analysis Agent, Literature Agent, Statistical Engine
4. **State & Storage Layer**: World Model, Local file storage
5. **Outputs Layer**: Analysis scripts, final reports

## Link Classification
- **Verified Links** (solid): All connections verified in source code
- **Inferred Links** (dotted): None - all flows traced through actual imports/calls

## Node Count: 11 (under 12 limit)
## Edge Count: 12 (under 16 limit)

## Assumptions
- Main entry point: streamlit_app_ULTIMATE.py (web) or main.py (CLI)
- Statistical analysis uses scipy (verified in imports)
- OpenAI API used for LLM features (optional, verified in code)
- All state persisted to local JSON files (verified)

## Unresolved Questions
None - all architecture components verified in codebase.
