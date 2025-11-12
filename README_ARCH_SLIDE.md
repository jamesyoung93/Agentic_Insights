# Architecture Overview Slide Documentation

## Overview

This document explains how the architecture slide (`architecture_overview.pptx`) was derived from the Agentic_Insights codebase and outlines the key assumptions and methodology used.

## Primary Entry Point

**Main Application:** `streamlit_app_ULTIMATE.py`

This is the core orchestration layer that:
- Provides the Streamlit web UI
- Manages discovery cycles
- Coordinates all other components
- Handles user interactions and configuration

## Architecture Derivation Methodology

### 1. Code Analysis

The architecture was derived through systematic analysis of:

1. **Entry Point Analysis**
   - Started with `streamlit_app_ULTIMATE.py` (1,642 lines)
   - Traced all import statements and function calls
   - Mapped data flow through the application

2. **Module Dependencies**
   - Analyzed `requirements.txt` for external dependencies
   - Examined internal module imports:
     - `auto_enhanced_report.py` - Report generation
     - `world_model_builder.py` - State management
     - `agents/literature_searcher.py` - Literature search
     - `agents/data_analyst.py` - Statistical analysis
     - `agents/world_model.py` - Agent-specific state

3. **Data Flow Tracing**
   - Tracked data from sources (CSV files) through processing to outputs
   - Identified key transformations at each stage
   - Mapped I/O operations (file reads, API calls, JSON persistence)

4. **Function Call Graph**
   - Identified primary functions and their relationships:
     - `load_data()` í data ingestion
     - `run_discovery_cycle()` í orchestration
     - `perform_statistical_analysis()` í scipy-based tests
     - `generate_research_questions_llm()` í OpenAI API
     - `synthesize_discoveries_llm()` í LLM synthesis
     - `generate_final_report()` í report generation

### 2. Component Categorization (MECE Framework)

Components were grouped into mutually exclusive, collectively exhaustive categories:

**DATA SOURCES**
- CSV files (customers.csv, competitor_data.csv)
- Literature knowledge base (knowledge/literature/)
- OpenAI API (external LLM)

**INGESTION**
- Data loading, merging, cleaning, imputation, feature engineering
- Implemented in: `load_data()`, `generate_sample_data()`, `_sanitize_for_analysis()`

**CORE PROCESSING**
- Discovery Cycle Engine: Orchestrates question í analysis í synthesis loop
- Statistical Analysis Engine: scipy-based tests (correlations, regression, ANOVA, t-tests)
- Literature Search Agent: LLM-powered knowledge base search
- Discovery Synthesis: Combines statistical + literature evidence via LLM
- Report Generator: Extracts statistics and formats reports

**STORAGE**
- World Model: In-memory knowledge state (discoveries, trajectories, hypotheses)
- Persistence: JSON serialization (world_model.json)

**OUTPUTS**
- Streamlit Web UI: Interactive interface
- Discovery Reports: Enhanced .txt files with statistical evidence
- World Model JSON: Persistent knowledge state

### 3. Data Flow Mapping

**Primary Flow (Left to Right):**

```
CSV Data í Data Ingestion í Discovery Engine í Statistical Analysis
                                ì
                         Literature Agent ê Literature KB
                                ì
                         OpenAI API (optional)
                                ì
                         Discovery Synthesis
                                ì
                         World Model í Report Generator í Reports
                                ì
                         JSON Persistence
                                ì
                         Streamlit UI (display)
```

**Key Triggers:**
- **User-initiated:** User clicks "Start Discovery" in Streamlit UI
- **Per-cycle:** 3-5 research questions generated and analyzed
- **Optional LLM:** If API key provided, uses GPT-3.5/4 for question generation and synthesis
- **End-of-discovery:** Report generation and JSON persistence

### 4. Protocol and Interface Identification

Each edge in the architecture was labeled with:
- **Protocol:** file I/O, REST API, function call, in-memory
- **Format:** CSV, JSON, pandas DataFrame, Python dict
- **Trigger:** user-initiated, per-cycle, on-load, immediate

**Verified Flows (Solid Lines):**
- CSV í DataFrame (pandas.read_csv)
- DataFrame í Statistical Analysis (scipy.stats)
- Statistical Results í World Model (object methods)
- World Model í JSON (file I/O)

**Optional/Inferred Flows (Dotted Lines):**
- OpenAI API integration (only if API key provided)
- Literature search (optional feature)
- LLM-based synthesis (fallback to statistical discoveries if no LLM)

## Key Assumptions

### 1. **Primary Use Case**
The system is designed for autonomous discovery on customer transaction data. The architecture assumes:
- Batch data processing (not real-time streaming)
- Single-user, local execution
- Research/analytics workload (not production serving)

### 2. **Optional LLM Integration**
- OpenAI API is **optional** - system functions without it
- When LLM is unavailable, system uses:
  - Curated research questions (instead of LLM-generated)
  - Direct statistical discoveries (instead of LLM synthesis)
- This dual-mode operation is a core architectural principle

### 3. **Data Sources**
- Primary data: CSV files in `data/` directory
- Fallback: If CSV files missing, generates synthetic sample data
- Literature: Assumes pre-curated papers in `knowledge/literature/`

### 4. **Persistence Strategy**
- Session state: In-memory (Streamlit session_state)
- Long-term: JSON files (world_model.json, reports)
- No database dependency (file-based persistence only)

### 5. **Scalability Constraints**
- Single-user application (no multi-tenancy)
- Local execution (not cloud-deployed in current form)
- In-memory data processing (limited to datasets that fit in RAM)

### 6. **Statistical Rigor**
- All statistical tests use scipy/numpy (not simulated)
- Effect sizes calculated (Cohen's d, ∑≤, R≤)
- P-values and confidence intervals reported
- Both significant AND non-significant findings tracked

## Verification Status

**Verified Components (from code inspection):**
-  CSV data loading (line 116-233 of streamlit_app_ULTIMATE.py)
-  Statistical analysis engine (line 283-461)
-  World model state management (world_model_builder.py)
-  Report generation (auto_enhanced_report.py)
-  Streamlit UI (line 1153-1642)

**Inferred Components (best-effort):**
- † Exact OpenAI API call patterns (verified prompts exist, but actual usage optional)
- † Literature search implementation (agent exists but not always invoked)
- † Discovery deduplication logic (implemented via `is_similar_discovery()`)

## Design Principles (Consulting-Style)

The slide follows McKinsey/Bain/BCG presentation standards:

### Visual Hierarchy
1. **Title:** Action-oriented, describes what the system does
2. **Subtitle:** One-sentence value proposition
3. **Main Content:** Left-to-right flow with MECE groupings
4. **Legend:** Bottom section with shape conventions and metadata

### Color Palette
- **Dark Blue (Primary):** Titles, important labels
- **Light Blue:** Data stores and processing components
- **Yellow/Orange:** External integrations (OpenAI API)
- **Green:** Outputs and deliverables
- **Gray:** Connectors and supporting text

### Shape Conventions (Industry Standard)
- **Cylinder:** Datastores (databases, files, caches)
- **Rectangle:** Processing modules (services, functions)
- **Hexagon:** External systems (APIs, third-party)
- **Parallelogram:** Input/Output artifacts (reports, files)

### MECE Structure
Each component belongs to exactly ONE category:
- No overlap between "Ingestion" and "Processing"
- No overlap between "Storage" and "Outputs"
- Collectively exhaustive: All major components included

## Files Generated

1. **architecture_overview.pptx** - Main consulting-style slide
2. **component_index.csv** - Detailed component inventory
3. **architecture_graph.json** - Machine-readable graph structure
4. **README_ARCH_SLIDE.md** - This documentation

## Reproducing the Architecture Slide

To regenerate the architecture slide:

```bash
# Ensure python-pptx is installed
pip install python-pptx

# Run the generation script
python create_arch_ppt.py

# Output: architecture_overview.pptx
```

The script programmatically creates the PowerPoint using the python-pptx library, ensuring:
- Consistent positioning and sizing
- Professional color scheme
- Proper shape relationships
- Readable text at all zoom levels

## Future Enhancements

If the codebase evolves, consider updating:

1. **Multi-environment support:** Add dev/staging/prod annotations
2. **API versioning:** Show OpenAI API version dependencies
3. **Performance metrics:** Add latency/throughput indicators if measured
4. **Deployment view:** If containerized (Docker/K8s), add infrastructure layer
5. **Security boundaries:** If authentication added, show trust zones

## Contact / Attribution

This architecture was derived through automated code analysis combined with manual verification. All component relationships were traced from actual import statements and function calls in the codebase.

**Last Updated:** 2025-11-12
**Code Version:** Based on streamlit_app_ULTIMATE.py (commit c5e87bc)
**Methodology:** MECE decomposition + left-to-right data flow mapping
