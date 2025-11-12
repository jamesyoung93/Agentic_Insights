# Architecture Overview Documentation

## 📋 Executive Summary

This document explains the **Agentic Insights** system architecture, how it was derived from the codebase, and the assumptions made during analysis.

**Generated:** 2025-11-12
**Primary Entry Point:** `streamlit_app_ULTIMATE.py`
**System Type:** Autonomous Data-Driven Discovery Engine
**Deployment:** Local/Containerized Python application with optional cloud deployment

---

## 🎯 What This System Does

**Agentic Insights** is an autonomous scientific discovery system that:

1. **Ingests** customer transaction data (CSV files) or generates synthetic datasets
2. **Analyzes** data through iterative discovery cycles using statistical methods (scipy/numpy)
3. **Synthesizes** findings using optional LLM integration (OpenAI GPT-3.5/4)
4. **Maintains** context via a structured "World Model" knowledge graph
5. **Produces** evidence-backed reports with extracted statistical rigor

**Key Value:** Automated hypothesis generation → statistical testing → discovery synthesis → publication-ready reports.

---

## 🔍 How the Architecture Was Derived

### Methodology

1. **Code Tracing**
   - Started from `streamlit_app_ULTIMATE.py` (primary entry point, 1643 lines)
   - Traced all imports: `auto_enhanced_report`, `world_model_builder`, `agents/literature_searcher`
   - Analyzed function call graphs and data flow patterns
   - Identified I/O operations (CSV reads, JSON saves, OpenAI API calls)

2. **Dependency Analysis**
   - Parsed `requirements.txt` for external dependencies (scipy, streamlit, openai, python-pptx)
   - Examined `config.py` for configuration patterns (API keys, model settings, directory paths)
   - Checked `data/` and `knowledge/` directories for data sources

3. **Data Flow Mapping**
   - Tracked data inputs: CSV files (`customers.csv`, `competitor_data.csv`), literature KB
   - Identified processing modules: statistical engine, question generator, synthesizer
   - Mapped outputs: world_model.json, auto_enhanced_report.txt, Streamlit UI

4. **Component Indexing**
   - Catalogued each module's key classes/functions
   - Documented inputs, outputs, and dependencies
   - Created `component_index.csv` with 12 distinct components

5. **Graph Construction**
   - Built node/edge graph representing data flow
   - Labeled edges with protocols (HTTP, File I/O, REST/JSON, in-memory)
   - Annotated triggers (user action, per cycle, on demand, async)
   - Saved to `architecture_graph.json` for reproducibility

---

## 📊 Architecture Overview

### System Components (MECE Breakdown)

#### 1. **Data Sources** (Input Layer)
| Component | Format | Location | Trigger |
|-----------|--------|----------|---------|
| CSV Data Files | CSV | `data/customers.csv`, `data/competitor_data.csv` | On load |
| Literature KB | JSON + TXT | `knowledge/literature/`, `knowledge/literature_index.json` | On query |
| OpenAI API | REST/JSON | External (api.openai.com) | Optional, per LLM call |

#### 2. **Processing Engine** (Core Logic)
| Module | Purpose | Key Functions | Statistical Methods |
|--------|---------|---------------|---------------------|
| **Streamlit UI** | Orchestration & rendering | `main()`, `run_discovery_cycle()` | N/A (coordinator) |
| **Question Generator** | Create research questions | `generate_research_questions_llm()`, `get_default_questions()` | N/A |
| **Statistical Analysis** | Rigorous statistical tests | `perform_statistical_analysis()` | Pearson correlation, linear regression, ANOVA, t-tests, effect sizes (Cohen's d, η², R²) |
| **Literature Search** | Query & synthesize papers | `LiteratureSearchAgent.search()` | N/A (uses LLM) |
| **Discovery Synthesizer** | Combine stats + literature | `synthesize_discoveries_llm()` | N/A (uses LLM) |

#### 3. **Storage & State** (Persistence Layer)
| Component | Type | Schema | Persistence |
|-----------|------|--------|-------------|
| **World Model** | In-memory + JSON | Discoveries, trajectories, hypotheses, cycle_summaries | `world_model.json` (saved on-demand) |
| **Report Generator** | Processing + File I/O | Extracts statistics, formats reports | `auto_enhanced_report.txt` (written on completion) |

#### 4. **Outputs** (Serving Layer)
| Output | Format | Consumer | Update Frequency |
|--------|--------|----------|------------------|
| **Interactive Dashboard** | HTML/Streamlit | User browser | Real-time (reactive) |
| **Enhanced Report** | Plain text | Human analysts | End of discovery cycles |
| **World Model JSON** | JSON | Downstream analysis / reloading | Per cycle or on-demand |

---

## 🔄 Data Flow (Primary User Journey)

```
1. USER SETUP
   ↓ (Configure objective, API key, # cycles via Streamlit UI)

2. DATA LOAD
   ↓ (Read CSV files → merge → clean → derive columns)

3. DISCOVERY LOOP (repeat 1-20 cycles)
   ├─ 3a. GENERATE QUESTIONS
   │   ↓ (LLM via OpenAI OR predefined question bank)
   │
   ├─ 3b. RUN STATISTICAL TESTS
   │   ↓ (scipy: correlation, t-test, ANOVA, regression)
   │   │  → Outputs: p-values, effect sizes, confidence intervals
   │
   ├─ 3c. SEARCH LITERATURE (optional)
   │   ↓ (Query knowledge base → LLM synthesis)
   │   │  → Outputs: relevant papers, insights
   │
   ├─ 3d. SYNTHESIZE DISCOVERIES
   │   ↓ (Combine stats + literature → LLM or direct mapping)
   │   │  → Outputs: Discovery objects with evidence
   │
   └─ 3e. UPDATE WORLD MODEL
       ↓ (Add discoveries & trajectories to graph)

4. GENERATE REPORT
   ↓ (Extract statistics → format → write TXT file)

5. DELIVER OUTPUTS
   ↓ (Dashboard + Enhanced Report + World Model JSON)

6. USER REVIEWS
   └─ (Download reports, explore discoveries, iterate)
```

**Typical Runtime:**
- 5 cycles with LLM: ~10-20 minutes
- 5 cycles statistical-only: ~2-5 minutes
- Scales with data size (tested up to ~5GB) and OpenAI API latency

---

## 🧩 Key Design Patterns

### 1. **World Model Pattern** (Context Management)
- **Inspired by:** Kosmos paper (arXiv:2511.02824v2)
- **Purpose:** Maintain structured knowledge across cycles (discoveries, trajectories, hypotheses)
- **Implementation:** `world_model_builder.py` (Discovery & Trajectory dataclasses, save/load to JSON)

### 2. **Auto-Extraction Pattern** (Statistical Rigor)
- **Purpose:** Automatically extract p-values, effect sizes, confidence intervals from analysis outputs
- **Implementation:** `auto_enhanced_report.py` (`extract_statistics_from_trajectory()`)
- **Benefit:** Ensures every claim is traceable to statistical evidence

### 3. **Hybrid LLM + Statistical** (Dual-Mode Operation)
- **Purpose:** Works with OR without LLM (OpenAI API)
- **With LLM:** Intelligent question generation, literature synthesis, discovery summarization
- **Without LLM:** Predefined question bank, direct statistical discoveries
- **Implementation:** Feature flags (`use_llm`, API key checks)

### 4. **Deduplication Pattern** (Avoid Repetition)
- **Purpose:** Prevent duplicate discoveries across cycles
- **Implementation:** `is_similar_discovery()` using Jaccard similarity on titles
- **Threshold:** 40% word overlap (lowered to catch more duplicates)

---

## 📐 Assumptions & Inferences

### Verified from Code
✅ **Primary entry point:** `streamlit_app_ULTIMATE.py` (explicitly stated in user request)
✅ **Data sources:** CSV files in `data/` (code reads `customers.csv`, `competitor_data.csv`)
✅ **Statistical methods:** scipy.stats (Pearson, t-test, ANOVA, regression) — code directly calls these
✅ **LLM integration:** OpenAI API (imported, configured in `config.py`)
✅ **Outputs:** `world_model.json`, `auto_enhanced_report.txt` (code writes these files)
✅ **Runtime trigger:** User-initiated via Streamlit UI (button click starts discovery loop)

### Inferred (High Confidence)
🔹 **Deployment:** Local/containerized (no cloud-specific code, but Dockerfile/Helm could be added)
🔹 **Scalability:** Single-node (no distributed processing, but handles datasets up to ~5GB based on README)
🔹 **Literature KB structure:** JSON index + text files (code reads `literature_index.json` and loads `.txt` files)
🔹 **API cost:** ~$0.02-0.20 per 5-cycle run (estimated from model settings and call frequency)

### Assumptions Made
⚠️ **Docker/Kubernetes:** Assumed containerization is possible but not currently implemented (no Dockerfile found in codebase)
⚠️ **Cloud deployment:** Marked as "optional" — system runs locally, cloud setup would require additional infra files
⚠️ **CI/CD:** No CI/CD YAML found; deployment is manual
⚠️ **Monitoring:** No Prometheus/Grafana integration detected; relies on Streamlit logs

---

## 🗂️ Deliverables Generated

| File | Description | Status |
|------|-------------|--------|
| `architecture_overview.pptx` | **Main deliverable:** 4-slide deck with executive overview, component table, user journey, deployment view | ✅ Created |
| `component_index.csv` | Detailed component catalog (12 components with inputs/outputs/dependencies) | ✅ Created |
| `architecture_graph.json` | Machine-readable graph (nodes, edges, protocols, triggers) for automation | ✅ Created |
| `READ_ME_ARCH_SLIDE.md` | This document (methodology, assumptions, usage guide) | ✅ Created |

---

## 🎨 PowerPoint Slide Breakdown

### **Slide 1: Executive Architecture Overview** (Main Slide)
- **Purpose:** Enable an exec to explain "how it works" in < 60 seconds
- **Layout:** Left-to-right flow (Sources → Processing → Storage → Outputs)
- **Shapes:**
  - Cylinders = Data stores (CSV, Literature KB, World Model)
  - Rectangles = Processing modules (Question Gen, Stats Engine, etc.)
  - Hexagons = External systems (OpenAI API)
  - Parallelograms = Output artifacts (Dashboard, Report, JSON)
- **Arrows:** Labeled with protocol + payload (e.g., "REST/JSON", "CSV Read")
- **Legend:** Bottom-right with shape meanings
- **Action Caption:** "Multi-cycle discovery engine combining statistical rigor with optional LLM intelligence to generate evidence-backed insights"

### **Slide 2: Component Details** (Appendix)
- **Purpose:** MECE breakdown of all components
- **Format:** Table (Component | Path | Key Functions | Inputs | Outputs)
- **Rows:** 5 core components (Streamlit UI, World Model, Report Gen, Stats Engine, Lit Search)
- **Footer:** Critical dependencies (Python 3.8+, Streamlit, SciPy, OpenAI)

### **Slide 3: Primary User Journey** (Appendix)
- **Purpose:** Step-by-step sequence from user input → output delivery
- **Format:** Vertical flow diagram (10 steps)
- **Highlighted:** Discovery loop (steps 3a-3e) as the core cycle
- **Timing:** Estimated runtime (10-20 min with LLM, 2-5 min without)

### **Slide 4: Deployment Architecture** (Appendix)
- **Purpose:** Environment-specific deployment options
- **Tiers:** Local Dev | Containerized | Cloud (optional)
- **Details:** How to run (e.g., `streamlit run`), config management, security notes

---

## 🔧 How to Use These Artifacts

### For Executives / Stakeholders
1. Open `architecture_overview.pptx`
2. View **Slide 1** for high-level understanding
3. Reference appendix slides (2-4) for deeper dives

### For Developers / Architects
1. Review `component_index.csv` for module responsibilities
2. Load `architecture_graph.json` for programmatic analysis (e.g., import into visualization tools)
3. Read this `READ_ME_ARCH_SLIDE.md` for context on how architecture was derived

### For Automation / CI/CD
1. Parse `architecture_graph.json` to generate:
   - Dependency graphs (nodes → dependencies)
   - Data lineage diagrams (sources → outputs)
   - API endpoint maps (if backend APIs added later)

---

## ✅ Verification Checklist

### Code-Based Verification (Completed)
- [x] Primary entry point identified (`streamlit_app_ULTIMATE.py`)
- [x] All imports traced (auto_enhanced_report, world_model_builder, agents)
- [x] Data sources enumerated (CSV files, literature KB, OpenAI API)
- [x] Processing modules catalogued (5 core modules)
- [x] Outputs documented (3 output types: UI, TXT report, JSON model)
- [x] Statistical methods verified (scipy.stats calls confirmed)
- [x] LLM integration confirmed (openai library imported, API calls traced)

### Deliverable Quality (Completed)
- [x] PowerPoint is self-contained (Slide 1 alone explains system in < 60 seconds)
- [x] All arrows are labeled with protocol + payload
- [x] Shapes follow legend (cylinder = datastore, rectangle = module, etc.)
- [x] No secrets exposed (API key redacted in documentation)
- [x] Component index is MECE (no overlaps, complete coverage)
- [x] Architecture graph is machine-readable (valid JSON, node/edge structure)

---

## 🚀 Next Steps (Optional Enhancements)

If you want to extend this architecture:

1. **Add CI/CD Pipeline**
   - Create `.github/workflows/deploy.yml` for automated testing & deployment
   - Run pytest on statistical functions, validate outputs

2. **Containerize**
   - Write `Dockerfile` to package app + dependencies
   - Use Docker Compose for multi-service setup (e.g., add Postgres for world model storage)

3. **Cloud Deployment**
   - Create Terraform/Helm charts for AWS/GCP/Azure
   - Add load balancer for horizontal scaling (multiple Streamlit instances)

4. **Monitoring**
   - Integrate Prometheus metrics (track cycles run, discoveries generated, API latency)
   - Add Grafana dashboards for real-time observability

5. **Data Pipeline Automation**
   - Schedule discovery runs via Airflow/Prefect
   - Automate CSV ingestion from S3/GCS

---

## 📚 References

- **Kosmos Paper:** arXiv:2511.02824v2 (inspiration for world model architecture)
- **Streamlit Docs:** https://docs.streamlit.io
- **SciPy Stats:** https://docs.scipy.org/doc/scipy/reference/stats.html
- **OpenAI API:** https://platform.openai.com/docs
- **Python-PPTX:** https://python-pptx.readthedocs.io

---

## 📝 Metadata

**Document Version:** 1.0
**Generated By:** Agentic architecture analysis (Claude Code)
**Generated Date:** 2025-11-12
**Codebase Analyzed:** Agentic_Insights repository (commit: 0b39f34)
**Primary Entry Point:** streamlit_app_ULTIMATE.py (1643 lines)
**Total Components Catalogued:** 12
**Total Dependencies:** 11 (Python packages)
**Architecture Complexity:** Moderate (5 core processing modules, 3 data sources, 3 outputs)

---

## 🤝 Contact & Feedback

For questions about this architecture documentation:
1. Review the code directly (`streamlit_app_ULTIMATE.py`, `world_model_builder.py`, etc.)
2. Check `component_index.csv` for specific module details
3. Load `architecture_graph.json` for programmatic analysis

**Assumptions or errors?** Cross-reference with actual code — this documentation was auto-generated via static analysis and may miss runtime-only behaviors.

---

**END OF DOCUMENT**
