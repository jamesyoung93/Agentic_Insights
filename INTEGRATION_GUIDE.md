# Integration Guide: Autonomous Discovery with Enhanced Analytical Rigor

## 🎯 Goal
Combine the **original KosmosFramework** autonomous discovery system with the **enhanced analytical rigor** from your newer components to create a truly autonomous, scientifically rigorous discovery system.

---

## 📊 Current State

You have two separate but complementary systems:

### 1. **Original Autonomous Framework** (`main.py` + agents/)
- ✅ Full autonomous orchestration
- ✅ Multi-agent coordination (data analysis + literature search)
- ✅ Iterative discovery cycles
- ✅ World model for context management
- ⚠️ Limited statistical rigor in reports
- ⚠️ Potential for superficial correlational claims

### 2. **Enhanced Reporting Components** (new files)
- ✅ Automatic statistical evidence extraction
- ✅ Causal inference assessment (Bradford Hill criteria)
- ✅ Explicit discussion of confounders
- ✅ Transparent limitations sections
- ⚠️ Not integrated with autonomous discovery
- ⚠️ Works on completed results, not during cycles

---

## 🔧 Integration Strategy

### Phase 1: Get Everything Running (Quick Wins)

#### A. Fix File Path Issues

**Problem:** Scripts looking for wrong filenames or in wrong locations

**Solution:** Use the fixed script

```bash
# Replace old check_dataset_schema.py with fixed version
python check_dataset_schema_FIXED.py

# Or specify path explicitly:
python check_dataset_schema_FIXED.py data/customers.csv
```

**What it does:**
- Searches common locations automatically
- Looks for all common dataset names (customers.csv, customer_data.csv, etc.)
- Provides detailed schema information
- Shows sample data and statistics

#### B. Use Integrated Streamlit App

**Problem:** Multiple streamlit app versions with inconsistent functionality

**Solution:** Use the new integrated app that works with both systems

```bash
# Run the integrated app
streamlit run streamlit_integrated.py
```

**Features:**
- ✅ Works with or without original Kosmos agents
- ✅ Loads datasets automatically from data/ directory
- ✅ Generates enhanced reports with statistical extraction
- ✅ Real-time progress tracking
- ✅ Clear error messages and diagnostics

---

### Phase 2: Full Integration (Combining Systems)

#### Option A: Enhance Original Kosmos (Recommended)

Modify your original `main.py` to use enhanced reporting:

```python
# In main.py, add at the top:
from auto_enhanced_report import AutoEnhancedReportGenerator

# In KosmosFramework class, modify the run() method:
def run(self, max_cycles: int = None):
    # ... existing code ...
    
    # At the end, replace the simple report generation with:
    print(f"\n{'=' * 80}")
    print("Generating enhanced discovery report...")
    
    # Collect all discoveries and trajectories
    all_discoveries = [d.to_dict() for d in self.world_model.discoveries]
    all_trajectories = [t.to_dict() for t in self.world_model.trajectories]
    
    # Generate enhanced report
    generator = AutoEnhancedReportGenerator()
    report_path = generator.generate_enhanced_report(
        discoveries=all_discoveries,
        trajectories=all_trajectories,
        world_model=self.world_model.to_dict()
    )
    
    print(f"✅ Enhanced report saved to: {report_path}")
    print(f"{'=' * 80}\n")
```

**Benefits:**
- Keep all autonomous discovery functionality
- Add enhanced statistical rigor automatically
- Minimal code changes
- Best of both worlds

#### Option B: Wrap Kosmos with Enhanced Orchestrator

Create a new high-level orchestrator that uses Kosmos and adds enhancement:

```python
# Create: enhanced_kosmos.py

from main import KosmosFramework
from auto_enhanced_report import AutoEnhancedReportGenerator
from world_model_builder import WorldModel as EnhancedWorldModel

class EnhancedKosmosFramework:
    """
    Enhanced version of Kosmos with rigorous statistical reporting
    """
    
    def __init__(self):
        self.kosmos = KosmosFramework()
        self.enhanced_wm = EnhancedWorldModel()
        self.report_generator = AutoEnhancedReportGenerator()
    
    def run(self, max_cycles: int = 10):
        """Run autonomous discovery with enhanced reporting"""
        
        # Set up enhanced world model
        self.enhanced_wm.set_objective(
            objective=self.kosmos.research_objectives,
            dataset_description="Customer transaction data"
        )
        
        # Run Kosmos discovery
        print("🚀 Starting Enhanced Kosmos Discovery")
        print("=" * 80)
        
        self.kosmos.run(max_cycles=max_cycles)
        
        # Extract results from Kosmos
        discoveries = self.kosmos.world_model.get_all_discoveries()
        analyses = self.kosmos.world_model.get_all_analyses()
        
        # Add to enhanced world model
        for disc in discoveries:
            self.enhanced_wm.add_discovery(
                title=disc['title'],
                summary=disc['description'],
                evidence=disc.get('evidence', []),
                trajectory_ids=disc.get('trajectory_ids', [])
            )
        
        for analysis in analyses:
            self.enhanced_wm.add_trajectory(
                trajectory_type='data_analysis',
                objective=analysis['question'],
                outputs=analysis.get('findings', {})
            )
        
        # Generate enhanced report
        print("\n📝 Generating enhanced scientific report...")
        report_path = self.report_generator.generate_enhanced_report(
            discoveries=[d.to_dict() for d in self.enhanced_wm.discoveries],
            trajectories=[t.to_dict() for t in self.enhanced_wm.trajectories],
            world_model=self.enhanced_wm.to_dict()
        )
        
        # Save enhanced world model
        self.enhanced_wm.save()
        
        print(f"✅ Enhanced report: {report_path}")
        print(f"✅ World model: {self.enhanced_wm.model_file}")
        print("=" * 80)

# Usage:
if __name__ == '__main__':
    framework = EnhancedKosmosFramework()
    framework.run(max_cycles=10)
```

---

## 🚀 Quick Start Guide

### 1. Check Your Setup

```bash
# Verify data exists
python check_dataset_schema_FIXED.py

# Expected output:
# ✅ Found dataset at: C:\Users\james\agentic-discovery\data\customers.csv
# 📊 Shape: 4,992 rows × 9 columns
# ... schema information ...
```

### 2. Test Component Scripts

```bash
# Test schema-aware questions
python schema_aware_questions.py data/customers.csv

# Test safe analysis
python safe_analysis_agent.py data/customers.csv

# Test enhanced report generation
python auto_enhanced_report.py
```

### 3. Run Integrated Streamlit App

```bash
streamlit run streamlit_integrated.py
```

Then in the UI:
1. Click "Load Default Dataset" in sidebar
2. Configure research objective
3. Set number of cycles (start with 3-5 for testing)
4. Click "Start Discovery"
5. Monitor progress in real-time
6. View enhanced reports in Results tab

### 4. Run Full Autonomous Discovery (if agents available)

```bash
# If you have the agents/ directory with all components:
python main.py

# Or use enhanced version if you created it:
python enhanced_kosmos.py
```

---

## 📁 File Organization

Recommended directory structure:

```
your-project/
│
├── agents/                           # Original Kosmos agents
│   ├── __init__.py
│   ├── world_model.py               # Original world model
│   ├── data_analyst.py              # Data analysis agent
│   └── literature_searcher.py       # Literature search agent
│
├── data/                            # Your datasets
│   └── customers.csv                # ← Your actual data file
│
├── knowledge/                       # Literature database
│   └── literature_index.json
│
├── prompts/                         # Prompt templates
│   └── research_objectives.txt
│
├── outputs/                         # Generated outputs
│   ├── auto_enhanced_report.txt    # Enhanced reports
│   └── world_model.json            # World model state
│
├── main.py                         # Original Kosmos orchestrator
├── config.py                       # Configuration
│
├── auto_enhanced_report.py         # NEW: Enhanced report generator
├── world_model_builder.py          # NEW: Enhanced world model
├── safe_analysis_agent.py          # NEW: Safe data analysis
├── schema_aware_questions.py       # NEW: Schema-based questions
│
├── check_dataset_schema_FIXED.py  # FIXED: Dataset checker
├── streamlit_integrated.py         # FIXED: Integrated UI
│
└── requirements.txt                # Dependencies
```

---

## 🔍 Troubleshooting

### Issue: "Could not find dataset"

**Symptoms:**
```
❌ Could not find dataset in standard locations
```

**Solutions:**
1. Use fixed checker: `python check_dataset_schema_FIXED.py`
2. Verify file exists: `dir data\` (Windows) or `ls data/` (Linux/Mac)
3. Check filename matches: Should be `customers.csv` not `customer_data.csv`
4. Specify path explicitly: `python script.py data/customers.csv`

### Issue: "Import Error - agents not found"

**Symptoms:**
```
ImportError: No module named 'agents'
```

**Solutions:**
1. Check if `agents/` directory exists
2. Verify `agents/__init__.py` exists (can be empty)
3. Use integrated streamlit app which works without agents: `streamlit run streamlit_integrated.py`

### Issue: "Reports not showing in Streamlit"

**Symptoms:**
- Discovery runs but no report appears
- "No enhanced report found" message

**Solutions:**
1. Check working directory in sidebar "File Paths" expander
2. Look for `.txt` files: `dir *.txt` (Windows) or `ls *.txt` (Linux/Mac)
3. Verify enhanced reports enabled in sidebar
4. Click "Retry Report Generation" button
5. Check logs tab for error messages

### Issue: "Statistical rigor concerns"

**Symptoms:**
- Reports show correlations without causal assessment
- Missing p-values or effect sizes
- No discussion of confounders

**Solutions:**
1. Ensure using `AutoEnhancedReportGenerator` 
2. Verify `auto_enhanced_report.py` is imported correctly
3. Check that trajectories include statistical outputs
4. Review generated report for "Causal Assessment" section

---

## ✅ Next Steps

### Immediate (Today):

1. **Fix path issues:**
   ```bash
   python check_dataset_schema_FIXED.py
   ```

2. **Test integrated UI:**
   ```bash
   streamlit run streamlit_integrated.py
   ```

3. **Run a short discovery (3 cycles)** to verify everything works

### Short-term (This Week):

1. **Integrate enhanced reporting into main.py** (see Option A above)

2. **Run full 10-cycle discovery** with enhanced reports

3. **Review generated reports** for scientific rigor

4. **Customize agents** to include more statistical tests

### Long-term (This Month):

1. **Add Bradford Hill criteria** evaluation to each discovery

2. **Implement causal DAG generation** for confounders

3. **Create automated literature synthesis** with citations

4. **Build deployment pipeline** for production use

---

## 📚 Key Principles

Remember the core principles driving this integration:

1. **Scientific Rigor Over Convenience**
   - Every claim needs evidence
   - Correlation ≠ causation
   - Explicit limitations

2. **Transparency and Traceability**
   - Code links for all analyses
   - Literature citations for claims
   - Clear methodology descriptions

3. **Autonomous but Rigorous**
   - Minimal human intervention
   - Maximal scientific validity
   - No superficial insights

4. **Iterative Enhancement**
   - Start with working system
   - Add rigor progressively
   - Test at each step

---

## 🆘 Getting Help

If you encounter issues:

1. **Check logs** in Streamlit app or console output
2. **Review file paths** in sidebar diagnostics
3. **Test components individually** before full integration
4. **Start small** (3-5 cycles) before scaling up

---

## 📝 Summary

**Your Goal:** Autonomous discovery + Enhanced analytical rigor

**Best Path:** 
1. Use `streamlit_integrated.py` for immediate functionality
2. Fix any remaining path issues with `check_dataset_schema_FIXED.py`
3. Integrate `AutoEnhancedReportGenerator` into `main.py`
4. Test with small cycles first, then scale up

**Expected Outcome:**
- Fully autonomous multi-cycle discovery
- Research-grade statistical analysis
- Rigorous causal assessment
- Transparent limitations
- Traceable evidence chains

You're close! Just need to connect the pieces. 🚀
