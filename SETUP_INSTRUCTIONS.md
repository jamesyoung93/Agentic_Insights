# Setup Instructions - Autonomous Discovery System

## Quick Start (5 minutes)

### 1. Copy Files to Your Project Directory

Copy these three Python files to your project directory:
- `auto_enhanced_report.py`
- `streamlit_app_enhanced.py`
- `world_model_builder.py`

### 2. Install Dependencies

```bash
pip install streamlit pandas
```

### 3. Run the Application

```bash
streamlit run streamlit_app_enhanced.py
```

That's it! The app should open in your browser automatically.

---

## What Was Fixed

### Problem
- Enhanced reports were being generated but saved to inconsistent locations
- Streamlit app couldn't find the generated reports
- File paths were relative instead of absolute

### Solution
All three files now use **absolute paths** based on `Path.cwd()` which ensures:
- ✅ Files are always saved to the same location
- ✅ Files are always found when loading
- ✅ Clear error messages show exactly where files should be
- ✅ Debug information helps troubleshoot any remaining issues

---

## File Overview

### 1. `auto_enhanced_report.py`
**Purpose**: Generates enhanced discovery reports with extracted statistics

**Key Features**:
- Automatically extracts statistical evidence from trajectories
- Formats discoveries with supporting data
- Uses absolute paths: `BASE_DIR / "auto_enhanced_report.txt"`
- Clear success/error messages

**Usage**:
```python
from auto_enhanced_report import AutoEnhancedReportGenerator

generator = AutoEnhancedReportGenerator()
report_path = generator.generate_enhanced_report(
    discoveries=my_discoveries,
    trajectories=my_trajectories,
    world_model=my_world_model
)
```

### 2. `streamlit_app_enhanced.py`
**Purpose**: Main web interface for running discoveries

**Key Features**:
- Interactive discovery execution
- Real-time progress tracking
- Enhanced report viewing and download
- Comprehensive logging
- **Fixed path resolution** - checks multiple locations and shows what it finds

**New Features**:
- Shows working directory in sidebar
- Displays debug info when reports aren't found
- Lists all .txt files in current directory
- Better error messages

### 3. `world_model_builder.py`
**Purpose**: Manages structured knowledge across discovery cycles

**Key Features**:
- Structured data models for discoveries and trajectories
- Persistent storage with JSON
- Context summarization for LLM prompts
- Cycle management and history tracking

**Usage**:
```python
from world_model_builder import WorldModel

# Create or load model
wm = WorldModel()
wm.set_objective("Research objective here")

# Add discoveries
wm.add_discovery(
    title="Important Finding",
    summary="What we discovered",
    evidence=["Evidence 1", "Evidence 2"],
    trajectory_ids=["traj_1"]
)

# Save model
wm.save()
```

---

## Testing the Setup

### 1. Test Enhanced Report Generation

```python
python auto_enhanced_report.py
```

**Expected Output**:
```
✅ Enhanced report saved to: /path/to/your/directory/auto_enhanced_report.txt
Report generated at: /path/to/your/directory/auto_enhanced_report.txt
```

### 2. Test World Model

```python
python world_model_builder.py
```

**Expected Output**:
```
✅ World model saved to: /path/to/your/directory/world_model.json
[Summary output with statistics]
```

### 3. Test Streamlit App

```bash
streamlit run streamlit_app_enhanced.py
```

**Expected**:
- Browser opens to http://localhost:8501
- Sidebar shows "Working Directory" path
- Can run a discovery and see reports in Results tab

---

## Troubleshooting

### Issue: "No enhanced report found"

**Check**:
1. In the Streamlit app sidebar, look at "File Paths" expander
2. Note the "Working Directory" path
3. Verify files are being saved there

**Solution**: The app now shows:
- Which paths it checked
- What .txt files it found
- Exact location where it's looking

### Issue: Report generates but doesn't show

**Possible Causes**:
1. File saved to different directory than expected
2. Permissions issue preventing read
3. File is empty

**Debug Steps**:
```python
# Run this in Python to check:
from pathlib import Path

base_dir = Path.cwd()
report_path = base_dir / "auto_enhanced_report.txt"

print(f"Current directory: {base_dir}")
print(f"Report exists: {report_path.exists()}")

if report_path.exists():
    print(f"File size: {report_path.stat().st_size} bytes")
    print(f"Contents preview:")
    print(report_path.read_text()[:200])
```

### Issue: Import errors

**Solution**:
```bash
# Install missing packages
pip install streamlit pandas

# If using virtual environment, activate it first
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

---

## Integration with Your Existing System

If you have existing discovery/analysis code, integrate it by:

### 1. Hook into Discovery Generation

```python
from auto_enhanced_report import AutoEnhancedReportGenerator
from world_model_builder import WorldModel

# Your existing discovery results
my_discoveries = [...]
my_trajectories = [...]

# Create world model
wm = WorldModel()
wm.set_objective("Your research objective")

# Add your discoveries to world model
for disc in my_discoveries:
    wm.add_discovery(
        title=disc['title'],
        summary=disc['summary'],
        evidence=disc.get('evidence', []),
        trajectory_ids=disc.get('trajectory_ids', [])
    )

# Generate enhanced report
generator = AutoEnhancedReportGenerator()
report_path = generator.generate_enhanced_report(
    discoveries=my_discoveries,
    trajectories=my_trajectories,
    world_model=wm.to_dict()
)

# Save world model
wm.save()
```

### 2. Use World Model for Context

```python
from world_model_builder import WorldModel

# Load existing world model
wm = WorldModel.load()

# Get context for LLM prompt
context = wm.generate_context_summary(max_discoveries=10)

# Use in your LLM calls
prompt = f"""
{context}

Based on the above context, analyze the following data:
...
"""
```

---

## Advanced Configuration

### Custom Base Directory

```python
from pathlib import Path
from auto_enhanced_report import AutoEnhancedReportGenerator
from world_model_builder import WorldModel

# Set custom directory
custom_dir = Path("/path/to/your/project")

# Use with all components
generator = AutoEnhancedReportGenerator(base_dir=custom_dir)
wm = WorldModel(base_dir=custom_dir)

# They will all use the same directory
```

### Multiple Projects

```python
# Project 1
project1_dir = Path("./projects/healthcare")
generator1 = AutoEnhancedReportGenerator(base_dir=project1_dir)
wm1 = WorldModel(base_dir=project1_dir)

# Project 2
project2_dir = Path("./projects/finance")
generator2 = AutoEnhancedReportGenerator(base_dir=project2_dir)
wm2 = WorldModel(base_dir=project2_dir)
```

---

## Next Steps

1. ✅ Copy files to your directory
2. ✅ Run tests to verify everything works
3. ✅ Run the Streamlit app
4. 🚀 Integrate with your existing analysis pipeline
5. 🔬 Start generating discoveries!

---

## Support

If you encounter issues:

1. **Check the working directory**: Look in the Streamlit sidebar under "File Paths"
2. **Verify file creation**: Run the test scripts
3. **Check permissions**: Ensure you can write to the directory
4. **Review logs**: Check the "Logs" tab in the Streamlit app

All files now include extensive error handling and debug output to help identify issues quickly.
