"""
Autonomous Discovery System - Streamlit Interface
Fixed version with proper file path handling
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from auto_enhanced_report import AutoEnhancedReportGenerator

# Set page config
st.set_page_config(
    page_title="Autonomous Discovery System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'discovery_state' not in st.session_state:
    st.session_state.discovery_state = {
        'running': False,
        'current_cycle': 0,
        'max_cycles': 10,
        'discoveries': [],
        'trajectories': [],
        'world_model': {},
        'logs': [],
        'enhanced_report_path': None
    }

# Get base directory (where app is running from)
BASE_DIR = Path.cwd()
ENHANCED_REPORT_PATH = BASE_DIR / "auto_enhanced_report.txt"

def log_message(message: str, level: str = "info"):
    """Add a log message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.discovery_state['logs'].append({
        'timestamp': timestamp,
        'level': level,
        'message': message
    })

def load_enhanced_report() -> str:
    """
    Load the enhanced report from file with improved error handling
    
    Returns:
        Report content as string, or error message
    """
    # Check multiple possible locations
    possible_paths = [
        ENHANCED_REPORT_PATH,  # Primary location
        BASE_DIR / "enhanced_discovery_report.txt",  # Fallback 1
        Path("auto_enhanced_report.txt"),  # Relative path fallback
        Path("enhanced_discovery_report.txt"),  # Relative path fallback 2
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():  # Make sure file is not empty
                    st.success(f"✅ Loaded report from: {path}")
                    return content
            except Exception as e:
                st.warning(f"⚠️ Could not read {path}: {e}")
                continue
    
    # If we get here, no report was found
    st.error("⚠️ No enhanced report found.")
    st.info(f"📁 Looking in: {BASE_DIR}")
    
    # Show what files ARE in the directory
    txt_files = list(BASE_DIR.glob("*.txt"))
    if txt_files:
        st.info(f"📄 Found these .txt files: {[f.name for f in txt_files]}")
    else:
        st.info("📄 No .txt files found in current directory")
    
    return None

def simulate_discovery_cycle():
    """Simulate running a discovery cycle"""
    state = st.session_state.discovery_state
    
    if state['current_cycle'] >= state['max_cycles']:
        state['running'] = False
        log_message("✅ Discovery complete! Generating enhanced report...", "success")
        generate_final_report()
        return
    
    state['current_cycle'] += 1
    log_message(f"🔄 Running cycle {state['current_cycle']}/{state['max_cycles']}", "info")
    
    # Simulate some work
    time.sleep(0.5)
    
    # Add a mock discovery every few cycles
    if state['current_cycle'] % 3 == 0:
        discovery = {
            'title': f'Discovery {len(state["discoveries"]) + 1}',
            'summary': f'Interesting finding from cycle {state["current_cycle"]}',
            'cycle': state['current_cycle'],
            'trajectory_ids': [f'traj_{state["current_cycle"]}_1', f'traj_{state["current_cycle"]}_2']
        }
        state['discoveries'].append(discovery)
        log_message(f"📊 New discovery identified: {discovery['title']}", "success")
    
    # Add mock trajectory
    trajectory = {
        'id': f'traj_{state["current_cycle"]}_1',
        'cycle': state['current_cycle'],
        'type': 'analysis',
        'outputs': {
            'p_value': 0.001 * state['current_cycle'],
            'effect_size': 0.5 + 0.1 * state['current_cycle']
        }
    }
    state['trajectories'].append(trajectory)

def generate_final_report():
    """Generate the final enhanced report"""
    state = st.session_state.discovery_state
    
    try:
        # Create generator with explicit base directory
        generator = AutoEnhancedReportGenerator(base_dir=BASE_DIR)
        
        # Prepare data
        cycle_results = {
            'discoveries': state['discoveries'],
            'trajectories': state['trajectories'],
            'world_model': state.get('world_model', {})
        }
        
        # Generate report
        report_path = generator.generate_from_cycle_data(cycle_results)
        state['enhanced_report_path'] = report_path
        
        log_message(f"✅ Enhanced report generated: {report_path}", "success")
        st.success(f"✅ Enhanced report saved to: {report_path}")
        
    except Exception as e:
        log_message(f"❌ Error generating report: {e}", "error")
        st.error(f"❌ Error generating report: {e}")
        import traceback
        st.code(traceback.format_exc())

def main():
    """Main app function"""
    
    # Sidebar configuration
    with st.sidebar:
        st.title("🔬 Discovery System")
        st.markdown("---")
        
        # Configuration
        st.subheader("Configuration")
        
        objective = st.text_area(
            "Research Objective",
            value="Investigate patterns in the dataset",
            height=100
        )
        
        max_cycles = st.slider(
            "Maximum Cycles",
            min_value=1,
            max_value=50,
            value=10
        )
        
        enable_enhanced_reports = st.checkbox(
            "Enable Enhanced Reports",
            value=True,
            help="Generate detailed reports with extracted statistics"
        )
        
        st.markdown("---")
        
        # System status
        st.subheader("System Status")
        state = st.session_state.discovery_state
        st.metric("Current Cycle", f"{state['current_cycle']}/{state['max_cycles']}")
        st.metric("Discoveries", len(state['discoveries']))
        st.metric("Trajectories", len(state['trajectories']))
        
        st.markdown("---")
        
        # File paths info
        with st.expander("📁 File Paths"):
            st.text(f"Working Directory:\n{BASE_DIR}")
            st.text(f"\nReport Location:\n{ENHANCED_REPORT_PATH}")
    
    # Main content area
    st.title("🔬 Autonomous Discovery System")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "▶️ Run Discovery", "📊 Results", "📝 Logs"])
    
    with tab1:
        st.header("Welcome to the Autonomous Discovery System")
        st.markdown("""
        This system performs autonomous data-driven discovery through iterative cycles of:
        - 📊 Data analysis
        - 📚 Literature search
        - 🧠 Hypothesis generation
        - 📈 World model updates
        
        ### Getting Started
        1. Configure your research objective in the sidebar
        2. Set the maximum number of cycles
        3. Enable enhanced reports for detailed statistical evidence
        4. Navigate to the "Run Discovery" tab to start
        
        ### Features
        - **Enhanced Reports**: Automatically extract and format statistical evidence
        - **World Model**: Maintain context across discovery cycles
        - **Real-time Monitoring**: Track progress and discoveries as they happen
        """)
        
        # Show current working directory
        st.info(f"📁 **Working Directory**: `{BASE_DIR}`")
    
    with tab2:
        st.header("Run Discovery")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start Discovery", disabled=state['running']):
                state['running'] = True
                state['current_cycle'] = 0
                state['discoveries'] = []
                state['trajectories'] = []
                state['logs'] = []
                state['world_model'] = {
                    'objective': objective,
                    'max_cycles': max_cycles,
                    'enhanced_reports': enable_enhanced_reports
                }
                log_message("🚀 Starting discovery process...", "info")
                st.rerun()
        
        with col2:
            if st.button("⏸️ Pause", disabled=not state['running']):
                state['running'] = False
                log_message("⏸️ Discovery paused", "warning")
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset"):
                st.session_state.discovery_state = {
                    'running': False,
                    'current_cycle': 0,
                    'max_cycles': 10,
                    'discoveries': [],
                    'trajectories': [],
                    'world_model': {},
                    'logs': [],
                    'enhanced_report_path': None
                }
                log_message("🔄 System reset", "info")
                st.rerun()
        
        # Progress display
        if state['running']:
            st.markdown("---")
            st.subheader("📈 Progress")
            
            progress = state['current_cycle'] / state['max_cycles']
            st.progress(progress)
            
            st.info(f"🔄 Running cycle {state['current_cycle']}/{state['max_cycles']}")
            
            # Auto-advance to next cycle
            simulate_discovery_cycle()
            time.sleep(1)
            st.rerun()
        
        elif state['current_cycle'] > 0:
            st.success("✅ Discovery complete! Check the Results tab.")
    
    with tab3:
        st.header("Discovery Results")
        
        if state['current_cycle'] == 0:
            st.info("👈 Run a discovery to see results here")
        else:
            # Show discoveries
            st.subheader(f"📊 Discoveries ({len(state['discoveries'])})")
            
            if state['discoveries']:
                for i, disc in enumerate(state['discoveries'], 1):
                    with st.expander(f"Discovery {i}: {disc['title']}"):
                        st.markdown(f"**Summary**: {disc['summary']}")
                        st.markdown(f"**Found in Cycle**: {disc['cycle']}")
                        st.markdown(f"**Related Trajectories**: {', '.join(disc['trajectory_ids'])}")
            else:
                st.info("No discoveries yet")
            
            st.markdown("---")
            
            # Enhanced Report Section
            st.subheader("📄 Enhanced Report")
            
            if enable_enhanced_reports:
                report_content = load_enhanced_report()
                
                if report_content:
                    # Add download button
                    st.download_button(
                        label="⬇️ Download Enhanced Report",
                        data=report_content,
                        file_name=f"discovery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                    # Display report
                    with st.expander("📖 View Full Report", expanded=True):
                        st.text(report_content)
                else:
                    if state['running']:
                        st.info("⏳ Report will be generated when discovery completes")
                    else:
                        st.warning("⚠️ No report found. Try running the discovery again.")
            else:
                st.info("Enhanced reports are disabled. Enable them in the sidebar to generate detailed reports.")
    
    with tab4:
        st.header("System Logs")
        
        if state['logs']:
            # Create a dataframe from logs
            log_df = pd.DataFrame(state['logs'])
            
            # Display logs in reverse chronological order
            for _, log in log_df.iloc[::-1].iterrows():
                level_emoji = {
                    'info': 'ℹ️',
                    'success': '✅',
                    'warning': '⚠️',
                    'error': '❌'
                }.get(log['level'], 'ℹ️')
                
                st.markdown(f"`{log['timestamp']}` {level_emoji} {log['message']}")
        else:
            st.info("No logs yet. Start a discovery to see activity logs.")

if __name__ == "__main__":
    main()
