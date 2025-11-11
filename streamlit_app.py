"""
Streamlit Web Interface for Kosmos Framework
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
import sys
import time

# Page config
st.set_page_config(
    page_title="Kosmos AI Scientist",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'run_started' not in st.session_state:
    st.session_state.run_started = False
if 'current_cycle' not in st.session_state:
    st.session_state.current_cycle = 0
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def check_setup():
    """Check if system is ready to run"""
    issues = []
    
    # Check API key
    try:
        from config import OPENAI_API_KEY
        if OPENAI_API_KEY == "your-api-key-here":
            issues.append("⚠️ OpenAI API key not configured")
    except:
        issues.append("⚠️ Config file not found")
    
    # Check data
    if not os.path.exists('data/customers.csv'):
        issues.append("⚠️ Data files not generated")
    
    # Check literature
    if not os.path.exists('knowledge/literature_index.json'):
        issues.append("⚠️ Literature not generated")
    
    return issues

def load_world_model_state():
    """Load current world model state"""
    try:
        with open('outputs/world_model_state.json', 'r') as f:
            return json.load(f)
    except:
        return None

def load_report():
    """Load discovery report if exists"""
    try:
        with open('outputs/discovery_report.txt', 'r') as f:
            return f.read()
    except:
        return None

# Sidebar
with st.sidebar:
    st.title("🔬 Kosmos AI Scientist")
    st.markdown("*Autonomous Data-Driven Discovery*")
    st.divider()
    
    # Configuration section
    st.subheader("⚙️ Configuration")
    
    # API Key
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value="",
        help="Your OpenAI API key (starts with sk-)"
    )
    
    if st.button("💾 Save API Key"):
        if api_key and api_key.startswith('sk-'):
            # Update config file
            config_path = 'config.py'
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            config_content = config_content.replace(
                'OPENAI_API_KEY = "your-api-key-here"',
                f'OPENAI_API_KEY = "{api_key}"'
            )
            
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            st.success("✅ API key saved!")
        else:
            st.error("Invalid API key format")
    
    st.divider()
    
    # Run settings
    st.subheader("🎯 Run Settings")
    max_cycles = st.slider("Research Cycles", 3, 20, 10)
    model_choice = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"])
    
    # Update config
    if st.button("💾 Save Settings"):
        config_path = 'config.py'
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        with open(config_path, 'w') as f:
            for line in lines:
                if line.startswith('MAX_CYCLES = '):
                    f.write(f'MAX_CYCLES = {max_cycles}\n')
                elif line.startswith('MODEL_NAME = '):
                    f.write(f'MODEL_NAME = "{model_choice}"\n')
                else:
                    f.write(line)
        
        st.success("✅ Settings saved!")
    
    # Estimated cost
    calls_per_cycle = 8
    total_calls = calls_per_cycle * max_cycles
    cost_per_call = 0.05 if 'gpt-4' in model_choice else 0.01
    estimated_cost = total_calls * cost_per_call
    
    st.info(f"""
    **Estimated Cost:**
    - {total_calls} API calls
    - ~${estimated_cost:.2f}
    - ~{max_cycles * 2}-{max_cycles * 4} min
    """)
    
    st.divider()
    
    # System status
    st.subheader("🔍 System Status")
    issues = check_setup()
    
    if not issues:
        st.success("✅ Ready to run")
    else:
        for issue in issues:
            st.warning(issue)
        
        if st.button("🔧 Run Setup"):
            with st.spinner("Setting up..."):
                os.system('python data/generate_data.py')
                os.system('python knowledge/generate_literature.py')
                st.success("Setup complete! Refresh page.")

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home", 
    "📝 Research Objectives", 
    "▶️ Run Discovery", 
    "📊 Results",
    "📚 Data Explorer"
])

with tab1:
    st.title("🔬 Kosmos AI Scientist")
    st.markdown("### Autonomous Data-Driven Discovery System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Research Domain", "Coffee Shop Analytics")
        st.metric("Customers", "4,992")
    
    with col2:
        st.metric("Transactions", "1.5M+")
        st.metric("Time Range", "2 years")
    
    with col3:
        st.metric("Literature Papers", "8")
        st.metric("Geographic Regions", "4")
    
    st.divider()
    
    st.markdown("""
    ### 🎯 What Kosmos Does
    
    Kosmos autonomously:
    1. **Generates research questions** from your objectives
    2. **Writes & executes Python code** to analyze data
    3. **Searches literature** for supporting evidence
    4. **Synthesizes discoveries** across multiple cycles
    5. **Produces traceable reports** linking code & papers
    
    ### 🚀 Quick Start
    
    1. **Configure** → Add your OpenAI API key in the sidebar
    2. **Customize** → Edit research objectives in the "Research Objectives" tab
    3. **Run** → Start discovery in the "Run Discovery" tab
    4. **Review** → View results in the "Results" tab
    
    ### 📖 How It Works
    
    Each research cycle:
    - Generates 3-5 research questions based on objectives
    - Launches parallel data analysis agents (write Python code)
    - Launches literature search agents (search papers)
    - Synthesizes findings into discoveries
    - Updates world model with new knowledge
    - Proposes hypotheses for next cycle
    """)

with tab2:
    st.title("📝 Research Objectives")
    st.markdown("Edit these to guide what Kosmos investigates")
    
    # Load current objectives
    try:
        with open('prompts/research_objectives.txt', 'r') as f:
            current_objectives = f.read()
    except:
        current_objectives = "# Add your research objectives here"
    
    # Editor
    new_objectives = st.text_area(
        "Research Objectives",
        value=current_objectives,
        height=400,
        help="Define your research goals, questions, and hypotheses"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💾 Save Objectives", type="primary"):
            with open('prompts/research_objectives.txt', 'w') as f:
                f.write(new_objectives)
            st.success("✅ Objectives saved!")
    
    with col2:
        if st.button("↩️ Reset to Default"):
            default_objectives = """# Research Objectives

## Primary Objective
Identify key drivers of customer loyalty and revenue in the coffee shop chain business.

## Specific Questions to Explore
1. What customer segments have the highest lifetime value?
2. How do seasonal patterns affect revenue across different regions?
3. What is the relationship between mobile app usage and customer retention?
4. How does competitor activity impact our customer behavior?

## Hypotheses to Test
- Loyalty program members have significantly higher lifetime value
- Mobile app users are more resistant to competitor switching
- Wait times above 5 minutes significantly impact satisfaction
"""
            with open('prompts/research_objectives.txt', 'w') as f:
                f.write(default_objectives)
            st.success("✅ Reset to default!")
            st.rerun()

with tab3:
    st.title("▶️ Run Discovery")
    
    # Check setup
    issues = check_setup()
    
    if issues:
        st.error("⚠️ Setup incomplete:")
        for issue in issues:
            st.write(issue)
        st.info("Fix issues in the sidebar before running")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if not st.session_state.run_started:
                if st.button("🚀 Start Discovery", type="primary", use_container_width=True):
                    st.session_state.run_started = True
                    st.session_state.log_messages = []
                    st.rerun()
            else:
                if st.button("⏹️ Stop", type="secondary", use_container_width=True):
                    st.session_state.run_started = False
                    st.warning("Stopped")
        
        with col2:
            if st.session_state.run_started:
                st.info(f"🔄 Running... (Cycle {st.session_state.current_cycle})")
        
        st.divider()
        
        # Progress
        if st.session_state.run_started:
            progress_placeholder = st.empty()
            log_placeholder = st.empty()
            
            # Import and run
            from config import MAX_CYCLES
            
            with st.spinner("Initializing..."):
                from main import KosmosFramework
                
                kosmos = KosmosFramework()
                
                # Custom run with progress updates
                for cycle in range(MAX_CYCLES):
                    if not st.session_state.run_started:
                        break
                    
                    st.session_state.current_cycle = cycle + 1
                    
                    # Update progress
                    progress_placeholder.progress(cycle / MAX_CYCLES)
                    
                    # Add log message
                    msg = f"Cycle {cycle + 1}/{MAX_CYCLES} - Generating research questions..."
                    st.session_state.log_messages.append(msg)
                    
                    # Display log
                    log_placeholder.text_area(
                        "Activity Log",
                        value="\n".join(st.session_state.log_messages[-20:]),
                        height=300
                    )
                    
                    # Run cycle (simplified - actual implementation would need more work)
                    try:
                        context = kosmos.world_model.get_context_summary()
                        questions = kosmos._generate_research_questions(context)
                        
                        for i, q in enumerate(questions[:3], 1):
                            msg = f"  Task {i}: {q['question'][:60]}..."
                            st.session_state.log_messages.append(msg)
                            log_placeholder.text_area(
                                "Activity Log",
                                value="\n".join(st.session_state.log_messages[-20:]),
                                height=300
                            )
                            
                            if q['type'] == 'analysis':
                                result = kosmos.data_analyst.analyze(q['question'], context=context)
                                if result.get('success'):
                                    kosmos.world_model.add_analysis({
                                        'question': q['question'],
                                        'findings': result.get('summary', {}),
                                    })
                            
                            time.sleep(1)
                        
                        kosmos._synthesize_cycle_discoveries(cycle)
                        kosmos.world_model.increment_cycle()
                        
                        msg = f"✓ Cycle {cycle + 1} complete"
                        st.session_state.log_messages.append(msg)
                        
                    except Exception as e:
                        msg = f"❌ Error in cycle {cycle + 1}: {str(e)}"
                        st.session_state.log_messages.append(msg)
                
                # Save report
                kosmos.world_model.save_report()
                
                progress_placeholder.progress(1.0)
                st.success("✅ Discovery complete! Check Results tab")
                st.session_state.run_started = False
        
        else:
            # Show world model state if exists
            state = load_world_model_state()
            if state:
                st.subheader("📊 Previous Run Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Cycles", state.get('current_cycle', 0))
                col2.metric("Discoveries", len(state.get('discoveries', [])))
                col3.metric("Analyses", len(state.get('analyses', [])))
                col4.metric("Papers Read", len(state.get('literature_findings', [])))

with tab4:
    st.title("📊 Discovery Results")
    
    # Load report
    report = load_report()
    
    if report:
        st.markdown("### 📄 Discovery Report")
        st.text_area("Report", value=report, height=600)
        
        # Download button
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"discovery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        st.divider()
        
        # World model visualization
        state = load_world_model_state()
        if state:
            st.markdown("### 🌍 World Model State")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Discoveries", len(state.get('discoveries', [])))
                st.metric("Analyses Performed", len(state.get('analyses', [])))
                st.metric("Papers Reviewed", len(state.get('literature_findings', [])))
            
            with col2:
                hypotheses = state.get('hypotheses', [])
                supported = len([h for h in hypotheses if h.get('status') == 'supported'])
                pending = len([h for h in hypotheses if h.get('status') == 'pending'])
                
                st.metric("Supported Hypotheses", supported)
                st.metric("Pending Hypotheses", pending)
                st.metric("Total Hypotheses", len(hypotheses))
            
            # Discoveries table
            if state.get('discoveries'):
                st.markdown("### 💡 Key Discoveries")
                
                discoveries_df = pd.DataFrame([
                    {
                        'Title': d.get('title', 'Untitled'),
                        'Cycle': d.get('cycle', 0),
                        'Description': d.get('description', '')[:100] + '...'
                    }
                    for d in state['discoveries']
                ])
                
                st.dataframe(discoveries_df, use_container_width=True)
    
    else:
        st.info("No results yet. Run a discovery first!")

with tab5:
    st.title("📚 Data Explorer")
    
    st.markdown("### 📊 Available Datasets")
    
    # Load and display data
    data_files = {
        'Customers': 'data/customers.csv',
        'Transactions': 'data/transactions.csv',
        'Competitor Data': 'data/competitor_data.csv'
    }
    
    selected_data = st.selectbox("Select Dataset", list(data_files.keys()))
    
    try:
        df = pd.read_csv(data_files[selected_data])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Memory", f"{df.memory_usage().sum() / 1024**2:.1f} MB")
        
        st.divider()
        
        # Show data
        st.dataframe(df.head(100), use_container_width=True)
        
        st.divider()
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Generate data first using the sidebar setup button")
    
    st.divider()
    
    # Literature
    st.markdown("### 📚 Research Papers")
    
    try:
        with open('knowledge/literature_index.json', 'r') as f:
            lit_index = json.load(f)
        
        papers = lit_index.get('papers', [])
        
        if papers:
            papers_df = pd.DataFrame([
                {
                    'ID': p.get('id', ''),
                    'Title': p.get('title', ''),
                    'Keywords': ', '.join(p.get('keywords', [])[:3])
                }
                for p in papers
            ])
            
            st.dataframe(papers_df, use_container_width=True)
            
            # View paper
            selected_paper = st.selectbox("View Paper", [p.get('title', '') for p in papers])
            
            if selected_paper:
                paper_id = next(p['id'] for p in papers if p.get('title') == selected_paper)
                paper_path = f"knowledge/literature/{paper_id}.txt"
                
                try:
                    with open(paper_path, 'r') as f:
                        paper_content = f.read()
                    
                    st.text_area("Paper Content", value=paper_content, height=400)
                except:
                    st.error("Could not load paper")
        
        else:
            st.info("No papers indexed. Run setup first.")
    
    except:
        st.info("Literature not generated. Use sidebar setup button.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Kosmos Framework - Autonomous Data-Driven Discovery</p>
    <p>Inspired by the Kosmos paper (arXiv:2511.02824)</p>
</div>
""", unsafe_allow_html=True)
