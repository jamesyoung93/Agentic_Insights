"""
Integration Guide: Enhanced Report Generator
=============================================

This script shows how to integrate the EnhancedReportGenerator with your 
existing discovery system to produce rigorous, traceable scientific reports.

Author: Assistant
Date: 2025
"""

import sys
import os

# Add the enhanced_report module to path
sys.path.insert(0, '/mnt/user-data/uploads')

from enhanced_report import EnhancedReportGenerator

# =============================================================================
# STEP 1: Enhance Your World Model
# =============================================================================

class EnhancedWorldModel:
    """
    Wrapper to ensure your world_model has all required fields for 
    the EnhancedReportGenerator
    """
    
    def __init__(self, existing_world_model=None):
        """
        Initialize with your existing world model or create new one
        
        Args:
            existing_world_model: Your current world_model object (optional)
        """
        if existing_world_model:
            # Use existing model
            self.current_cycle = getattr(existing_world_model, 'current_cycle', 0)
            self.discoveries = getattr(existing_world_model, 'discoveries', [])
            self.analyses = getattr(existing_world_model, 'analyses', [])
            self.literature_findings = getattr(existing_world_model, 'literature_findings', [])
        else:
            # Create new model
            self.current_cycle = 0
            self.discoveries = []
            self.analyses = []
            self.literature_findings = []
    
    def add_discovery(self, title, description, cycle=None, 
                     statistical_support=None, 
                     supporting_analyses=None,
                     supporting_literature=None):
        """
        Add a discovery with all the metadata needed for rigorous reporting
        
        Args:
            title: Short title of discovery
            description: Detailed description
            cycle: Which research cycle produced this
            statistical_support: Statistical evidence (p-values, effect sizes, etc.)
            supporting_analyses: List of analysis IDs that support this finding
            supporting_literature: List of paper IDs that support this finding
        """
        discovery = {
            'title': title,
            'description': description,
            'cycle': cycle or self.current_cycle,
            'statistical_support': statistical_support,
            'supporting_analyses': supporting_analyses or [],
            'supporting_literature': supporting_literature or []
        }
        self.discoveries.append(discovery)
        return discovery
    
    def add_analysis(self, question, analysis_id, code_path=None, results=None):
        """
        Add an analysis with code traceability
        
        Args:
            question: Research question this analysis addresses
            analysis_id: Unique identifier for this analysis
            code_path: Path to the code file (will be created if not exists)
            results: Dictionary of results from the analysis
        """
        analysis = {
            'question': question,
            'analysis_id': analysis_id,
            'code_path': code_path,
            'results': results or {}
        }
        self.analyses.append(analysis)
        return analysis


# =============================================================================
# STEP 2: Example Integration with Customer Satisfaction Data
# =============================================================================

def create_example_discoveries():
    """
    Example showing how to create discoveries from your customer satisfaction analysis
    with proper statistical support
    """
    
    # Initialize enhanced world model
    world_model = EnhancedWorldModel()
    world_model.current_cycle = 6  # From your screenshot
    
    # Discovery 1: Customer Complaints Impact Satisfaction
    # =====================================================
    world_model.add_discovery(
        title="Customer complaints related to product quality negatively impact satisfaction",
        description="""
        Analysis revealed a strong negative correlation between product quality complaints 
        and customer satisfaction scores. This relationship persists after controlling for 
        complaint volume and customer tenure.
        """,
        cycle=3,
        statistical_support="""
        - Pearson correlation: r = -0.67, p < 0.001, 95% CI [-0.75, -0.58]
        - Effect size (Cohen's d): 1.2 (large effect)
        - Multiple regression β = -0.58, p < 0.001 (controlling for complaint volume, tenure)
        - Explains 45% of variance in satisfaction (R² = 0.45)
        """,
        supporting_analyses=['analysis_001', 'analysis_002'],
        supporting_literature=[]
    )
    
    # Discovery 2: Timely Resolution Improves Satisfaction
    # ====================================================
    world_model.add_discovery(
        title="Prompt resolution of customer complaints leads to higher satisfaction",
        description="""
        Customer complaints resolved within 24 hours showed significantly higher 
        satisfaction scores compared to complaints resolved after 24 hours.
        Time-to-resolution is a critical factor in customer retention.
        """,
        cycle=3,
        statistical_support="""
        - Mean satisfaction difference: 1.8 points (5-point scale)
        - Independent samples t-test: t(458) = 8.3, p < 0.001
        - Effect size (Cohen's d): 0.89 (large effect)
        - 95% CI for difference: [1.4, 2.2]
        - Logistic regression: Each hour delay reduces satisfaction likelihood by 5% (OR=0.95, p<0.01)
        """,
        supporting_analyses=['analysis_003', 'analysis_004'],
        supporting_literature=[]
    )
    
    # Add corresponding analyses with code traceability
    # =================================================
    world_model.add_analysis(
        question="What is the relationship between product quality complaints and satisfaction?",
        analysis_id='analysis_001',
        code_path='outputs/analyses/analysis_001.py',
        results={
            'correlation': -0.67,
            'p_value': 0.0001,
            'effect_size': 1.2
        }
    )
    
    world_model.add_analysis(
        question="Does resolution time affect customer satisfaction?",
        analysis_id='analysis_003',
        code_path='outputs/analyses/analysis_003.py',
        results={
            't_statistic': 8.3,
            'p_value': 0.0001,
            'mean_difference': 1.8
        }
    )
    
    return world_model


# =============================================================================
# STEP 3: Generate Enhanced Report
# =============================================================================

def generate_enhanced_report(world_model, output_dir='outputs'):
    """
    Generate a comprehensive report with all statistical rigor and causal inference
    
    Args:
        world_model: Your enhanced world model
        output_dir: Where to save the report
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/analyses", exist_ok=True)
    
    # Initialize the enhanced report generator
    reporter = EnhancedReportGenerator(world_model, output_dir=output_dir)
    
    # Generate the full report
    report_path = reporter.save_report(filename='enhanced_discovery_report.txt')
    print(f"✅ Enhanced report saved to: {report_path}")
    
    # Also generate a Jupyter notebook for reproducibility
    notebook_path = reporter.generate_jupyter_notebook()
    print(f"✅ Jupyter notebook saved to: {notebook_path}")
    
    return report_path, notebook_path


# =============================================================================
# STEP 4: Run the Integration
# =============================================================================

def main():
    """
    Main execution: Create enhanced discoveries and generate report
    """
    
    print("=" * 80)
    print("GENERATING ENHANCED SCIENTIFIC REPORT")
    print("=" * 80)
    print()
    
    # Create example discoveries with proper statistical support
    print("📊 Creating enhanced world model with statistical evidence...")
    world_model = create_example_discoveries()
    print(f"   ✓ Added {len(world_model.discoveries)} discoveries")
    print(f"   ✓ Added {len(world_model.analyses)} analyses")
    print()
    
    # Generate enhanced report
    print("📝 Generating enhanced report...")
    report_path, notebook_path = generate_enhanced_report(world_model)
    print()
    
    print("=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    print()
    print("Your enhanced report includes:")
    print("  ✓ Executive summary")
    print("  ✓ Detailed methodology section")
    print("  ✓ Statistical evidence for each discovery")
    print("  ✓ Code transparency and traceability")
    print("  ✓ Causal inference assessment (Bradford Hill criteria)")
    print("  ✓ Identified confounders and alternative explanations")
    print("  ✓ Explicit limitations")
    print("  ✓ Reproducible Jupyter notebook")
    print()
    print(f"📄 Read your report: {report_path}")
    print(f"📓 Run analyses: {notebook_path}")
    

if __name__ == "__main__":
    main()


# =============================================================================
# QUICK START GUIDE
# =============================================================================

"""
QUICK START: How to Use This Integration
=========================================

1. WITH EXISTING WORLD MODEL:
   ---------------------------
   from integrate_enhanced_reports import EnhancedWorldModel, generate_enhanced_report
   
   # Wrap your existing world model
   enhanced_model = EnhancedWorldModel(your_existing_world_model)
   
   # Add statistical support to existing discoveries
   for discovery in enhanced_model.discoveries:
       discovery['statistical_support'] = "Add your stats here: p-value, CI, effect size"
   
   # Generate report
   generate_enhanced_report(enhanced_model)


2. FROM SCRATCH (RECOMMENDED):
   ---------------------------
   from integrate_enhanced_reports import EnhancedWorldModel
   
   # Create new enhanced model
   model = EnhancedWorldModel()
   model.current_cycle = 6
   
   # Add discoveries with proper statistical support
   model.add_discovery(
       title="Your discovery title",
       description="Detailed description",
       statistical_support="r = 0.67, p < 0.001, 95% CI [0.58, 0.75]",
       supporting_analyses=['analysis_001']
   )
   
   # Add the analysis that supports it
   model.add_analysis(
       question="What relationship exists?",
       analysis_id='analysis_001',
       code_path='outputs/analyses/analysis_001.py'
   )
   
   # Generate report
   from integrate_enhanced_reports import generate_enhanced_report
   generate_enhanced_report(model)


3. RUN THE EXAMPLE:
   ----------------
   python integrate_enhanced_reports.py
   
   This will create a sample report showing you the format and rigor expected.
"""
