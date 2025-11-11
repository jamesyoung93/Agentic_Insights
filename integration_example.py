"""
Integration Example - Complete Discovery Pipeline
Shows how to use all components together
"""

from pathlib import Path
from auto_enhanced_report import AutoEnhancedReportGenerator
from world_model_builder import WorldModel
import json


def run_complete_discovery_example():
    """
    Example of a complete discovery cycle using all components
    """
    
    print("=" * 80)
    print("AUTONOMOUS DISCOVERY - COMPLETE EXAMPLE")
    print("=" * 80)
    print()
    
    # ============================================================================
    # STEP 1: Initialize World Model
    # ============================================================================
    print("📊 STEP 1: Initializing World Model...")
    
    wm = WorldModel()
    wm.set_objective(
        objective="Investigate customer transaction patterns and identify opportunities",
        dataset_description="Customer transaction data over 3 years with demographics"
    )
    
    print(f"✅ World model initialized")
    print(f"   Objective: {wm.objective}")
    print()
    
    # ============================================================================
    # STEP 2: Simulate Discovery Cycles
    # ============================================================================
    print("🔄 STEP 2: Running Discovery Cycles...")
    
    for cycle in range(1, 4):
        print(f"\n   Cycle {cycle}:")
        wm.increment_cycle()
        
        # Simulate data analysis
        trajectory = wm.add_trajectory(
            trajectory_type="data_analysis",
            objective=f"Analyze correlations in cycle {cycle}",
            outputs={
                "correlation": 0.7 + (cycle * 0.05),
                "p_value": 0.001 / cycle,
                "effect_size": 0.5 + (cycle * 0.1),
                "sample_size": 1000
            }
        )
        print(f"      ✓ Added trajectory: {trajectory.id}")
        
        # Add discovery every other cycle
        if cycle % 2 == 1:
            discovery = wm.add_discovery(
                title=f"Pattern Discovery from Cycle {cycle}",
                summary=f"Significant correlation found between variables X and Y (r={trajectory.outputs['correlation']:.2f})",
                evidence=[
                    f"Pearson correlation: {trajectory.outputs['correlation']:.3f}",
                    f"P-value: {trajectory.outputs['p_value']:.4f}",
                    f"Effect size: {trajectory.outputs['effect_size']:.2f}"
                ],
                trajectory_ids=[trajectory.id],
                confidence=0.85 + (cycle * 0.05)
            )
            print(f"      ✓ Added discovery: {discovery.id}")
        
        # Add cycle summary
        wm.add_cycle_summary(
            cycle=cycle,
            summary=f"Completed analysis of segment {cycle}, found {'significant' if cycle % 2 == 1 else 'moderate'} patterns"
        )
    
    print(f"\n✅ Completed {wm.current_cycle} discovery cycles")
    print()
    
    # ============================================================================
    # STEP 3: Generate Enhanced Report
    # ============================================================================
    print("📝 STEP 3: Generating Enhanced Report...")
    
    generator = AutoEnhancedReportGenerator()
    
    # Prepare data for report generation
    cycle_results = {
        'discoveries': [d.to_dict() for d in wm.discoveries],
        'trajectories': [t.to_dict() for t in wm.trajectories],
        'world_model': {
            'objective': wm.objective,
            'dataset_description': wm.dataset_description,
            'current_cycle': wm.current_cycle
        }
    }
    
    report_path = generator.generate_from_cycle_data(cycle_results)
    print(f"✅ Enhanced report generated: {report_path}")
    print()
    
    # ============================================================================
    # STEP 4: Save World Model
    # ============================================================================
    print("💾 STEP 4: Saving World Model...")
    
    model_path = wm.save()
    print(f"✅ World model saved: {model_path}")
    print()
    
    # ============================================================================
    # STEP 5: Display Summary
    # ============================================================================
    print("=" * 80)
    print("DISCOVERY SUMMARY")
    print("=" * 80)
    
    summary = wm.get_summary()
    print(json.dumps(summary, indent=2))
    print()
    
    # ============================================================================
    # STEP 6: Generate Context for Next Cycle
    # ============================================================================
    print("=" * 80)
    print("WORLD MODEL CONTEXT (for LLM)")
    print("=" * 80)
    
    context = wm.generate_context_summary(max_discoveries=5)
    print(context)
    print()
    
    # ============================================================================
    # STEP 7: Demonstrate Reloading
    # ============================================================================
    print("=" * 80)
    print("DEMONSTRATION: Reloading Saved State")
    print("=" * 80)
    
    # Load the saved world model
    wm_loaded = WorldModel.load()
    
    print(f"✅ World model reloaded successfully")
    print(f"   Discoveries: {len(wm_loaded.discoveries)}")
    print(f"   Trajectories: {len(wm_loaded.trajectories)}")
    print(f"   Current cycle: {wm_loaded.current_cycle}")
    print()
    
    return {
        'world_model': wm,
        'report_path': report_path,
        'model_path': model_path
    }


def demonstrate_advanced_features():
    """
    Demonstrate advanced world model features
    """
    
    print("\n" + "=" * 80)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("=" * 80)
    print()
    
    wm = WorldModel()
    wm.set_objective("Advanced discovery demonstration")
    
    # ============================================================================
    # Research Questions
    # ============================================================================
    print("📋 Research Questions:")
    
    questions = [
        "What factors most strongly predict customer retention?",
        "Are there seasonal patterns in transaction behavior?",
        "Do demographic factors influence product preferences?"
    ]
    
    for q in questions:
        wm.add_research_question(q)
        print(f"   • {q}")
    
    print()
    
    # ============================================================================
    # Hypotheses
    # ============================================================================
    print("🧪 Hypotheses:")
    
    wm.add_hypothesis(
        hypothesis="Older customers prefer premium products",
        supporting_evidence=["traj_1_1", "traj_1_2"],
        status="active"
    )
    print("   • Older customers prefer premium products [ACTIVE]")
    
    wm.add_hypothesis(
        hypothesis="Transaction frequency increases in Q4",
        supporting_evidence=["traj_2_1"],
        status="supported"
    )
    print("   • Transaction frequency increases in Q4 [SUPPORTED]")
    
    print()
    
    # ============================================================================
    # Query Functions
    # ============================================================================
    print("🔍 Query Capabilities:")
    
    # Add some data first
    for i in range(3):
        wm.increment_cycle()
        wm.add_trajectory(
            trajectory_type="data_analysis" if i % 2 == 0 else "literature_search",
            objective=f"Objective {i}",
            outputs={"result": f"Result {i}"}
        )
    
    data_analyses = wm.get_trajectories_by_type("data_analysis")
    literature = wm.get_trajectories_by_type("literature_search")
    
    print(f"   • Data analysis trajectories: {len(data_analyses)}")
    print(f"   • Literature search trajectories: {len(literature)}")
    print()
    
    print("✅ Advanced features demonstration complete")
    print()


def main():
    """
    Run the complete example
    """
    
    # Show file paths
    base_dir = Path.cwd()
    print("=" * 80)
    print("FILE LOCATIONS")
    print("=" * 80)
    print(f"Working Directory: {base_dir}")
    print(f"World Model File:  {base_dir / 'world_model.json'}")
    print(f"Enhanced Report:   {base_dir / 'auto_enhanced_report.txt'}")
    print("=" * 80)
    print()
    
    # Run complete example
    results = run_complete_discovery_example()
    
    # Demonstrate advanced features
    demonstrate_advanced_features()
    
    # Final summary
    print("=" * 80)
    print("🎉 EXAMPLE COMPLETE!")
    print("=" * 80)
    print()
    print("Generated files:")
    print(f"  • World Model: {results['model_path']}")
    print(f"  • Enhanced Report: {results['report_path']}")
    print()
    print("Next steps:")
    print("  1. View the generated files")
    print("  2. Run: streamlit run streamlit_app_enhanced.py")
    print("  3. Integrate with your own discovery pipeline")
    print()


if __name__ == "__main__":
    main()
