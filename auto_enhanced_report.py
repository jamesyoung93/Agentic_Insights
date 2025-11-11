"""
Auto-Enhanced Report Generator
Automatically extracts statistical evidence from analyses and formats them into rigorous reports
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

class AutoEnhancedReportGenerator:
    """Generates enhanced discovery reports with auto-extracted statistics"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the generator
        
        Args:
            base_dir: Base directory for file operations (defaults to current working directory)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.output_file = self.base_dir / "auto_enhanced_report.txt"
        
    def extract_statistics_from_trajectory(self, trajectory_data: Dict) -> List[Dict]:
        """Extract statistical evidence from a trajectory"""
        stats = []

        # Look for common statistical patterns in outputs
        if 'outputs' in trajectory_data:
            outputs = trajectory_data['outputs']

            # Handle dict of dicts structure (from streamlit_app_ULTIMATE.py)
            if isinstance(outputs, dict):
                for finding_key, finding_stats in outputs.items():
                    if isinstance(finding_stats, dict):
                        # Extract all statistical measures for this finding
                        formatted_stats = self._format_finding_statistics(finding_key, finding_stats)
                        if formatted_stats:
                            stats.extend(formatted_stats)
            # Handle list structure (legacy support)
            elif isinstance(outputs, list):
                for output in outputs:
                    if isinstance(output, dict):
                        for key, value in output.items():
                            if any(stat_word in key.lower() for stat_word in
                                   ['p_value', 'pvalue', 'p-value', 'correlation', 'coefficient',
                                    't_stat', 'z_score', 'effect_size', 'confidence']):
                                stats.append({
                                    'type': key,
                                    'value': value,
                                    'source': trajectory_data.get('id', 'unknown')
                                })

        return stats

    def _format_finding_statistics(self, finding_key: str, stats_dict: Dict) -> List[Dict]:
        """Format statistics for a specific finding"""
        formatted = []

        # Create a readable finding name
        finding_name = finding_key.replace('_', ' ').title()

        # Extract and format each statistic
        if 'p_value' in stats_dict:
            formatted.append({
                'type': f"{finding_name} - p-value",
                'value': f"{stats_dict['p_value']:.4f}" if isinstance(stats_dict['p_value'], (int, float)) else stats_dict['p_value'],
                'source': finding_key
            })

        if 'correlation' in stats_dict:
            formatted.append({
                'type': f"{finding_name} - Correlation",
                'value': f"r = {stats_dict['correlation']:.3f}" if isinstance(stats_dict['correlation'], (int, float)) else stats_dict['correlation'],
                'source': finding_key
            })

        if 't_statistic' in stats_dict:
            formatted.append({
                'type': f"{finding_name} - t-statistic",
                'value': f"t = {stats_dict['t_statistic']:.3f}" if isinstance(stats_dict['t_statistic'], (int, float)) else stats_dict['t_statistic'],
                'source': finding_key
            })

        if 'f_statistic' in stats_dict:
            df_between = stats_dict.get('df_between', '?')
            df_within = stats_dict.get('df_within', '?')
            formatted.append({
                'type': f"{finding_name} - F-statistic",
                'value': f"F({df_between}, {df_within}) = {stats_dict['f_statistic']:.3f}" if isinstance(stats_dict['f_statistic'], (int, float)) else stats_dict['f_statistic'],
                'source': finding_key
            })

        if 'cohens_d' in stats_dict:
            effect_label = stats_dict.get('effect_size_label', '')
            formatted.append({
                'type': f"{finding_name} - Effect Size (Cohen's d)",
                'value': f"d = {stats_dict['cohens_d']:.3f} ({effect_label})" if isinstance(stats_dict['cohens_d'], (int, float)) else stats_dict['cohens_d'],
                'source': finding_key
            })

        if 'eta_squared' in stats_dict:
            formatted.append({
                'type': f"{finding_name} - Effect Size (η²)",
                'value': f"η² = {stats_dict['eta_squared']:.3f}" if isinstance(stats_dict['eta_squared'], (int, float)) else stats_dict['eta_squared'],
                'source': finding_key
            })

        if 'r_squared' in stats_dict:
            formatted.append({
                'type': f"{finding_name} - R²",
                'value': f"R² = {stats_dict['r_squared']:.3f}" if isinstance(stats_dict['r_squared'], (int, float)) else stats_dict['r_squared'],
                'source': finding_key
            })

        if 'slope' in stats_dict:
            formatted.append({
                'type': f"{finding_name} - Regression Slope",
                'value': f"β = {stats_dict['slope']:.4f}" if isinstance(stats_dict['slope'], (int, float)) else stats_dict['slope'],
                'source': finding_key
            })

        if 'n' in stats_dict:
            formatted.append({
                'type': f"{finding_name} - Sample Size",
                'value': f"n = {stats_dict['n']}" if isinstance(stats_dict['n'], (int, float)) else stats_dict['n'],
                'source': finding_key
            })

        # Add interpretation if available
        if 'interpretation' in stats_dict:
            formatted.append({
                'type': f"{finding_name} - Interpretation",
                'value': stats_dict['interpretation'],
                'source': finding_key
            })

        return formatted
    
    def format_discovery_with_stats(self, discovery: Dict, stats: List[Dict]) -> str:
        """Format a discovery with its supporting statistics"""
        formatted = f"\n{'='*80}\n"
        formatted += f"DISCOVERY: {discovery.get('title', 'Untitled Discovery')}\n"
        formatted += f"{'='*80}\n\n"

        # Summary
        formatted += "SUMMARY:\n"
        formatted += f"{discovery.get('summary', 'No summary available')}\n\n"

        # Statistical Support (from LLM synthesis or direct stats)
        if discovery.get('statistical_support'):
            formatted += "STATISTICAL SUPPORT:\n"
            formatted += "-" * 80 + "\n"
            formatted += f"{discovery['statistical_support']}\n\n"

        # Detailed Statistical Evidence (from trajectories)
        if stats:
            formatted += "DETAILED STATISTICAL EVIDENCE:\n"
            formatted += "-" * 80 + "\n"
            for i, stat in enumerate(stats, 1):
                formatted += f"{i}. {stat['type']}: {stat['value']}\n"
            formatted += "\n"
        elif not discovery.get('statistical_support'):
            formatted += "STATISTICAL EVIDENCE: Statistical details not available.\n\n"
        
        # Methodology
        if 'methodology' in discovery:
            formatted += "METHODOLOGY:\n"
            formatted += "-" * 80 + "\n"
            formatted += f"{discovery['methodology']}\n\n"
        
        # Key Findings
        if 'findings' in discovery:
            formatted += "KEY FINDINGS:\n"
            formatted += "-" * 80 + "\n"
            for i, finding in enumerate(discovery['findings'], 1):
                formatted += f"{i}. {finding}\n"
            formatted += "\n"
        
        # Limitations and Future Directions
        if 'limitations' in discovery:
            formatted += "LIMITATIONS:\n"
            formatted += "-" * 80 + "\n"
            formatted += f"{discovery['limitations']}\n\n"
        
        return formatted
    
    def generate_enhanced_report(self, 
                                discoveries: List[Dict], 
                                trajectories: List[Dict],
                                world_model: Optional[Dict] = None) -> str:
        """
        Generate an enhanced report with auto-extracted statistics
        
        Args:
            discoveries: List of discovery dictionaries
            trajectories: List of trajectory dictionaries with analysis outputs
            world_model: Optional world model context
            
        Returns:
            Path to the generated report file
        """
        print("📊 Generating enhanced report with auto-extracted statistics...")
        
        # Extract statistics from all trajectories
        all_stats = {}
        for traj in trajectories:
            traj_stats = self.extract_statistics_from_trajectory(traj)
            if traj_stats:
                all_stats[traj.get('id', 'unknown')] = traj_stats
        
        # Build the report
        report_content = []
        report_content.append("=" * 80)
        report_content.append("AUTONOMOUS DISCOVERY REPORT")
        report_content.append("Enhanced with Auto-Extracted Statistical Evidence")
        report_content.append("=" * 80)
        report_content.append("")
        
        # Add world model context if available
        if world_model:
            report_content.append("RESEARCH CONTEXT:")
            report_content.append("-" * 80)
            if 'objective' in world_model:
                report_content.append(f"Objective: {world_model['objective']}")
            if 'dataset_description' in world_model:
                report_content.append(f"Dataset: {world_model['dataset_description']}")
            report_content.append("")
        
        # Add each discovery with its statistics
        for i, discovery in enumerate(discoveries, 1):
            report_content.append(f"\n{'#'*80}")
            report_content.append(f"# DISCOVERY {i} of {len(discoveries)}")
            report_content.append(f"{'#'*80}\n")
            
            # Get relevant stats for this discovery
            discovery_stats = []
            if 'trajectory_ids' in discovery:
                for traj_id in discovery['trajectory_ids']:
                    if traj_id in all_stats:
                        discovery_stats.extend(all_stats[traj_id])
            
            # Format the discovery
            formatted = self.format_discovery_with_stats(discovery, discovery_stats)
            report_content.append(formatted)
        
        # Add summary statistics
        report_content.append("\n" + "=" * 80)
        report_content.append("SUMMARY STATISTICS")
        report_content.append("=" * 80)
        report_content.append(f"Total Discoveries: {len(discoveries)}")
        report_content.append(f"Total Trajectories Analyzed: {len(trajectories)}")
        report_content.append(f"Trajectories with Statistical Evidence: {len(all_stats)}")
        report_content.append("")
        
        # Write to file
        report_text = "\n".join(report_content)
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Write the report
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✅ Enhanced report saved to: {self.output_file}")
        return str(self.output_file)
    
    def generate_from_cycle_data(self, cycle_results: Dict) -> str:
        """
        Generate enhanced report from cycle results data
        
        Args:
            cycle_results: Dictionary containing cycle results with discoveries and trajectories
            
        Returns:
            Path to the generated report file
        """
        discoveries = cycle_results.get('discoveries', [])
        trajectories = cycle_results.get('trajectories', [])
        world_model = cycle_results.get('world_model')
        
        return self.generate_enhanced_report(discoveries, trajectories, world_model)


def main():
    """Example usage"""
    # Example data structure
    example_discoveries = [
        {
            'title': 'Example Discovery',
            'summary': 'This is an example discovery showing the format.',
            'methodology': 'Statistical analysis was performed using appropriate methods.',
            'findings': [
                'Finding 1: Significant correlation observed',
                'Finding 2: Effect size was moderate'
            ],
            'limitations': 'Sample size was limited.',
            'trajectory_ids': ['traj_001', 'traj_002']
        }
    ]
    
    example_trajectories = [
        {
            'id': 'traj_001',
            'outputs': {
                'p_value': 0.001,
                'correlation': 0.85,
                'effect_size': 0.6
            }
        }
    ]
    
    example_world_model = {
        'objective': 'Investigate the relationship between X and Y',
        'dataset_description': 'Dataset containing measurements from experiment'
    }
    
    # Generate report
    generator = AutoEnhancedReportGenerator()
    report_path = generator.generate_enhanced_report(
        example_discoveries,
        example_trajectories,
        example_world_model
    )
    
    print(f"Report generated at: {report_path}")


if __name__ == "__main__":
    main()
