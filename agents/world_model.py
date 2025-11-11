"""
World Model - Manages discoveries, context, and state across agent cycles
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Any

class WorldModel:
    """
    Structured world model to track discoveries, analyses, and literature findings
    Similar to Kosmos paper's approach to context management
    """
    
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        self.discoveries = []
        self.analyses = []
        self.literature_findings = []
        self.hypotheses = []
        self.current_cycle = 0
        
        os.makedirs(output_dir, exist_ok=True)
        self.state_file = f'{output_dir}/world_model_state.json'
        
        # Load existing state if available
        if os.path.exists(self.state_file):
            self.load_state()
    
    def add_analysis(self, analysis: Dict[str, Any]):
        """Add a data analysis result to the world model"""
        analysis['cycle'] = self.current_cycle
        analysis['timestamp'] = datetime.now().isoformat()
        analysis['type'] = 'analysis'
        self.analyses.append(analysis)
        self.save_state()
    
    def add_literature_finding(self, finding: Dict[str, Any]):
        """Add a literature search finding to the world model"""
        finding['cycle'] = self.current_cycle
        finding['timestamp'] = datetime.now().isoformat()
        finding['type'] = 'literature'
        self.literature_findings.append(finding)
        self.save_state()
    
    def add_discovery(self, discovery: Dict[str, Any]):
        """Add a synthesized discovery to the world model"""
        discovery['cycle'] = self.current_cycle
        discovery['timestamp'] = datetime.now().isoformat()
        discovery['type'] = 'discovery'
        
        # Link to supporting evidence
        if 'supporting_analyses' not in discovery:
            discovery['supporting_analyses'] = []
        if 'supporting_literature' not in discovery:
            discovery['supporting_literature'] = []
        
        self.discoveries.append(discovery)
        self.save_state()
    
    def add_hypothesis(self, hypothesis: Dict[str, Any]):
        """Add a hypothesis to be tested"""
        hypothesis['cycle'] = self.current_cycle
        hypothesis['timestamp'] = datetime.now().isoformat()
        hypothesis['status'] = 'pending'  # pending, supported, refuted
        self.hypotheses.append(hypothesis)
        self.save_state()
    
    def update_hypothesis(self, hypothesis_id: int, status: str, evidence: Dict = None):
        """Update hypothesis status based on findings"""
        if 0 <= hypothesis_id < len(self.hypotheses):
            self.hypotheses[hypothesis_id]['status'] = status
            if evidence:
                self.hypotheses[hypothesis_id]['evidence'] = evidence
            self.save_state()
    
    def get_context_summary(self) -> str:
        """Generate a summary of current knowledge for agents"""
        summary = f"""
CURRENT WORLD MODEL STATE (Cycle {self.current_cycle})
{'=' * 60}

DISCOVERIES ({len(self.discoveries)}):
"""
        for i, disc in enumerate(self.discoveries[-5:], 1):  # Last 5 discoveries
            summary += f"\n{i}. {disc.get('title', 'Untitled')}"
            if 'key_finding' in disc:
                summary += f"\n   → {disc['key_finding']}"
        
        summary += f"\n\nANALYSES COMPLETED ({len(self.analyses)})"
        summary += f"\nLITERATURE REVIEWED ({len(self.literature_findings)} papers)"
        
        summary += f"\n\nACTIVE HYPOTHESES ({len([h for h in self.hypotheses if h['status'] == 'pending'])}):"
        for hyp in [h for h in self.hypotheses if h['status'] == 'pending']:
            summary += f"\n- {hyp.get('hypothesis', '')}"
        
        summary += f"\n\nSUPPORTED FINDINGS ({len([h for h in self.hypotheses if h['status'] == 'supported'])}):"
        for hyp in [h for h in self.hypotheses if h['status'] == 'supported'][:3]:
            summary += f"\n- {hyp.get('hypothesis', '')}"
        
        return summary
    
    def get_recent_analyses(self, n=5) -> List[Dict]:
        """Get most recent analyses"""
        return self.analyses[-n:] if len(self.analyses) >= n else self.analyses
    
    def get_recent_literature(self, n=5) -> List[Dict]:
        """Get most recent literature findings"""
        return self.literature_findings[-n:] if len(self.literature_findings) >= n else self.literature_findings
    
    def increment_cycle(self):
        """Move to next research cycle"""
        self.current_cycle += 1
        self.save_state()
    
    def save_state(self):
        """Persist world model state to disk"""
        state = {
            'current_cycle': self.current_cycle,
            'discoveries': self.discoveries,
            'analyses': self.analyses,
            'literature_findings': self.literature_findings,
            'hypotheses': self.hypotheses,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        """Load world model state from disk"""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.current_cycle = state.get('current_cycle', 0)
            self.discoveries = state.get('discoveries', [])
            self.analyses = state.get('analyses', [])
            self.literature_findings = state.get('literature_findings', [])
            self.hypotheses = state.get('hypotheses', [])
            
            print(f"✓ Loaded world model state from cycle {self.current_cycle}")
        except Exception as e:
            print(f"Could not load previous state: {e}")
    
    def generate_report(self) -> str:
        """Generate final discovery report"""
        report = f"""
KOSMOS DISCOVERY REPORT
{'=' * 80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Cycles: {self.current_cycle}
Total Discoveries: {len(self.discoveries)}

"""
        
        # Key discoveries
        report += "\nKEY DISCOVERIES\n" + "-" * 80 + "\n"
        for i, disc in enumerate(self.discoveries, 1):
            report += f"\n{i}. {disc.get('title', 'Discovery')}\n"
            report += f"   {disc.get('description', '')}\n"
            
            if 'statistical_support' in disc:
                report += f"   Statistical Support: {disc['statistical_support']}\n"
            
            # Link to evidence
            if disc.get('supporting_analyses'):
                report += f"   Supporting Analyses: {len(disc['supporting_analyses'])}\n"
            if disc.get('supporting_literature'):
                report += f"   Supporting Literature: {len(disc['supporting_literature'])}\n"
        
        # Supported hypotheses
        supported = [h for h in self.hypotheses if h['status'] == 'supported']
        if supported:
            report += f"\n\nSUPPORTED HYPOTHESES ({len(supported)})\n" + "-" * 80 + "\n"
            for hyp in supported:
                report += f"\n✓ {hyp.get('hypothesis', '')}\n"
                if 'evidence' in hyp:
                    report += f"  Evidence: {hyp['evidence']}\n"
        
        # Summary statistics
        report += f"\n\nSUMMARY STATISTICS\n" + "-" * 80 + "\n"
        report += f"Data Analyses Performed: {len(self.analyses)}\n"
        report += f"Papers Reviewed: {len(self.literature_findings)}\n"
        report += f"Hypotheses Tested: {len(self.hypotheses)}\n"
        report += f"Supported Findings: {len(supported)}\n"
        
        return report
    
    def save_report(self, filename='discovery_report.txt'):
        """Save final report to file"""
        report = self.generate_report()
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"✓ Report saved to {filepath}")
        return filepath
