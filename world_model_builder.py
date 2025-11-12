"""
World Model Builder
Manages structured knowledge representation across discovery cycles
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class Discovery:
    """Represents a single discovery"""
    id: str
    title: str
    summary: str
    evidence: List[str]
    cycle: int
    trajectory_ids: List[str]
    timestamp: str
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Trajectory:
    """Represents an analysis trajectory"""
    id: str
    cycle: int
    type: str  # 'data_analysis', 'literature_search', 'synthesis'
    objective: str
    outputs: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class WorldModel:
    """
    Structured world model for managing discovery context
    Tracks discoveries, analyses, and findings across research cycles
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the world model
        
        Args:
            base_dir: Base directory for saving model state
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.model_file = self.base_dir / "world_model.json"
        
        # Core model components
        self.objective: str = ""
        self.dataset_description: str = ""
        self.discoveries: List[Discovery] = []
        self.trajectories: List[Trajectory] = []
        self.cycle_summaries: Dict[int, str] = {}
        self.research_questions: List[str] = []
        self.hypotheses: List[Dict] = []
        
        # Metadata
        self.created_at: str = datetime.now().isoformat()
        self.updated_at: str = datetime.now().isoformat()
        self.current_cycle: int = 0
        
    def set_objective(self, objective: str, dataset_description: str = ""):
        """Set the research objective and dataset description"""
        self.objective = objective
        self.dataset_description = dataset_description
        self.updated_at = datetime.now().isoformat()
        
    def add_discovery(self, 
                     title: str,
                     summary: str,
                     evidence: List[str],
                     trajectory_ids: List[str],
                     confidence: float = 0.0) -> Discovery:
        """
        Add a new discovery to the world model
        
        Args:
            title: Discovery title
            summary: Brief summary
            evidence: List of supporting evidence
            trajectory_ids: Associated trajectory IDs
            confidence: Confidence score (0-1)
            
        Returns:
            The created Discovery object
        """
        discovery = Discovery(
            id=f"disc_{len(self.discoveries) + 1}",
            title=title,
            summary=summary,
            evidence=evidence,
            cycle=self.current_cycle,
            trajectory_ids=trajectory_ids,
            timestamp=datetime.now().isoformat(),
            confidence=confidence
        )
        
        self.discoveries.append(discovery)
        self.updated_at = datetime.now().isoformat()
        
        return discovery
    
    def add_trajectory(self,
                      trajectory_type: str,
                      objective: str,
                      outputs: Dict[str, Any]) -> Trajectory:
        """
        Add a new trajectory to the world model
        
        Args:
            trajectory_type: Type of trajectory ('data_analysis', 'literature_search', 'synthesis')
            objective: Trajectory objective
            outputs: Output data from the trajectory
            
        Returns:
            The created Trajectory object
        """
        trajectory = Trajectory(
            id=f"traj_{self.current_cycle}_{len(self.trajectories) + 1}",
            cycle=self.current_cycle,
            type=trajectory_type,
            objective=objective,
            outputs=outputs,
            timestamp=datetime.now().isoformat()
        )
        
        self.trajectories.append(trajectory)
        self.updated_at = datetime.now().isoformat()
        
        return trajectory
    
    def add_cycle_summary(self, cycle: int, summary: str):
        """Add a summary for a completed cycle"""
        self.cycle_summaries[cycle] = summary
        self.updated_at = datetime.now().isoformat()
    
    def add_research_question(self, question: str):
        """Add a research question"""
        if question not in self.research_questions:
            self.research_questions.append(question)
            self.updated_at = datetime.now().isoformat()
    
    def add_hypothesis(self, 
                      hypothesis: str, 
                      supporting_evidence: List[str],
                      status: str = "active"):
        """
        Add a hypothesis to test
        
        Args:
            hypothesis: The hypothesis statement
            supporting_evidence: Supporting evidence IDs
            status: Status ('active', 'supported', 'refuted', 'inconclusive')
        """
        hyp = {
            'id': f"hyp_{len(self.hypotheses) + 1}",
            'hypothesis': hypothesis,
            'supporting_evidence': supporting_evidence,
            'status': status,
            'cycle': self.current_cycle,
            'timestamp': datetime.now().isoformat()
        }
        
        self.hypotheses.append(hyp)
        self.updated_at = datetime.now().isoformat()
    
    def get_discoveries_by_cycle(self, cycle: int) -> List[Discovery]:
        """Get all discoveries from a specific cycle"""
        return [d for d in self.discoveries if d.cycle == cycle]
    
    def get_trajectories_by_cycle(self, cycle: int) -> List[Trajectory]:
        """Get all trajectories from a specific cycle"""
        return [t for t in self.trajectories if t.cycle == cycle]
    
    def get_trajectories_by_type(self, traj_type: str) -> List[Trajectory]:
        """Get all trajectories of a specific type"""
        return [t for t in self.trajectories if t.type == traj_type]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the world model state"""
        return {
            'objective': self.objective,
            'dataset_description': self.dataset_description,
            'current_cycle': self.current_cycle,
            'total_discoveries': len(self.discoveries),
            'total_trajectories': len(self.trajectories),
            'discoveries_by_cycle': {
                cycle: len(self.get_discoveries_by_cycle(cycle))
                for cycle in range(self.current_cycle + 1)
            },
            'trajectories_by_type': {
                traj_type: len(self.get_trajectories_by_type(traj_type))
                for traj_type in ['data_analysis', 'literature_search', 'synthesis']
            },
            'research_questions': len(self.research_questions),
            'active_hypotheses': len([h for h in self.hypotheses if h['status'] == 'active']),
            'updated_at': self.updated_at
        }
    
    def increment_cycle(self):
        """Move to the next cycle"""
        self.current_cycle += 1
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert world model to dictionary"""
        return {
            'objective': self.objective,
            'dataset_description': self.dataset_description,
            'discoveries': [d.to_dict() for d in self.discoveries],
            'trajectories': [t.to_dict() for t in self.trajectories],
            'cycle_summaries': self.cycle_summaries,
            'research_questions': self.research_questions,
            'hypotheses': self.hypotheses,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'current_cycle': self.current_cycle
        }
    
    def save(self) -> str:
        """
        Save the world model to disk
        
        Returns:
            Path to the saved file
        """
        # Ensure directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(self.model_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"✅ World model saved to: {self.model_file}")
        return str(self.model_file)
    
    @classmethod
    def load(cls, base_dir: Optional[Path] = None) -> 'WorldModel':
        """
        Load a world model from disk
        
        Args:
            base_dir: Base directory where model is saved
            
        Returns:
            Loaded WorldModel instance
        """
        base_dir = Path(base_dir) if base_dir else Path.cwd()
        model_file = base_dir / "world_model.json"
        
        if not model_file.exists():
            print(f"⚠️ No saved model found at {model_file}, creating new model")
            return cls(base_dir=base_dir)
        
        # Load from JSON
        with open(model_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create instance and populate
        model = cls(base_dir=base_dir)
        model.objective = data.get('objective', '')
        model.dataset_description = data.get('dataset_description', '')
        model.cycle_summaries = data.get('cycle_summaries', {})
        model.research_questions = data.get('research_questions', [])
        model.hypotheses = data.get('hypotheses', [])
        model.created_at = data.get('created_at', datetime.now().isoformat())
        model.updated_at = data.get('updated_at', datetime.now().isoformat())
        model.current_cycle = data.get('current_cycle', 0)
        
        # Reconstruct discoveries
        for disc_data in data.get('discoveries', []):
            discovery = Discovery(**disc_data)
            model.discoveries.append(discovery)
        
        # Reconstruct trajectories
        for traj_data in data.get('trajectories', []):
            trajectory = Trajectory(**traj_data)
            model.trajectories.append(trajectory)
        
        print(f"✅ World model loaded from: {model_file}")
        return model
    
    def generate_context_summary(self, max_discoveries: int = 10) -> str:
        """
        Generate a text summary of the world model for LLM context
        
        Args:
            max_discoveries: Maximum number of recent discoveries to include
            
        Returns:
            Formatted text summary
        """
        lines = []
        lines.append("=" * 80)
        lines.append("WORLD MODEL CONTEXT")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"OBJECTIVE: {self.objective}")
        if self.dataset_description:
            lines.append(f"DATASET: {self.dataset_description}")
        lines.append(f"CURRENT CYCLE: {self.current_cycle}")
        lines.append("")
        
        # Recent discoveries
        recent_discoveries = sorted(self.discoveries, 
                                   key=lambda d: d.cycle, 
                                   reverse=True)[:max_discoveries]
        
        if recent_discoveries:
            lines.append("RECENT DISCOVERIES:")
            lines.append("-" * 80)
            for disc in recent_discoveries:
                lines.append(f"  • [{disc.id}] {disc.title} (Cycle {disc.cycle})")
                lines.append(f"    {disc.summary}")
            lines.append("")
        
        # Active research questions
        if self.research_questions:
            lines.append("RESEARCH QUESTIONS:")
            lines.append("-" * 80)
            for i, q in enumerate(self.research_questions, 1):
                lines.append(f"  {i}. {q}")
            lines.append("")
        
        # Active hypotheses
        active_hyps = [h for h in self.hypotheses if h['status'] == 'active']
        if active_hyps:
            lines.append("ACTIVE HYPOTHESES:")
            lines.append("-" * 80)
            for hyp in active_hyps:
                lines.append(f"  • {hyp['hypothesis']}")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Example usage"""
    # Create a new world model
    wm = WorldModel()
    
    # Set objective
    wm.set_objective(
        objective="Investigate correlations in customer transaction data",
        dataset_description="Customer transactions over 3 years with demographic info"
    )
    
    # Add some discoveries
    wm.add_discovery(
        title="Strong correlation between age and product category",
        summary="Customers over 50 prefer category A, while younger prefer B",
        evidence=["Statistical test p<0.001", "Effect size Cohen's d=0.8"],
        trajectory_ids=["traj_1_1", "traj_1_2"],
        confidence=0.95
    )
    
    # Add a trajectory
    wm.add_trajectory(
        trajectory_type="data_analysis",
        objective="Test correlation between age and category preference",
        outputs={
            "correlation": 0.85,
            "p_value": 0.0001,
            "effect_size": 0.8
        }
    )
    
    # Save the model
    wm.save()
    
    # Print summary
    print("\n" + "=" * 80)
    print("WORLD MODEL SUMMARY")
    print("=" * 80)
    import json
    print(json.dumps(wm.get_summary(), indent=2))
    
    # Generate context summary
    print("\n" + wm.generate_context_summary())


if __name__ == "__main__":
    main()
