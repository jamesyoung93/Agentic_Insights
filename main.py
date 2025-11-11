"""
Main Kosmos Framework - Orchestrates autonomous data-driven discovery
"""
import os
import sys
import openai
from typing import List, Dict, Any
import time

from agents.world_model import WorldModel
from agents.data_analyst import DataAnalysisAgent
from agents.literature_searcher import LiteratureSearchAgent
from config import (
    OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, 
    MAX_CYCLES, MAX_PARALLEL_TASKS, OUTPUT_DIR
)

openai.api_key = OPENAI_API_KEY

class KosmosFramework:
    """
    Main framework for autonomous scientific discovery
    Coordinates data analysis and literature search agents
    """
    
    def __init__(self):
        self.world_model = WorldModel(OUTPUT_DIR)
        self.data_analyst = DataAnalysisAgent()
        self.literature_searcher = LiteratureSearchAgent()
        self.research_objectives = self._load_objectives()
    
    def _load_objectives(self) -> str:
        """Load research objectives from file"""
        try:
            with open('prompts/research_objectives.txt', 'r') as f:
                return f.read()
        except:
            return "Explore the data and identify interesting patterns."
    
    def run(self, max_cycles: int = None):
        """
        Run autonomous discovery process
        
        Args:
            max_cycles: Maximum number of research cycles to run
        """
        if max_cycles is None:
            max_cycles = MAX_CYCLES
        
        print(f"""
{'=' * 80}
KOSMOS FRAMEWORK - AUTONOMOUS DATA-DRIVEN DISCOVERY
{'=' * 80}

Research Objectives:
{self.research_objectives[:200]}...

Starting {max_cycles} research cycles...
{'=' * 80}
""")
        
        for cycle in range(max_cycles):
            print(f"\n{'#' * 80}")
            print(f"# CYCLE {cycle + 1}/{max_cycles}")
            print(f"{'#' * 80}\n")
            
            # Get current context
            context = self.world_model.get_context_summary()
            
            # Generate research questions for this cycle
            questions = self._generate_research_questions(context)
            
            print(f"Generated {len(questions)} research questions for this cycle")
            
            # Execute parallel tasks (simplified - run sequentially for now)
            for i, question in enumerate(questions[:MAX_PARALLEL_TASKS], 1):
                print(f"\n--- Task {i}/{len(questions[:MAX_PARALLEL_TASKS])} ---")
                
                task_type = question['type']
                
                if task_type == 'analysis':
                    # Run data analysis
                    result = self.data_analyst.analyze(
                        question['question'],
                        context=context
                    )
                    
                    if result['success']:
                        self.world_model.add_analysis({
                            'question': question['question'],
                            'findings': result.get('summary', {}),
                            'output': result.get('output', ''),
                            'analysis_id': result.get('analysis_id')
                        })
                
                elif task_type == 'literature':
                    # Search literature
                    result = self.literature_searcher.search(
                        question['question'],
                        context=context
                    )
                    
                    if result['success']:
                        self.world_model.add_literature_finding({
                            'query': question['question'],
                            'insights': result.get('insights', {}),
                            'papers': result.get('papers', [])
                        })
                
                # Small delay to avoid rate limits
                time.sleep(1)
            
            # Synthesize discoveries from this cycle
            self._synthesize_cycle_discoveries(cycle)
            
            # Move to next cycle
            self.world_model.increment_cycle()
            
            print(f"\n✓ Cycle {cycle + 1} complete")
        
        # Generate final report
        print(f"\n{'=' * 80}")
        print("Generating final discovery report...")
        report_path = self.world_model.save_report()
        print(f"\n{'=' * 80}")
        print(f"✓ DISCOVERY PROCESS COMPLETE")
        print(f"✓ Report saved to: {report_path}")
        print(f"{'=' * 80}\n")
    
    def _generate_research_questions(self, context: str) -> List[Dict[str, str]]:
        """Generate research questions for this cycle using LLM"""
        
        prompt = f"""Based on the research objectives and current discoveries, generate 3-5 specific research questions to investigate next.

RESEARCH OBJECTIVES:
{self.research_objectives[:1000]}

CURRENT KNOWLEDGE:
{context}

Generate a mix of:
1. Data analysis questions (to be answered by analyzing the customer data)
2. Literature review questions (to be answered by searching research papers)

For each question, specify:
- type: "analysis" or "literature"
- question: the specific question to investigate

Format your response as a JSON array:
[
  {{"type": "analysis", "question": "What is the correlation between..."}},
  {{"type": "literature", "question": "What does research say about..."}}
]

Return ONLY valid JSON."""

        try:
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a research strategist designing scientific investigations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            import json
            questions = json.loads(response_text)
            
            return questions
        
        except Exception as e:
            print(f"⚠ Error generating questions: {e}")
            # Fallback questions
            return [
                {"type": "analysis", "question": "What are the key customer segments by value?"},
                {"type": "literature", "question": "What factors drive customer loyalty in retail?"}
            ]
    
    def _synthesize_cycle_discoveries(self, cycle: int):
        """Synthesize discoveries from the current cycle"""
        
        recent_analyses = self.world_model.get_recent_analyses(3)
        recent_literature = self.world_model.get_recent_literature(3)
        
        if not recent_analyses and not recent_literature:
            return
        
        # Prepare summary of findings
        analyses_summary = "\n".join([
            f"- {a.get('question', 'Unknown')}: {str(a.get('findings', {}))[:200]}"
            for a in recent_analyses
        ])
        
        literature_summary = "\n".join([
            f"- {l.get('query', 'Unknown')}: {l.get('insights', {}).get('synthesis', '')[:200]}"
            for l in recent_literature
        ])
        
        prompt = f"""Synthesize discoveries from this research cycle. Identify key insights that combine data analysis and literature.

DATA ANALYSES:
{analyses_summary}

LITERATURE FINDINGS:
{literature_summary}

Identify:
1. Key discoveries (supported by both data and literature)
2. Novel insights from data analysis
3. Contradictions or gaps to explore

Return a JSON object:
{{
  "discoveries": [
    {{
      "title": "Discovery title",
      "description": "Brief description",
      "statistical_support": "Key statistics",
      "literature_support": "Supporting papers/findings"
    }}
  ],
  "hypotheses_to_test": ["Hypothesis 1", "Hypothesis 2"]
}}

Return ONLY valid JSON."""

        try:
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You synthesize scientific discoveries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            import json
            synthesis = json.loads(response_text)
            
            # Add discoveries to world model
            for discovery in synthesis.get('discoveries', []):
                discovery['supporting_analyses'] = [a.get('analysis_id') for a in recent_analyses if a.get('analysis_id')]
                discovery['supporting_literature'] = [p['id'] for l in recent_literature for p in l.get('papers', [])]
                self.world_model.add_discovery(discovery)
                print(f"\n   💡 Discovery: {discovery.get('title', 'Untitled')}")
            
            # Add hypotheses
            for hypothesis in synthesis.get('hypotheses_to_test', []):
                self.world_model.add_hypothesis({'hypothesis': hypothesis})
        
        except Exception as e:
            print(f"⚠ Error synthesizing discoveries: {e}")


def main():
    """Main entry point"""
    
    # Check if data exists
    if not os.path.exists('data/customers.csv'):
        print("Data not found. Generating sample data...")
        os.system('python data/generate_data.py')
    
    # Check if literature exists
    if not os.path.exists('knowledge/literature_index.json'):
        print("Literature not found. Generating sample papers...")
        os.system('python knowledge/generate_literature.py')
    
    # Initialize and run framework
    kosmos = KosmosFramework()
    kosmos.run()


if __name__ == '__main__':
    main()
