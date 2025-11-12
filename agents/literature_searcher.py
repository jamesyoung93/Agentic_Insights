"""
Literature Search Agent - Searches and extracts insights from research papers
"""
import os
import json
from typing import List, Dict, Any
import openai
from config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE

openai.api_key = OPENAI_API_KEY

class LiteratureSearchAgent:
    """
    Agent that searches literature and extracts relevant findings
    """
    
    def __init__(self, literature_dir='knowledge/literature'):
        self.literature_dir = literature_dir
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load literature index"""
        index_path = 'knowledge/literature_index.json'
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                return json.load(f)
        return {'papers': [], 'keywords_index': {}, 'authors_index': {}}
    
    def search(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Search literature for relevant papers and extract insights
        
        Args:
            query: Research question or topic to search
            context: Additional context from world model
            
        Returns:
            Dictionary with search results and extracted insights
        """
        print(f"\n📚 Searching literature: {query[:80]}...")
        
        # Find relevant papers
        relevant_papers = self._find_relevant_papers(query)
        
        if not relevant_papers:
            return {
                'success': False,
                'message': 'No relevant papers found'
            }
        
        print(f"   Found {len(relevant_papers)} relevant papers")
        
        # Extract insights from papers
        insights = self._extract_insights(query, relevant_papers, context)
        
        return {
            'success': True,
            'papers': relevant_papers,
            'insights': insights,
            'query': query
        }
    
    def _find_relevant_papers(self, query: str, max_papers: int = 5) -> List[Dict]:
        """Find papers relevant to query using keyword matching and LLM ranking"""
        
        # Extract keywords from query using LLM
        keywords = self._extract_keywords(query)
        
        # Find papers matching keywords
        candidate_papers = set()
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Check keyword index
            for indexed_keyword, paper_ids in self.index['keywords_index'].items():
                if keyword_lower in indexed_keyword.lower():
                    candidate_papers.update(paper_ids)
        
        # If no keyword matches, consider all papers
        if not candidate_papers:
            candidate_papers = set(p['id'] for p in self.index['papers'])
        
        # Load paper metadata
        papers = []
        for paper_id in candidate_papers:
            paper = next((p for p in self.index['papers'] if p['id'] == paper_id), None)
            if paper:
                papers.append(paper)
        
        # Rank papers by relevance using LLM
        if len(papers) > max_papers:
            papers = self._rank_papers(query, papers, max_papers)
        
        return papers[:max_papers]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from query"""
        
        prompt = f"""Extract 3-5 key terms/keywords from this research query that would be useful for searching academic papers.

Query: {query}

Return ONLY a comma-separated list of keywords."""

        try:
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You extract keywords from research queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [k.strip() for k in keywords_text.split(',')]
            return keywords
        
        except Exception as e:
            print(f"   ⚠ Error extracting keywords: {e}")
            # Fallback to simple word extraction
            return query.lower().split()[:5]
    
    def _rank_papers(self, query: str, papers: List[Dict], top_n: int) -> List[Dict]:
        """Rank papers by relevance to query"""
        
        papers_summary = "\n".join([
            f"{i+1}. {p['title']} ({p['year']}) - {p['abstract'][:200]}"
            for i, p in enumerate(papers)
        ])
        
        prompt = f"""Rank these papers by relevance to the research query. Return the IDs of the top {top_n} most relevant papers.

Query: {query}

Papers:
{papers_summary}

Return ONLY a comma-separated list of paper numbers (1, 2, 3, etc.) in order of relevance."""

        try:
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You rank academic papers by relevance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            ranking_text = response.choices[0].message.content.strip()
            rankings = [int(n.strip()) - 1 for n in ranking_text.split(',') if n.strip().isdigit()]
            
            # Reorder papers
            ranked_papers = [papers[i] for i in rankings if 0 <= i < len(papers)]
            # Add any missed papers
            ranked_papers += [p for p in papers if p not in ranked_papers]
            
            return ranked_papers
        
        except Exception as e:
            print(f"   ⚠ Error ranking papers: {e}")
            return papers  # Return original order if ranking fails
    
    def _extract_insights(self, query: str, papers: List[Dict], context: str) -> Dict[str, Any]:
        """Extract relevant insights from papers using LLM"""
        
        # Load full paper texts
        papers_content = []
        for paper in papers:
            paper_file = os.path.join(self.literature_dir, f"{paper['id']}.txt")
            if os.path.exists(paper_file):
                with open(paper_file, 'r') as f:
                    content = f.read()
                papers_content.append({
                    'id': paper['id'],
                    'title': paper['title'],
                    'content': content
                })
        
        # Generate synthesis prompt
        papers_text = "\n\n---\n\n".join([
            f"Paper {i+1}: {p['title']}\n{p['content']}"
            for i, p in enumerate(papers_content)
        ])
        
        prompt = f"""Analyze these research papers to answer the query. Extract key findings and synthesize insights.

Query: {query}

Context from Previous Analyses:
{context[:500] if context else 'None'}

Papers:
{papers_text[:8000]}  # Truncated to fit context

Provide:
1. Key findings from the papers relevant to the query
2. Statistical evidence (correlations, p-values, effect sizes)
3. How these findings relate to the query
4. Any conflicting results

Format as a structured summary."""

        try:
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a research analyst synthesizing scientific literature."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE
            )
            
            synthesis = response.choices[0].message.content
            
            return {
                'synthesis': synthesis,
                'papers_reviewed': [p['id'] for p in papers],
                'num_papers': len(papers)
            }
        
        except Exception as e:
            print(f"   ✗ Error extracting insights: {e}")
            return {
                'synthesis': 'Error extracting insights',
                'error': str(e)
            }
