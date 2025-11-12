"""
Data Analysis Agent - Generates and executes Python code for data analysis
"""
import os
import json
import traceback
from typing import Dict, Any, Optional
import openai
from config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_CODE_RETRIES

openai.api_key = OPENAI_API_KEY

class DataAnalysisAgent:
    """
    Agent that generates Python code to analyze data and extract insights
    """
    
    def __init__(self, data_dir='data', output_dir='outputs/analyses'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load analysis approaches reference
        self.load_approaches()
    
    def load_approaches(self):
        """Load reference code patterns"""
        try:
            with open('prompts/analysis_approaches.py', 'r') as f:
                self.approaches_reference = f.read()
        except:
            self.approaches_reference = "# No approaches file found"
    
    def analyze(self, research_question: str, context: str = "") -> Dict[str, Any]:
        """
        Generate and execute analysis code for a research question
        
        Args:
            research_question: The specific question to investigate
            context: Additional context from world model
            
        Returns:
            Dictionary with analysis results and code
        """
        print(f"\n🔬 Analyzing: {research_question[:80]}...")
        
        # Generate analysis code
        code = self._generate_code(research_question, context)
        
        if not code:
            return {'success': False, 'error': 'Failed to generate code'}
        
        # Execute code with retries
        for attempt in range(MAX_CODE_RETRIES):
            result = self._execute_code(code)
            
            if result['success']:
                # Save successful analysis
                analysis_id = self._save_analysis(research_question, code, result)
                result['analysis_id'] = analysis_id
                return result
            elif attempt < MAX_CODE_RETRIES - 1:
                # Try to fix error
                print(f"   ⚠ Attempt {attempt + 1} failed, fixing...")
                code = self._fix_code(code, result['error'])
            else:
                return result
        
        return {'success': False, 'error': 'Max retries exceeded'}
    
    def _generate_code(self, research_question: str, context: str) -> Optional[str]:
        """Generate Python code using LLM"""
        
        prompt = f"""You are a data analyst generating Python code to analyze coffee shop customer data.

AVAILABLE DATA:
- data/customers.csv: Customer demographics and attributes
- data/transactions.csv: Transaction history with dates, amounts, satisfaction scores
- data/competitor_data.csv: Customer competitor interactions

RESEARCH QUESTION:
{research_question}

CONTEXT FROM PREVIOUS ANALYSES:
{context}

REFERENCE CODE PATTERNS:
{self.approaches_reference[:3000]}  # Truncated

Generate Python code to analyze this question. The code should:
1. Load necessary data from CSV files
2. Perform appropriate statistical analysis
3. Print key findings with statistics (means, p-values, correlations, etc.)
4. Save any plots to outputs/analyses/
5. Return a summary dictionary with findings

IMPORTANT:
- Use proper statistical tests and report p-values
- Handle missing data appropriately
- Include confidence intervals where applicable
- Code should be executable and self-contained
- Do not use display() or show() - save plots to files instead

Return ONLY the Python code, no explanations."""

        try:
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert data scientist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE
            )
            
            code = response.choices[0].message.content
            
            # Extract code from markdown if present
            if '```python' in code:
                code = code.split('```python')[1].split('```')[0].strip()
            elif '```' in code:
                code = code.split('```')[1].split('```')[0].strip()
            
            return code
        
        except Exception as e:
            print(f"   ✗ Error generating code: {e}")
            return None
    
    def _execute_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code safely and capture output"""
        
        # Setup execution environment
        exec_globals = {
            '__builtins__': __builtins__,
            'os': os,
            'json': json
        }
        exec_locals = {}
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        
        try:
            with redirect_stdout(output_buffer):
                exec(code, exec_globals, exec_locals)
            
            # Get printed output
            output_text = output_buffer.getvalue()
            
            # Look for summary in locals
            summary = exec_locals.get('summary', {})
            if not summary and 'result' in exec_locals:
                summary = exec_locals['result']
            
            return {
                'success': True,
                'output': output_text,
                'summary': summary,
                'code': code
            }
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return {
                'success': False,
                'error': error_msg,
                'code': code
            }
    
    def _fix_code(self, code: str, error: str) -> str:
        """Attempt to fix code based on error"""
        
        prompt = f"""This Python code has an error. Fix it.

CODE:
{code}

ERROR:
{error}

Return ONLY the fixed Python code, no explanations."""

        try:
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You fix Python code errors."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE
            )
            
            fixed_code = response.choices[0].message.content
            
            # Extract code
            if '```python' in fixed_code:
                fixed_code = fixed_code.split('```python')[1].split('```')[0].strip()
            elif '```' in fixed_code:
                fixed_code = fixed_code.split('```')[1].split('```')[0].strip()
            
            return fixed_code
        
        except Exception as e:
            print(f"   ✗ Error fixing code: {e}")
            return code  # Return original if fix fails
    
    def _save_analysis(self, question: str, code: str, result: Dict) -> str:
        """Save analysis to file"""
        
        import hashlib
        analysis_id = hashlib.md5(question.encode()).hexdigest()[:8]
        
        analysis = {
            'analysis_id': analysis_id,
            'question': question,
            'code': code,
            'output': result.get('output', ''),
            'summary': result.get('summary', {})
        }
        
        filepath = os.path.join(self.output_dir, f'analysis_{analysis_id}.json')
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Also save code separately
        code_filepath = os.path.join(self.output_dir, f'analysis_{analysis_id}.py')
        with open(code_filepath, 'w') as f:
            f.write(f"# Research Question: {question}\n\n")
            f.write(code)
        
        print(f"   ✓ Analysis saved: {analysis_id}")
        return analysis_id
