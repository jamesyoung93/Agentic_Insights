"""
Safe Data Analysis Agent with Column Validation
Prevents KeyError by validating columns exist before analysis
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import traceback


class SafeDataAnalysisAgent:
    """Data analysis agent with built-in column validation"""
    
    def __init__(self, data_path: str):
        """Initialize with dataset"""
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self.available_columns = set(self.df.columns)
        
    def validate_columns(self, required_columns: List[str]) -> tuple[bool, List[str]]:
        """
        Check if required columns exist in dataset
        
        Returns:
            (all_exist: bool, missing_columns: List[str])
        """
        required_set = set(required_columns)
        missing = required_set - self.available_columns
        return len(missing) == 0, list(missing)
    
    def extract_columns_from_question(self, question: str) -> List[str]:
        """
        Extract potential column names from research question
        Uses heuristics to identify column references
        """
        # Simple heuristic: look for words that match column names
        words = question.lower().replace('?', '').replace(',', '').split()
        
        # Check each column name (case-insensitive)
        mentioned_columns = []
        for col in self.available_columns:
            col_lower = col.lower()
            col_words = col_lower.replace('_', ' ').split()
            
            # Check if column name or its components appear in question
            if col_lower in ' '.join(words):
                mentioned_columns.append(col)
            elif any(word in words for word in col_words if len(word) > 3):
                mentioned_columns.append(col)
        
        return mentioned_columns
    
    def suggest_alternative_question(self, original_question: str, missing_columns: List[str]) -> str:
        """
        Suggest alternative question using available columns
        """
        # Find similar available columns
        alternatives = {}
        for missing in missing_columns:
            # Look for columns with similar names
            similar = [col for col in self.available_columns 
                      if any(word in col.lower() for word in missing.lower().split('_'))]
            if similar:
                alternatives[missing] = similar[0]
        
        # Replace missing columns in question
        suggested_question = original_question
        for missing, alternative in alternatives.items():
            suggested_question = suggested_question.replace(missing, alternative)
        
        return suggested_question
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze research question with column validation
        
        Returns analysis result or error with suggestions
        """
        result = {
            "question": question,
            "status": "pending",
            "columns_mentioned": [],
            "missing_columns": [],
            "analysis": None,
            "error": None,
            "suggestion": None
        }
        
        # Extract columns from question
        mentioned_columns = self.extract_columns_from_question(question)
        result["columns_mentioned"] = mentioned_columns
        
        if not mentioned_columns:
            result["status"] = "warning"
            result["error"] = "Could not identify specific columns in question"
            result["suggestion"] = f"Available columns: {', '.join(sorted(self.available_columns))}"
            return result
        
        # Validate columns exist
        all_exist, missing = self.validate_columns(mentioned_columns)
        
        if not all_exist:
            result["status"] = "failed"
            result["missing_columns"] = missing
            result["error"] = f"Columns not found in dataset: {', '.join(missing)}"
            result["suggestion"] = self.suggest_alternative_question(question, missing)
            result["available_columns"] = sorted(self.available_columns)
            return result
        
        # If validation passes, proceed with analysis
        try:
            # Here you would call your actual analysis logic
            # For now, return success with basic stats
            result["status"] = "success"
            result["analysis"] = {
                "columns_analyzed": mentioned_columns,
                "basic_stats": {
                    col: {
                        "count": int(self.df[col].count()),
                        "null_count": int(self.df[col].isna().sum()),
                        "unique_values": int(self.df[col].nunique())
                    }
                    for col in mentioned_columns
                }
            }
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
        
        return result
    
    def get_available_columns_by_type(self) -> Dict[str, List[str]]:
        """Return columns grouped by data type"""
        return {
            "numeric": [col for col in self.df.columns 
                       if pd.api.types.is_numeric_dtype(self.df[col])],
            "categorical": [col for col in self.df.columns 
                          if pd.api.types.is_object_dtype(self.df[col]) or 
                          pd.api.types.is_categorical_dtype(self.df[col])],
            "datetime": [col for col in self.df.columns 
                        if pd.api.types.is_datetime64_any_dtype(self.df[col])],
            "boolean": [col for col in self.df.columns 
                       if pd.api.types.is_bool_dtype(self.df[col])]
        }


def test_safe_analysis():
    """Test the safe analysis agent"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python safe_analysis_agent.py <path_to_dataset.csv>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    if not Path(data_path).exists():
        print(f"❌ Error: Dataset not found at {data_path}")
        sys.exit(1)
    
    # Initialize agent
    agent = SafeDataAnalysisAgent(data_path)
    
    print(f"📊 Dataset loaded: {agent.df.shape}")
    print(f"\n📋 Available columns by type:")
    for dtype, cols in agent.get_available_columns_by_type().items():
        if cols:
            print(f"   {dtype}: {', '.join(cols)}")
    
    # Test with some questions
    test_questions = [
        "What is the relationship between satisfaction and loyalty?",  # May fail
        "How does income influence spending?",  # May fail
        f"What are the trends in {agent.available_columns.pop()}?",  # Should work
    ]
    
    print(f"\n🧪 Testing analysis with validation:\n")
    for question in test_questions:
        print(f"Q: {question}")
        result = agent.analyze_question(question)
        print(f"   Status: {result['status']}")
        if result['missing_columns']:
            print(f"   ❌ Missing: {', '.join(result['missing_columns'])}")
            print(f"   💡 Suggestion: {result['suggestion']}")
        elif result['status'] == 'success':
            print(f"   ✅ Analyzed: {', '.join(result['columns_mentioned'])}")
        print()


if __name__ == "__main__":
    test_safe_analysis()
