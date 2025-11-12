"""
Schema-Aware Question Generator
Automatically generates research questions based on actual dataset columns
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict
import json


class SchemaAwareQuestionGenerator:
    """Generates research questions based on actual dataset schema"""
    
    def __init__(self, data_path: str):
        """Initialize with dataset path"""
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self.columns = list(self.df.columns)
        self.dtypes = self.df.dtypes.to_dict()
        
        # Categorize columns
        self.numeric_cols = [col for col, dtype in self.dtypes.items() 
                            if pd.api.types.is_numeric_dtype(dtype)]
        self.categorical_cols = [col for col, dtype in self.dtypes.items() 
                                if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype)]
        self.datetime_cols = [col for col, dtype in self.dtypes.items() 
                             if pd.api.types.is_datetime64_any_dtype(dtype)]
        
        print(f"📊 Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"   Numeric columns: {len(self.numeric_cols)}")
        print(f"   Categorical columns: {len(self.categorical_cols)}")
        print(f"   Datetime columns: {len(self.datetime_cols)}")
    
    def generate_questions(self, num_questions: int = 10) -> List[str]:
        """Generate research questions based on actual schema"""
        questions = []
        
        # Time series questions if datetime columns exist
        if self.datetime_cols and self.numeric_cols:
            for date_col in self.datetime_cols[:1]:  # Just first datetime
                for num_col in self.numeric_cols[:2]:  # First 2 numeric
                    questions.append(
                        f"What are the temporal trends in {num_col} over time (using {date_col})?"
                    )
        
        # Numeric relationships
        if len(self.numeric_cols) >= 2:
            for i, col1 in enumerate(self.numeric_cols[:3]):
                for col2 in self.numeric_cols[i+1:4]:
                    questions.append(
                        f"What is the relationship between {col1} and {col2}?"
                    )
        
        # Categorical analysis
        if self.categorical_cols and self.numeric_cols:
            for cat_col in self.categorical_cols[:2]:
                for num_col in self.numeric_cols[:2]:
                    questions.append(
                        f"How does {cat_col} affect {num_col}?"
                    )
        
        # Group comparisons
        if self.categorical_cols and self.numeric_cols:
            for cat_col in self.categorical_cols[:1]:
                for num_col in self.numeric_cols[:1]:
                    questions.append(
                        f"Do different {cat_col} categories show different {num_col} patterns?"
                    )
        
        # Statistical summaries
        if self.numeric_cols:
            questions.append(
                f"What are the key statistical characteristics of {self.numeric_cols[0]}?"
            )
        
        # Return requested number of questions
        return questions[:num_questions]
    
    def get_schema_info(self) -> Dict:
        """Return schema information for world model"""
        return {
            "total_columns": len(self.columns),
            "numeric_columns": self.numeric_cols,
            "categorical_columns": self.categorical_cols,
            "datetime_columns": self.datetime_cols,
            "sample_values": {
                col: self.df[col].dropna().head(3).tolist() 
                for col in self.columns[:5]  # First 5 columns
            },
            "dataset_shape": self.df.shape,
            "column_descriptions": {
                col: {
                    "dtype": str(dtype),
                    "null_count": int(self.df[col].isna().sum()),
                    "unique_values": int(self.df[col].nunique()) if self.df[col].nunique() < 100 else "100+"
                }
                for col, dtype in self.dtypes.items()
            }
        }


def update_world_model_with_schema(data_path: str, output_path: str = "world_model_schema.json"):
    """Create world model context from dataset schema"""
    generator = SchemaAwareQuestionGenerator(data_path)
    
    # Generate initial research questions
    questions = generator.generate_questions(num_questions=10)
    
    # Get schema info
    schema_info = generator.get_schema_info()
    
    # Create world model context
    world_model = {
        "dataset_path": str(data_path),
        "schema": schema_info,
        "initial_research_questions": questions,
        "discoveries": [],
        "analyzed_questions": []
    }
    
    # Save to file
    output = Path(output_path)
    with open(output, 'w') as f:
        json.dump(world_model, f, indent=2)
    
    print(f"\n✅ World model saved to: {output}")
    print(f"\n📋 Generated {len(questions)} research questions:")
    for i, q in enumerate(questions, 1):
        print(f"   {i}. {q}")
    
    return world_model


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python schema_aware_questions.py <path_to_dataset.csv>")
        print("\nExample: python schema_aware_questions.py data/customer_data.csv")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    if not Path(data_path).exists():
        print(f"❌ Error: Dataset not found at {data_path}")
        sys.exit(1)
    
    world_model = update_world_model_with_schema(data_path)
