#!/bin/bash
# Quick Setup Script for Autonomous Discovery System
# Run this from your agentic-discovery directory

echo "========================================"
echo "AUTONOMOUS DISCOVERY SYSTEM - SETUP"
echo "========================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

python3 --version
echo ""

echo "Step 1: Running integration test..."
echo "----------------------------------------"
python3 test_integration.py
if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Some tests failed. Check output above."
    echo ""
fi

echo ""
echo "Step 2: Checking dataset..."
echo "----------------------------------------"
python3 check_dataset_schema_FIXED.py
if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Dataset not found or has issues"
    echo "Make sure you have data/customers.csv"
    echo ""
fi

echo ""
echo "========================================"
echo "SETUP COMPLETE!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Review any warnings above"
echo ""
echo "2. Start the Streamlit app:"
echo "   streamlit run streamlit_integrated.py"
echo ""
echo "3. Or test individual components:"
echo "   python3 schema_aware_questions.py data/customers.csv"
echo "   python3 safe_analysis_agent.py data/customers.csv"
echo ""
echo "4. See INTEGRATION_GUIDE.md for full details"
echo ""
