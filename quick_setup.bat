@echo off
REM Quick Setup Script for Autonomous Discovery System
REM Run this from your agentic-discovery directory

echo ========================================
echo AUTONOMOUS DISCOVERY SYSTEM - SETUP
echo ========================================
echo.

REM Check Python version
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo.
echo Step 1: Running integration test...
echo ----------------------------------------
python test_integration.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Some tests failed. Check output above.
    echo.
)

echo.
echo Step 2: Checking dataset...
echo ----------------------------------------
python check_dataset_schema_FIXED.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Dataset not found or has issues
    echo Make sure you have data/customers.csv
    echo.
)

echo.
echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Review any warnings above
echo.
echo 2. Start the Streamlit app:
echo    streamlit run streamlit_integrated.py
echo.
echo 3. Or test individual components:
echo    python schema_aware_questions.py data/customers.csv
echo    python safe_analysis_agent.py data/customers.csv
echo.
echo 4. See INTEGRATION_GUIDE.md for full details
echo.
pause
