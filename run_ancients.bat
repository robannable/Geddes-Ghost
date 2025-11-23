@echo off
REM === With The Ancients Local Runner ===

REM Step 1: Check for venv, create if missing
IF NOT EXIST .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Step 2: Activate venv and install requirements
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Step 3: Run Streamlit app and force browser open
set STREAMLIT_BROWSER_GOTO_NEW_TAB=true
start "" http://localhost:8501
streamlit run ancients.py --server.headless true --browser.serverAddress localhost --server.runOnSave true
