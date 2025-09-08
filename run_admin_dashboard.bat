@echo off
REM === GeddesGhost Admin Dashboard Runner ===

REM Step 1: Check for venv, create if missing
IF NOT EXIST .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Step 2: Activate venv and install requirements
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install plotly --upgrade

REM Step 3: Run Streamlit admin dashboard and force browser open
set STREAMLIT_BROWSER_GOTO_NEW_TAB=true
start "" http://localhost:8502
streamlit run admin_dashboard.py --server.headless true --browser.serverAddress localhost --server.port 8502 --server.runOnSave true 