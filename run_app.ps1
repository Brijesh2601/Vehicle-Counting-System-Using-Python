
# Unset PYTHONPATH to avoid conflicts with incompatible libs directory
$env:PYTHONPATH = ""

# Run the Streamlit app using the virtual environment if available
if (Test-Path ".\venv\Scripts\python.exe") {
    & ".\venv\Scripts\python.exe" -m streamlit run app.py
} else {
    python -m streamlit run app.py
}
