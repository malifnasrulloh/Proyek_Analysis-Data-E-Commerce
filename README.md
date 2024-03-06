# Proyek Analisis Data

## Setup environment (Linux)
_pakai virtual environment untuk menghindari package bentrok_

### Linux (Turunan Debian)
```
cd Proyek_Analysis-Data-E-Commerce
python3 -m pip install venv
python3 -m venv virtvenv
source virtenv/bin/activate
python3 -m pip install -r requirements.txt
```

### Windows (Powershell)
```
cd Proyek_Analysis-Data-E-Commerce
python -m pip install venv
python -m venv virtvenv
.\virtvenv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run steamlit app

### Linux
```
python3 -m streamlit run main.py
```
### Windows (Powershell)
```
python -m streamlit run main.py
```