# AI-Powered Multi-Tenant B2B Lending Engine - ML Risk Engine

This is the Machine Learning microservice for the B2B Lending Engine. Built with Python and FastAPI, it evaluates business loan applications in real-time, engineering financial features and utilizing a Random Forest Classifier to predict the probability of default.

## 🚀 Features
- **Feature Engineering**: Calculates critical loan metrics on the fly, including:
  - DSCR (Debt Service Coverage Ratio)
  - Debt-to-Revenue Ratio
  - Loan-to-Revenue Ratio
- **Predictive Modeling**: Utilizes a pre-trained `scikit-learn` Random Forest Classifier (`n_estimators=100`) to assess risk.
- **Fail-Safe Heuristics**: If the `.pkl` model is unavailable, the engine gracefully falls back to a hardcoded DSCR mathematical penalty system to ensure loan processing is never interrupted.
- **Explainable AI**: Generates human-readable "reasons" for its score to help Loan Officers understand the AI's decision.
- **Microservice Architecture**: Fully decoupled via a fast REST API, allowing data scientists to train and deploy new `.pkl` models without touching the Java backend.

## 🛠 Tech Stack
- **API Framework**: FastAPI & Uvicorn
- **Machine Learning**: scikit-learn (RandomForest)
- **Data Manipulation**: Pandas, NumPy
- **Validation**: Pydantic

## 💻 Local Development

1. **Setup Virtual Environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **(Optional) Train a New Model:**
You can generate synthetic data and train a new model by running:
```bash
python ml/data_generator.py
python ml/train_model.py
```
This will output a new `risk_model.pkl` in the `models/` directory.

4. **Run the API Server:**
```bash
uvicorn main:app --reload --port 8000
```
The server will start on `http://localhost:8000`. You can view the interactive Swagger documentation at `http://localhost:8000/docs`.

## 📦 Deployment
Deploy this service easily to platforms like Render or Hugging Face Spaces by pointing them to this repository and using `uvicorn main:app --host 0.0.0.0 --port $PORT` as the start command.
