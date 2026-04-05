# kpop-scandal-predictor
K-Pop Scandal Impact Predictor
A machine learning model that predicts whether a breaking K-pop scandal will escalate into a High Crisis (career-altering: group departure, contract termination, criminal charges) or remain Manageable (the idol recovers within weeks to months).
Live Demo
👉 Try it on Streamlit Cloud (update link after deployment)
Model

Algorithm: Random Forest (200 trees, balanced class weights)
Dataset: 103 real K-pop scandals from 2006–2024
Performance: 83.5% accuracy, F1=0.75 on minority class (high crisis), AUC=0.87
Baseline: A model that always guesses "manageable" gets 66% — ours beats that by 17.5 points

Top Predictive Features
FeatureImportancePublic reaction spike (Google Trends)19.0%Fandom size15.4%Response delay (days)12.1%Criminal scandal type9.3%Company response strength9.0%
Limitations

Small dataset (103 cases, 35 "high crisis") — performance estimates have high variance
40% of reaction_spike values were imputed from other features
fandom_size and agency_tier are highly correlated (r=0.62) — the model can't fully disentangle them
Two scandal types (behavior, political) had zero "high" cases in training — the model may be overconfident about these
This is a research prototype, not a crisis management tool

Run Locally
bashpip install -r requirements.txt
python train_model.py          # retrain the model (optional)
streamlit run app.py           # launch the app
Deploy to Streamlit Cloud

Push this repo to GitHub
Go to share.streamlit.io
Connect your GitHub repo
Set main file path to app.py
Deploy — you'll get a public URL

Files
FilePurposeapp.pyStreamlit app (landing page, form, results)train_model.pyModel training + evaluation scriptmodel_base.joblibSerialized trained modelpipeline_config.jsonFeature names, input ranges, metricskpop_scandals_FINALFINAL.csvThe 103-row datasetrequirements.txtPython dependencies.streamlit/config.tomlDark theme config
