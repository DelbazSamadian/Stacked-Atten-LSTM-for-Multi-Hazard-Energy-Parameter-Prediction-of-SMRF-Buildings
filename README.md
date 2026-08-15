# Stacked-Atten-LSTM for Multi-Hazard Energy-Parameter Prediction of SMRF Buildings

An open-access Streamlit application for predicting **Total Energy** in steel
special moment-resisting frame (SMRF) buildings subjected to sequential
earthquake–flood hazards using a hybrid static-feature + Attention-LSTM
surrogate model.

## Features

- Predicts **Total Energy** for earthquake–flood scenarios.
- Accepts two orthogonal ground-motion acceleration records.
- Supports acceleration-only files and time–acceleration files.
- Supports acceleration units in `g` or `m/s²`.
- Provides three building-input modes:
  - **Reference building**
  - **Quick custom building**
  - **Full custom building (CSV)**
- Applies the preprocessing and scalers saved from the trained model.
- Displays temporal attention weights for interpretation.
- Allows users to download the prediction and model inputs as CSV.
- Open-source and designed for research and engineering use.

## Live Demo

👉 **Launch the Streamlit App:**  
Replace the line below with your public Streamlit URL:

`https://divwxtbmqh4e4olsye8fwc.streamlit.app/`

## How to Use

### 1️⃣ Define the Hazard Scenario

- Select the earthquake return period / intensity level.
- Select the flood height.

### 2️⃣ Upload Ground-Motion Records

- Select the ground-motion file format.
- Enter the time step `Δt` for acceleration-only records.
- Upload the X-direction acceleration record.
- Upload the Y-direction acceleration record.
- Select whether the acceleration is in `g` or `m/s²`.

### 3️⃣ Specify the Building

Choose one of the following:

**Reference building**  
Select a fully specified building from the saved model database.

**Quick custom building**  
Enter the principal global structural parameters. The remaining member-level
inputs are assigned their training-set mean values.

**Full custom building (CSV)**  
Download the structural-input template, enter all required structural
parameters, and upload the completed CSV file.

### 4️⃣ Run Prediction

Click **Predict Total Energy**.

The application displays the predicted Total Energy in `kN·m` and reports
warnings if the supplied inputs are outside the central training domain.

### 5️⃣ View and Download Results

- Review the predicted Total Energy.
- Optionally inspect the temporal attention weights.
- Download the prediction and input parameters as a CSV file.

## Getting Started Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/DelbazSamadian/Stacked-Atten-LSTM-for-Multi-Hazard-Energy-Parameter-Prediction-of-SMRF-Buildings.git
cd Stacked-Atten-LSTM-for-Multi-Hazard-Energy-Parameter-Prediction-of-SMRF-Buildings
```

### 2️⃣ Create a Virtual Environment (Recommended)

**Windows**

```bash
python -m venv venv
venv\\Scripts\\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
python -m streamlit run GUI.py
```

## Repository Files

```text
GUI.py
README.md
best_model.pth
preprocessing.joblib
grouped_data_splits.xlsx
requirements.txt
```

`best_model.pth` and `preprocessing.joblib` must come from the same final
training run.

## Developers

- **Hadi Eslamnia**
- **Delbaz Samadian**
- **Imrose B. Muhit**

**Teesside University**

## Research-Use Note

This application is a research surrogate model and is not a replacement for
full nonlinear structural analysis. Predictions outside the model training
domain should be treated as extrapolations.
