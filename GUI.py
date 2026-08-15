# -*- coding: utf-8 -*-
"""
Open-access Streamlit GUI for the trained Hybrid Attention-LSTM Total_Energy model.

Recommended public-repository layout
------------------------------------
repo/
├── GUI.py
├── requirements.txt
├── model_artifacts/
│   ├── best_model.pth
│   ├── preprocessing.joblib
│   └── grouped_data_splits.xlsx      # optional but recommended
└── README.md

The app does NOT need data_dict.npz. Users upload their own two-component
ground-motion records.

Important:
- best_model.pth and preprocessing.joblib MUST come from the same training run.
- The inference pipeline reproduces the final training preprocessing.
"""

from __future__ import annotations

import io
import os
import pathlib
from pathlib import Path
from typing import Dict, Mapping, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


# =============================================================================
# APP CONFIGURATION
# =============================================================================

APP_DIR = Path(__file__).resolve().parent

APP_TITLE = "Total Energy Prediction of SMRF Buildings"
APP_SUBTITLE = "Sequential Earthquake–Flood Hazard Surrogate Model"

# Model artifacts may be stored either in the repository root or in
# a model_artifacts/ subfolder. The app checks both locations.
MODEL_DIR = APP_DIR / "model_artifacts"

IRP_OPTIONS = {
    "25-year return period": 1.0,
    "100-year return period": 2.0,
    "500-year return period": 3.0,
    "1500-year return period": 4.0,
    "2500-year return period": 5.0,
}

FH_OPTIONS_M = [0.30, 1.00, 2.00, 4.00]
GRAVITY_MS2 = 9.80665

# Quick-custom mode exposes a compact group of structural parameters.
# All other parameters are set to their training-set means and clearly reported.
QUICK_STRUCTURAL_FEATURES = [
    "T1",
    "Ω",
    "µ",
    "Vy/W",
    "Mass",
    "M1",
    "H_storey",
    "L_bay",
    "Fy",
    "Es",
]


# =============================================================================
# PAGE + STYLE
# =============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"], .stApp, .block-container, label, p, div, input,
    button, textarea {
        font-family: "Times New Roman", Times, serif !important;
    }

    .stApp { background-color: #f4f5f7; }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: white;
        border: 1px solid #d8dce3;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,.04);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.15rem;
        line-height: 1.15;
    }

    .hero p {
        margin: 6px 0 0 0;
        font-size: 1.05rem;
    }

    .stepbox {
        background: white;
        border: 1px solid #d8dce3;
        border-radius: 10px;
        padding: 12px 15px;
        margin: 8px 0 14px 0;
    }

    .note {
        background: #fff;
        border-left: 5px solid #657080;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 8px 0 12px 0;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #d3d7de;
        border-radius: 10px;
        padding: 13px;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border-radius: 9px;
        border: 1px solid #d1d5dc;
        padding: 7px;
    }

    .stButton > button {
        border-radius: 8px;
        font-size: 1.08rem;
        min-height: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <h1>{APP_TITLE}</h1>
      <p><strong>{APP_SUBTITLE}</strong></p>
      <p>
        Upload two orthogonal ground-motion records, define the earthquake and
        flood scenario, specify the building, and estimate Total Energy.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# MODEL DEFINITION — MUST MATCH FINAL TRAINING CODE
# =============================================================================

class TemporalAttention(nn.Module):
    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_size, feature_size)
        self.score = nn.Linear(feature_size, 1, bias=False)

    def forward(self, sequence_outputs: torch.Tensor):
        energy = self.score(
            torch.tanh(self.projection(sequence_outputs))
        ).squeeze(-1)
        weights = torch.softmax(energy, dim=1)
        context = torch.sum(
            sequence_outputs * weights.unsqueeze(-1),
            dim=1,
        )
        return context, weights


class HybridAttentionLSTM(nn.Module):
    def __init__(
        self,
        static_input_size: int,
        number_of_channels: int,
        hidden_size: int,
        num_layers: int,
        lstm_dropout: float,
        bidirectional: bool,
        static_hidden_size: int,
        fusion_hidden_size: int,
        head_dropout: float,
    ) -> None:
        super().__init__()

        effective_dropout = lstm_dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=number_of_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=bidirectional,
        )

        temporal_size = hidden_size * (2 if bidirectional else 1)
        self.attention = TemporalAttention(temporal_size)

        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_size, static_hidden_size),
            nn.LayerNorm(static_hidden_size),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(static_hidden_size, static_hidden_size // 2),
            nn.GELU(),
        )

        fused_size = temporal_size + static_hidden_size // 2

        self.regression_head = nn.Sequential(
            nn.Linear(fused_size, fusion_hidden_size),
            nn.LayerNorm(fusion_hidden_size),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(head_dropout / 2.0),
            nn.Linear(fusion_hidden_size // 2, 1),
        )

    def forward(
        self,
        static_inputs: torch.Tensor,
        time_series_inputs: torch.Tensor,
        return_attention: bool = False,
    ):
        temporal_outputs, _ = self.lstm(time_series_inputs)
        temporal_context, attention_weights = self.attention(temporal_outputs)
        static_context = self.static_encoder(static_inputs)
        fused = torch.cat([temporal_context, static_context], dim=1)
        prediction = self.regression_head(fused).squeeze(-1)

        if return_attention:
            return prediction, attention_weights
        return prediction


def build_model(
    static_input_size: int,
    number_of_channels: int,
    params: Mapping[str, object],
) -> HybridAttentionLSTM:
    return HybridAttentionLSTM(
        static_input_size=static_input_size,
        number_of_channels=number_of_channels,
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        lstm_dropout=float(params["lstm_dropout"]),
        bidirectional=bool(params["bidirectional"]),
        static_hidden_size=int(params["static_hidden_size"]),
        fusion_hidden_size=int(params["fusion_hidden_size"]),
        head_dropout=float(params["head_dropout"]),
    )


# =============================================================================
# ARTIFACT DISCOVERY + LOADING
# =============================================================================

def resolve_artifact(filename: str, required: bool = True) -> Path:
    """
    Find a deployment artifact in either:
      1) <repo>/model_artifacts/<filename>
      2) <repo>/<filename>

    This makes the GitHub deployment tolerant of either repository layout.
    """
    candidates = [
        MODEL_DIR / filename,
        APP_DIR / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if required:
        checked = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            f"Could not find required file: {filename}\n"
            f"Checked:\n{checked}\n\n"
            "Upload the file either to the repository root or to the "
            "model_artifacts/ folder, then reboot/redeploy the Streamlit app."
        )

    # Optional artifact: return preferred path even when absent.
    return candidates[0]


try:
    # Required deployment artifacts.
    MODEL_PATH = resolve_artifact("best_model.pth", required=True)
    PREPROCESSING_PATH = resolve_artifact("preprocessing.joblib", required=True)

    # Optional artifact used for reference-building mode.
    SPLIT_PATH = resolve_artifact("grouped_data_splits.xlsx", required=False)
except Exception as artifact_error:
    st.error("Required model files are missing from the deployed repository.")
    st.code(str(artifact_error))
    st.stop()



def load_joblib_cross_platform(path: Path):
    """
    Load a joblib/pickle file created on Windows when the app runs on Linux.

    preprocessing.joblib contains the saved Config dictionary, including
    CFG.base_dir. Because training was performed on Windows, that path may be
    serialized as pathlib.WindowsPath. Linux cannot normally instantiate
    WindowsPath during unpickling, which raises:

        cannot instantiate 'WindowsPath' on your system

    We temporarily map WindowsPath to PosixPath only while unpickling. The
    stored training path is metadata only and is not used to locate deployment
    files in this GUI.
    """
    original_windows_path = pathlib.WindowsPath

    try:
        if os.name != "nt":
            pathlib.WindowsPath = pathlib.PosixPath

        bundle = joblib.load(path)

    finally:
        pathlib.WindowsPath = original_windows_path

    # Sanitize the saved training-machine path because it is irrelevant on the
    # deployed Linux server and should never be used for deployment I/O.
    if isinstance(bundle, dict):
        config = bundle.get("config")
        if isinstance(config, dict) and "base_dir" in config:
            config = dict(config)
            config["base_dir"] = str(config["base_dir"])
            bundle["config"] = config

    return bundle


@st.cache_resource(show_spinner="Loading trained model...")
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_PATH}\n"
            "Place best_model.pth in model_artifacts/."
        )

    if not PREPROCESSING_PATH.exists():
        raise FileNotFoundError(
            f"Missing preprocessing file: {PREPROCESSING_PATH}\n"
            "Place preprocessing.joblib in model_artifacts/."
        )

    bundle = load_joblib_cross_platform(PREPROCESSING_PATH)

    required_bundle = {
        "input_columns",
        "static_scaler",
        "target_scaler",
        "time_series_channel_mean",
        "time_series_channel_std",
    }
    missing = required_bundle.difference(bundle)
    if missing:
        raise KeyError(
            "preprocessing.joblib is missing: "
            + ", ".join(sorted(missing))
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

    required_checkpoint = {
        "model_state_dict",
        "model_params",
        "static_input_size",
    }
    missing = required_checkpoint.difference(checkpoint)
    if missing:
        raise KeyError(
            "best_model.pth is missing: "
            + ", ".join(sorted(missing))
        )

    number_of_channels = int(checkpoint.get("number_of_channels", 2))
    time_steps = int(
        checkpoint.get(
            "time_steps",
            bundle.get("config", {}).get("time_steps", 512),
        )
    )

    model = build_model(
        static_input_size=int(checkpoint["static_input_size"]),
        number_of_channels=number_of_channels,
        params=checkpoint["model_params"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_columns = list(bundle["input_columns"])
    summary_names = list(bundle.get("summary_feature_names", []))
    summary_scaler = bundle.get("summary_scaler")

    expected_size = len(input_columns)
    if summary_scaler is not None:
        expected_size += len(summary_names)

    if expected_size != int(checkpoint["static_input_size"]):
        raise ValueError(
            "Model/preprocessing mismatch. Use best_model.pth and "
            "preprocessing.joblib from the same training run."
        )

    return {
        "model": model,
        "device": device,
        "checkpoint": checkpoint,
        "bundle": bundle,
        "input_columns": input_columns,
        "summary_names": summary_names,
        "summary_scaler": summary_scaler,
        "time_steps": time_steps,
        "number_of_channels": number_of_channels,
    }


try:
    ART = load_artifacts()
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.code(str(exc))
    st.stop()

MODEL = ART["model"]
DEVICE = ART["device"]
BUNDLE = ART["bundle"]
INPUT_COLUMNS = ART["input_columns"]
SUMMARY_NAMES = ART["summary_names"]
SUMMARY_SCALER = ART["summary_scaler"]
TIME_STEPS = ART["time_steps"]
NUMBER_OF_CHANNELS = ART["number_of_channels"]

STATIC_SCALER = BUNDLE["static_scaler"]
TARGET_SCALER = BUNDLE["target_scaler"]

CHANNEL_MEAN = np.asarray(
    BUNDLE["time_series_channel_mean"], dtype=np.float64
)
CHANNEL_STD = np.asarray(
    BUNDLE["time_series_channel_std"], dtype=np.float64
)
CHANNEL_STD = np.where(np.abs(CHANNEL_STD) < 1e-12, 1.0, CHANNEL_STD)

if NUMBER_OF_CHANNELS != 2:
    st.error(
        f"This public GUI expects two ground-motion components, but the "
        f"checkpoint expects {NUMBER_OF_CHANNELS} channels."
    )
    st.stop()


# =============================================================================
# TRAINING REFERENCE DATA
# =============================================================================

STATIC_DEFAULTS = {
    name: float(STATIC_SCALER.mean_[i])
    for i, name in enumerate(INPUT_COLUMNS)
}


@st.cache_data(show_spinner=False)
def load_reference_buildings(path_string: str, columns: Tuple[str, ...]):
    if not path_string:
        return pd.DataFrame()

    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()

    try:
        train = pd.read_excel(path, sheet_name="Train", engine="openpyxl")
        structural_columns = [
            c for c in columns
            if c not in {"iRP", "FH"}
            and c in train.columns
        ]

        if "iModel" not in train.columns:
            return pd.DataFrame()

        reference = (
            train[["iModel"] + structural_columns]
            .groupby("iModel", as_index=False)
            .first()
            .sort_values("iModel")
            .reset_index(drop=True)
        )
        return reference
    except Exception:
        return pd.DataFrame()


REFERENCE_BUILDINGS = load_reference_buildings(
    str(SPLIT_PATH) if SPLIT_PATH.exists() else "",
    tuple(INPUT_COLUMNS),
)


def feature_unit(name: str) -> str:
    if name == "T1":
        return "sec"
    if name == "FH":
        return "m"
    if name in {"H_storey", "L_bay"}:
        return "in"
    if name in {"Es", "Fy"}:
        return "ksi"
    if name.startswith(("Acol", "Abeam")):
        return "in²"
    if name.startswith(("Icol", "Ibeam")):
        return "in⁴"
    if name.startswith(("Zcol", "Zbeam")):
        return "in³"
    if name.startswith(("ϴp", "ϴpc")):
        return "rad"
    return "–"


def feature_description(name: str) -> str:
    descriptions = {
        "Ω": "Overstrength factor",
        "µ": "Ductility capacity",
        "H_storey": "Storey height",
        "Vy/W": "Normalised lateral strength",
        "T1": "Fundamental period",
        "Mass": "Structural mass",
        "M1": "First-mode mass participation",
        "M2": "Second-mode mass participation",
        "M3": "Third-mode mass participation",
        "L_bay": "Bay length",
        "Es": "Elastic modulus",
        "Fy": "Steel yield strength",
        "FH": "Flood height",
        "iRP": "Seismic hazard level",
    }
    return descriptions.get(name, name.replace("_", " "))


# =============================================================================
# GROUND-MOTION READING + PREPROCESSING
# =============================================================================

def read_numeric_file(file_bytes: bytes) -> np.ndarray:
    delimiters = [None, ",", "\t", ";"]
    candidates = []

    for delimiter in delimiters:
        try:
            arr = np.genfromtxt(
                io.BytesIO(file_bytes),
                delimiter=delimiter,
                comments="#",
                dtype=float,
                invalid_raise=False,
            )
            arr = np.asarray(arr, dtype=float)
            finite_count = int(np.isfinite(arr).sum())
            if finite_count > 0:
                candidates.append((finite_count, arr))
        except Exception:
            pass

    if not candidates:
        raise ValueError("No numeric data were found in the uploaded file.")

    return max(candidates, key=lambda item: item[0])[1]


def parse_ground_motion(
    uploaded_file,
    dt: float,
    file_format: str,
    acceleration_unit: str,
):
    raw = read_numeric_file(uploaded_file.getvalue())

    if file_format == "Acceleration values only":
        acceleration = raw[np.isfinite(raw)].reshape(-1)
        if acceleration.size < 2:
            raise ValueError("At least two acceleration samples are required.")
        time_values = np.arange(acceleration.size, dtype=float) * dt

    else:
        raw = np.atleast_2d(raw)
        if raw.shape[1] < 2:
            raise ValueError(
                "Two-column format requires time in column 1 and "
                "acceleration in column 2."
            )

        time_values = raw[:, 0]
        acceleration = raw[:, 1]
        mask = np.isfinite(time_values) & np.isfinite(acceleration)
        time_values = time_values[mask]
        acceleration = acceleration[mask]

        order = np.argsort(time_values)
        time_values = time_values[order]
        acceleration = acceleration[order]

        time_values, indices = np.unique(
            time_values, return_index=True
        )
        acceleration = acceleration[indices]

        if len(time_values) < 2 or time_values[-1] <= time_values[0]:
            raise ValueError("Time values must increase.")

    if acceleration_unit == "m/s²":
        acceleration = acceleration / GRAVITY_MS2

    return (
        np.asarray(time_values, dtype=np.float64),
        np.asarray(acceleration, dtype=np.float64),
    )


def resample_channel(time_values, signal_values, number_of_steps):
    if signal_values.size == 1 or time_values[-1] <= time_values[0]:
        return np.full(
            number_of_steps, signal_values[0], dtype=np.float32
        )

    new_time = np.linspace(
        time_values[0], time_values[-1], number_of_steps
    )
    return np.interp(
        new_time, time_values, signal_values
    ).astype(np.float32)


def summarise_channel(time_values, signal_values):
    duration = float(max(time_values[-1] - time_values[0], 0.0))
    mean = float(np.mean(signal_values))
    std = float(np.std(signal_values))
    rms = float(np.sqrt(np.mean(np.square(signal_values))))
    max_abs = float(np.max(np.abs(signal_values)))
    peak_to_peak = float(np.ptp(signal_values))
    abs_p95 = float(np.percentile(np.abs(signal_values), 95.0))

    if signal_values.size >= 2 and duration > 0:
        squared_integral = float(
            np.trapz(np.square(signal_values), time_values)
        )
    else:
        squared_integral = 0.0

    return np.asarray(
        [
            duration,
            mean,
            std,
            rms,
            max_abs,
            peak_to_peak,
            abs_p95,
            squared_integral,
        ],
        dtype=np.float64,
    )


def prepare_time_series(t1, a1, t2, a2):
    seq1 = resample_channel(t1, a1, TIME_STEPS)
    seq2 = resample_channel(t2, a2, TIME_STEPS)

    raw_sequence = np.stack([seq1, seq2], axis=1).astype(np.float32)
    scaled_sequence = (raw_sequence - CHANNEL_MEAN) / CHANNEL_STD

    raw_summary = np.concatenate(
        [
            summarise_channel(t1, a1),
            summarise_channel(t2, a2),
        ]
    )

    return scaled_sequence.astype(np.float32), raw_summary


# =============================================================================
# BUILDING INPUT HELPERS
# =============================================================================

def base_static_values() -> Dict[str, float]:
    return dict(STATIC_DEFAULTS)


def reference_building_values(model_id: int) -> Dict[str, float]:
    values = base_static_values()

    if REFERENCE_BUILDINGS.empty:
        return values

    row = REFERENCE_BUILDINGS.loc[
        REFERENCE_BUILDINGS["iModel"] == model_id
    ]

    if row.empty:
        return values

    row = row.iloc[0]
    for column in INPUT_COLUMNS:
        if column in row.index and pd.notna(row[column]):
            values[column] = float(row[column])

    return values


def parse_building_csv(uploaded_file) -> Dict[str, float]:
    table = pd.read_csv(uploaded_file)
    if table.empty:
        raise ValueError("The building CSV is empty.")

    row = table.iloc[0]
    values = base_static_values()

    missing = []
    for column in INPUT_COLUMNS:
        if column in {"iRP", "FH"}:
            continue
        if column not in row.index or pd.isna(row[column]):
            missing.append(column)
        else:
            values[column] = float(row[column])

    if missing:
        raise ValueError(
            "The CSV is missing required structural inputs: "
            + ", ".join(missing)
        )

    return values


def manual_number_input(column: str, key_prefix: str, default: float):
    unit = feature_unit(column)
    label = f"{feature_description(column)}"
    if unit != "–":
        label += f" [{unit}]"

    if abs(default) >= 1000:
        step = 10.0
    elif abs(default) >= 100:
        step = 1.0
    elif abs(default) >= 10:
        step = 0.1
    else:
        step = 0.01

    return float(
        st.number_input(
            label,
            value=float(default),
            step=float(step),
            format="%.6f",
            key=f"{key_prefix}_{column}",
            help=f"Model variable: {column}",
        )
    )


def z_score_warnings(values: Mapping[str, float]):
    raw = np.asarray(
        [[values[c] for c in INPUT_COLUMNS]], dtype=np.float64
    )
    z = STATIC_SCALER.transform(raw)[0]

    return [
        (INPUT_COLUMNS[i], float(value))
        for i, value in enumerate(z)
        if abs(value) > 3.0
    ]


# Downloadable complete structural-input template.
STRUCTURAL_TEMPLATE_COLUMNS = [
    c for c in INPUT_COLUMNS if c not in {"iRP", "FH"}
]
STRUCTURAL_TEMPLATE = pd.DataFrame(
    [
        {
            c: STATIC_DEFAULTS[c]
            for c in STRUCTURAL_TEMPLATE_COLUMNS
        }
    ]
).to_csv(index=False).encode("utf-8-sig")


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("How to use")

    st.markdown(
        """
        **1.** Define the hazard scenario  
        **2.** Upload X and Y ground motions  
        **3.** Select how the building is specified  
        **4.** Click **Predict Total Energy**
        """
    )

    st.divider()

    st.subheader("Building input modes")
    st.markdown(
        """
        **Reference building**  
        Easiest and fully specified from the model database.

        **Quick custom building**  
        Enter a compact set of global parameters; remaining member-level
        parameters use training means.

        **Full custom building**  
        Upload all structural parameters in one CSV file.
        """
    )

    st.divider()
    st.caption(
        "Research surrogate model. Predictions outside the training domain "
        "should be treated as extrapolations."
    )


# =============================================================================
# STEP 1 — HAZARD
# =============================================================================

st.subheader("Step 1 — Define the hazard scenario")

haz1, haz2 = st.columns(2)

with haz1:
    irp_label = st.selectbox(
        "Earthquake intensity",
        list(IRP_OPTIONS.keys()),
        index=2,
    )
    selected_irp = float(IRP_OPTIONS[irp_label])

with haz2:
    selected_fh = float(
        st.selectbox(
            "Flood height",
            FH_OPTIONS_M,
            index=2,
            format_func=lambda x: f"{x:.2f} m",
        )
    )


# =============================================================================
# STEP 2 — GROUND MOTIONS
# =============================================================================

st.subheader("Step 2 — Upload two ground-motion components")

fmt1, fmt2, fmt3 = st.columns([1.5, 1.0, 1.0])

with fmt1:
    gm_format = st.radio(
        "File format",
        ["Acceleration values only", "Two columns: time, acceleration"],
        horizontal=True,
    )

with fmt2:
    dt = float(
        st.number_input(
            "Time step Δt [sec]",
            min_value=0.000001,
            value=0.005,
            step=0.001,
            format="%.6f",
            disabled=(gm_format == "Two columns: time, acceleration"),
        )
    )

with fmt3:
    acceleration_unit = st.selectbox(
        "Acceleration unit",
        ["g", "m/s²"],
        index=0,
    )

gm1_col, gm2_col = st.columns(2)

with gm1_col:
    file1 = st.file_uploader(
        "Component 1 — X direction",
        type=["txt", "csv", "dat"],
        key="gm_x",
    )

with gm2_col:
    file2 = st.file_uploader(
        "Component 2 — Y direction",
        type=["txt", "csv", "dat"],
        key="gm_y",
    )

parsed = {}

for label, uploaded, column in [
    ("X", file1, gm1_col),
    ("Y", file2, gm2_col),
]:
    if uploaded is not None:
        try:
            t, a = parse_ground_motion(
                uploaded,
                dt,
                gm_format,
                acceleration_unit,
            )
            parsed[label] = (t, a)

            with column:
                fig, ax = plt.subplots(figsize=(6.5, 2.2))
                ax.plot(t, a, linewidth=0.8)
                ax.set_xlabel("Time (sec)")
                ax.set_ylabel("Acceleration (g)")
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.caption(
                    f"{len(a):,} samples • "
                    f"{t[-1]-t[0]:.2f} sec • "
                    f"PGA = {np.max(np.abs(a)):.3f} g"
                )
        except Exception as exc:
            with column:
                st.error(f"Could not read the {label}-component: {exc}")


# =============================================================================
# STEP 3 — BUILDING
# =============================================================================

st.subheader("Step 3 — Specify the building")

mode = st.radio(
    "Choose the easiest input method for you",
    [
        "Reference building",
        "Quick custom building",
        "Full custom building (CSV)",
    ],
    horizontal=True,
)

building_values = base_static_values()
input_mode_note = ""

if mode == "Reference building":
    if REFERENCE_BUILDINGS.empty:
        st.info(
            "No grouped_data_splits.xlsx file was found. "
            "The app will use the training-mean reference profile. "
            "For a public release, include grouped_data_splits.xlsx in "
            "model_artifacts/ to enable selectable reference buildings."
        )
        input_mode_note = "Training-mean reference profile"
    else:
        model_ids = REFERENCE_BUILDINGS["iModel"].astype(int).tolist()

        select_col, summary_col = st.columns([1.0, 2.0])

        with select_col:
            selected_model = int(
                st.selectbox(
                    "Reference building ID",
                    model_ids,
                    index=0,
                )
            )

        building_values = reference_building_values(selected_model)
        input_mode_note = f"Reference building iModel = {selected_model}"

        with summary_col:
            summary_items = []
            for feature in ["T1", "H_storey", "L_bay", "Mass", "Fy"]:
                if feature in building_values:
                    unit = feature_unit(feature)
                    summary_items.append(
                        f"**{feature}:** {building_values[feature]:.3f} {unit}"
                    )
            st.markdown(" &nbsp; | &nbsp; ".join(summary_items), unsafe_allow_html=True)

elif mode == "Quick custom building":
    st.markdown(
        """
        <div class="note">
        <strong>Quick estimate:</strong> enter the main global structural
        parameters below. Member-level properties not shown here are fixed at
        their training-set mean values. Use the Full custom mode for a fully
        specified engineering case.
        </div>
        """,
        unsafe_allow_html=True,
    )

    quick_cols = st.columns(2)

    for index, feature in enumerate(QUICK_STRUCTURAL_FEATURES):
        if feature not in INPUT_COLUMNS:
            continue

        with quick_cols[index % 2]:
            building_values[feature] = manual_number_input(
                feature,
                "quick",
                building_values[feature],
            )

    input_mode_note = "Quick custom building; unspecified inputs = training means"

else:
    st.markdown(
        """
        Upload one CSV row containing all structural model inputs. Hazard
        variables **iRP** and **FH** are selected separately in Step 1.
        """
    )

    template_col, upload_col = st.columns([1.0, 2.0])

    with template_col:
        st.download_button(
            "Download structural-input template",
            data=STRUCTURAL_TEMPLATE,
            file_name="SMRF_structural_input_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with upload_col:
        building_csv = st.file_uploader(
            "Upload completed structural-input CSV",
            type=["csv"],
            key="building_csv",
        )

    if building_csv is not None:
        try:
            building_values = parse_building_csv(building_csv)
            st.success("Structural input file loaded successfully.")
            input_mode_note = f"Full custom building: {building_csv.name}"

            preview_columns = [
                c for c in ["T1", "Ω", "µ", "Vy/W", "Mass", "H_storey", "L_bay"]
                if c in building_values
            ]
            st.dataframe(
                pd.DataFrame(
                    [{c: building_values[c] for c in preview_columns}]
                ),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(str(exc))
            building_values = None
    else:
        building_values = None


# Always override hazards with Step 1 values.
if building_values is not None:
    if "iRP" in INPUT_COLUMNS:
        building_values["iRP"] = selected_irp
    if "FH" in INPUT_COLUMNS:
        building_values["FH"] = selected_fh


# Optional full input review.
if building_values is not None:
    with st.expander("Review all model inputs", expanded=False):
        review = pd.DataFrame(
            {
                "Variable": INPUT_COLUMNS,
                "Value": [building_values[c] for c in INPUT_COLUMNS],
                "Unit": [feature_unit(c) for c in INPUT_COLUMNS],
            }
        )
        st.dataframe(review, use_container_width=True, hide_index=True)


# =============================================================================
# STEP 4 — PREDICT
# =============================================================================

st.subheader("Step 4 — Predict Total Energy")

ready = (
    "X" in parsed
    and "Y" in parsed
    and building_values is not None
)

if not ready:
    st.info(
        "Complete Steps 1–3. Two valid ground-motion files and a building "
        "definition are required before prediction."
    )

predict = st.button(
    "Predict Total Energy",
    type="primary",
    use_container_width=True,
    disabled=not ready,
)

if predict:
    try:
        # Static inputs.
        raw_static = np.asarray(
            [[building_values[c] for c in INPUT_COLUMNS]],
            dtype=np.float64,
        )
        scaled_static = STATIC_SCALER.transform(raw_static).astype(np.float32)

        # Uploaded time histories.
        t1, a1 = parsed["X"]
        t2, a2 = parsed["Y"]
        scaled_sequence, raw_summary = prepare_time_series(t1, a1, t2, a2)

        # Time-series summary features.
        summary_warnings = []

        if SUMMARY_SCALER is not None:
            scaled_summary = SUMMARY_SCALER.transform(
                raw_summary.reshape(1, -1)
            ).astype(np.float32)

            scaled_static = np.concatenate(
                [scaled_static, scaled_summary],
                axis=1,
            )

            for i, z in enumerate(scaled_summary[0]):
                if abs(z) > 3.0:
                    name = (
                        SUMMARY_NAMES[i]
                        if i < len(SUMMARY_NAMES)
                        else f"summary_{i}"
                    )
                    summary_warnings.append((name, float(z)))

        expected_size = int(ART["checkpoint"]["static_input_size"])

        if scaled_static.shape[1] != expected_size:
            raise ValueError(
                f"Input-size mismatch: GUI produced {scaled_static.shape[1]} "
                f"static-branch values but the model expects {expected_size}."
            )

        x_static = torch.from_numpy(scaled_static).to(DEVICE)
        x_ts = torch.from_numpy(scaled_sequence[None, :, :]).to(DEVICE)

        with torch.no_grad():
            y_scaled, attention = MODEL(
                x_static,
                x_ts,
                return_attention=True,
            )

        y_scaled_value = float(y_scaled.detach().cpu().numpy()[0])

        predicted_energy = float(
            TARGET_SCALER.inverse_transform(
                np.array([[y_scaled_value]], dtype=np.float64)
            )[0, 0]
        )

        st.success("Prediction completed.")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Predicted Total Energy",
            f"{predicted_energy:,.1f} kN·m",
        )
        m2.metric("Flood height", f"{selected_fh:.2f} m")
        m3.metric("Earthquake level", irp_label)

        st.caption(input_mode_note)

        # Domain checks.
        structural_warnings = z_score_warnings(building_values)

        if structural_warnings:
            names = ", ".join(
                f"{name} ({z:+.1f} SD)"
                for name, z in structural_warnings
            )
            st.warning(
                "Some model inputs are outside ±3 training standard deviations: "
                f"{names}. This prediction should be treated as extrapolation."
            )

        if summary_warnings:
            names = ", ".join(
                f"{name} ({z:+.1f} SD)"
                for name, z in summary_warnings
            )
            st.warning(
                "The uploaded record contains ground-motion descriptors outside "
                f"the central training range: {names}."
            )

        # Optional interpretation details.
        with st.expander("Model interpretation — temporal attention", expanded=False):
            weights = attention[0].detach().cpu().numpy()
            x = np.linspace(0.0, 1.0, len(weights))

            fig, ax = plt.subplots(figsize=(9, 2.8))
            ax.plot(x, weights, linewidth=1.1)
            ax.set_xlabel("Normalised ground-motion sequence position")
            ax.set_ylabel("Attention weight")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.caption(
                "Attention weights show which resampled time locations received "
                "greater model attention; they are not causal attributions."
            )

        # Download result.
        result = {
            "Predicted_Total_Energy_kN_m": predicted_energy,
            "Earthquake_return_period_label": irp_label,
            "iRP": selected_irp,
            "FH_m": selected_fh,
            "Building_input_mode": mode,
            "Ground_motion_X": file1.name if file1 else "",
            "Ground_motion_Y": file2.name if file2 else "",
        }

        for column in INPUT_COLUMNS:
            result[column] = building_values[column]

        csv = pd.DataFrame([result]).to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "Download prediction record",
            data=csv,
            file_name="total_energy_prediction.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error("Prediction failed.")
        st.exception(exc)


# =============================================================================
# ABOUT
# =============================================================================

st.divider()

with st.expander("About this research tool", expanded=False):
    st.markdown(
        """
        This open-access interface is an engineering research surrogate, not a
        replacement for nonlinear structural analysis. Predictions are most
        defensible for structural and hazard conditions represented by the
        training database.

        **Three input modes are provided:**
        - **Reference building:** easiest reproducible use of a fully specified
          building already represented in the model database.
        - **Quick custom building:** simplified exploratory use; omitted
          member-level properties are fixed to training means.
        - **Full custom building:** preferred for a new engineering case because
          all trained structural inputs are explicitly supplied.
        """
    )

    provenance = BUNDLE.get("target_provenance")
    if provenance:
        st.markdown(f"**Target provenance stored with model:** {provenance}")

    st.markdown(
        f"**Model sequence length:** {TIME_STEPS} resampled steps  \n"
        f"**Model static variables:** {len(INPUT_COLUMNS)}"
    )
