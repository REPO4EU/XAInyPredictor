import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


AA_HYDRO = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

AA_GROUPS = {
    "hydrophobic": set("AVILMFWYC"),
    "aromatic": set("FWYH"),
    "polar": set("STNQYC"),
    "basic": set("KRH"),
    "acidic": set("DE"),
    "small": set("AGCSTP"),
    "special": set("GPC"),
    "aliphatic": set("AVILM"),
    "amide": set("NQ"),
}

AA_FORMAL_CHARGE = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}

BASE_NUMERIC_COLUMNS = [
    "Molecular_weight",
    "GRAVY",
    "instability",
    "mhcflurry_affinity_percentile",
    "pI",
    "mhcflurry_processing_score",
    "mhcflurry_presentation_score",
    "Half_life",
    "charge",
    "score",
    "hydro_P2",
    "hydro_P9",
]

METADATA_COLUMNS = ["Peptide", "hla", "pseudosequence"]


@dataclass
class NeoagPreparedData:
    features: pd.DataFrame
    metadata: pd.DataFrame
    target: pd.Series | None


def _safe_fraction(seq: str, residue_set: set[str]) -> float:
    length = max(len(seq), 1)
    return sum(aa in residue_set for aa in seq) / length


def _mean_hydrophobicity(seq: str) -> float:
    vals = [AA_HYDRO[aa] for aa in seq if aa in AA_HYDRO]
    return float(np.mean(vals)) if vals else 0.0


def _approx_sidechain_charge(seq: str) -> float:
    return float(sum(AA_FORMAL_CHARGE.get(aa, 0.0) for aa in seq))


def _sequence_group_features(seq: str, prefix: str) -> dict[str, float]:
    features = {
        f"{prefix}_hydro_mean": _mean_hydrophobicity(seq),
        f"{prefix}_estimated_sidechain_charge": _approx_sidechain_charge(seq),
    }
    for group_name, residue_set in AA_GROUPS.items():
        features[f"{prefix}_frac_{group_name}"] = _safe_fraction(seq, residue_set)
    return features


def _extract_hla_tokens(hla: str) -> dict[str, str]:
    match = re.match(r"HLA-([A-Z])(\d+):(\d+)", str(hla))
    if match is None:
        return {"hla_locus": "other", "hla_group": "unknown", "hla_protein": "unknown"}
    locus, group, protein = match.groups()
    return {"hla_locus": locus, "hla_group": f"{locus}{group}", "hla_protein": protein}


def _clean_raw_df(df: pd.DataFrame, target: str | None) -> pd.DataFrame:
    required = set(METADATA_COLUMNS + BASE_NUMERIC_COLUMNS)
    if target:
        required.add(target)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required neoantigen column(s): {sorted(missing)}")

    clean_df = df.copy()
    for col in METADATA_COLUMNS:
        clean_df[col] = clean_df[col].where(clean_df[col].notna(), "")

    clean_df["Peptide"] = clean_df["Peptide"].astype(str).str.strip().str.upper()
    clean_df["pseudosequence"] = clean_df["pseudosequence"].astype(str).str.strip().str.upper()
    clean_df["hla"] = clean_df["hla"].astype(str).str.strip()

    empty_text_cols = [col for col in METADATA_COLUMNS if clean_df[col].eq("").any()]
    if empty_text_cols:
        raise ValueError(f"Neoantigen text column(s) contain empty values: {empty_text_cols}")

    for col in BASE_NUMERIC_COLUMNS:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    if clean_df[BASE_NUMERIC_COLUMNS].isna().any().any():
        bad_cols = clean_df[BASE_NUMERIC_COLUMNS].columns[
            clean_df[BASE_NUMERIC_COLUMNS].isna().any()
        ].tolist()
        raise ValueError(f"Neoantigen numeric column(s) contain invalid values: {bad_cols}")

    if target and target in clean_df.columns:
        clean_df[target] = pd.to_numeric(clean_df[target], errors="coerce").astype("Int64")

    return clean_df


def prepare_neoantigen_features(
    raw_df: pd.DataFrame,
    feature_order: list[str],
    target: str | None = "Qualitative_Measure",
) -> NeoagPreparedData:
    target_col = target if target in raw_df.columns else None
    clean_df = _clean_raw_df(raw_df, target_col)
    metadata = clean_df[METADATA_COLUMNS].copy().reset_index(drop=True)
    features = clean_df[BASE_NUMERIC_COLUMNS].copy()

    peptide_features = (
        clean_df["Peptide"].apply(lambda seq: _sequence_group_features(seq, "peptide")).apply(pd.Series)
    )
    pseudo_features = (
        clean_df["pseudosequence"].apply(lambda seq: _sequence_group_features(seq, "pseudo")).apply(pd.Series)
    )
    features["peptide_length"] = clean_df["Peptide"].str.len().astype(float)
    features["anchor_hydro_mean"] = (clean_df["hydro_P2"] + clean_df["hydro_P9"]) / 2.0
    features["anchor_hydro_diff"] = clean_df["hydro_P2"] - clean_df["hydro_P9"]

    hla_tokens = clean_df["hla"].apply(_extract_hla_tokens).apply(pd.Series)
    hla_locus_dummies = pd.get_dummies(hla_tokens["hla_locus"], prefix="hla_locus")
    for col in ["hla_locus_A", "hla_locus_B", "hla_locus_C"]:
        if col not in hla_locus_dummies.columns:
            hla_locus_dummies[col] = 0

    features = pd.concat(
        [
            features.reset_index(drop=True),
            peptide_features.reset_index(drop=True),
            pseudo_features.reset_index(drop=True),
            hla_locus_dummies[["hla_locus_A", "hla_locus_B", "hla_locus_C"]].reset_index(drop=True),
        ],
        axis=1,
    )
    features = features.reindex(columns=feature_order, fill_value=0.0).astype(float)

    target_series = None
    if target_col:
        target_series = clean_df[target_col].astype(int).reset_index(drop=True)

    return NeoagPreparedData(features=features, metadata=metadata, target=target_series)
