import pandas as pd
import numpy as np


def drop_high_missing(df, threshold=0.5):
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    print(f"Dropping {len(cols_to_drop)} columns with >{threshold*100:.0f}% missing")
    return df.drop(columns=cols_to_drop)


def transform_days_columns(df):
    df = df.copy()
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365
        df = df.drop(columns=["DAYS_BIRTH"])
    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] > 0).astype(int)
        df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"].clip(upper=0)) / 365
        df = df.drop(columns=["DAYS_EMPLOYED"])
    for col in ["DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE", "DAYS_REGISTRATION"]:
        if col in df.columns:
            df[col.replace("DAYS_", "YEARS_")] = (-df[col]) / 365
            df = df.drop(columns=[col])
    return df


def create_derived_features(df):
    df = df.copy()
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["LOAN_TERM_MONTHS"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
        df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
    return df


def encode_categoricals(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        if df[col].nunique() == 2:
            df[col] = pd.factorize(df[col])[0]
        else:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


def impute_missing(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def full_pipeline(df):
    df = drop_high_missing(df)
    df = transform_days_columns(df)
    df = create_derived_features(df)
    df = encode_categoricals(df)
    df = impute_missing(df)
    print(f"Pipeline complete. Final shape: {df.shape}")
    return df
