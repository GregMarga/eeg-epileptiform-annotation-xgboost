import pandas as pd

for tag in ["MODEL_A", "MODEL_B"]:
    df = pd.read_csv(f"clinical_review_csv/predictions_{tag}.csv")
    print(f"\n===== {tag} =====")

    ct = pd.crosstab(df.segment_type, df.binary_prediction)
    ct.columns = [f"pred={c}" for c in ct.columns]
    ct["pos_rate"] = (ct.get("pred=1", 0) / ct.sum(axis=1)).round(3)
    print(ct)

    print("\nreference vs prediction (whole test set):")
    print(pd.crosstab(df.reference_label, df.binary_prediction))

    print("\nper-segment positive rate:")
    per_seg = df.groupby(["segment_id", "segment_type"]).agg(
        n=("binary_prediction", "size"),
        pred_pos=("binary_prediction", "sum"),
        ref_pos=("reference_label", "sum"),
        mean_p=("probability", "mean"),
    )
    per_seg["pred_rate"] = (per_seg.pred_pos / per_seg.n).round(3)
    print(per_seg.to_string())