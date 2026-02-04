import pandas as pd
import re 

def extract_alphamissense_rows(
    alphamissense_tsv: str,
    sav_list: list[str],
    label_list: list | None = None,
    threshold: float = 0.452,
):
    """
    Extract AlphaMissense prediction rows corresponding to a list of SAVs.

    Parameters
    ----------
    alphamissense_tsv : str
        Path to AlphaMissense TSV file.
    sav_list : list of str
        SAV list in format: "P21817 2458 R H"
    label_list : list or None
        Optional labels corresponding to sav_list
    threshold : float
        Pathogenicity score threshold

    Returns
    -------
    pd.DataFrame
        Subset of AlphaMissense dataframe matching the SAVs.
    """

    # Load AlphaMissense file
    df_am = pd.read_csv(alphamissense_tsv, sep="\t")

    # Parse SAV list
    sav_df = pd.DataFrame(
        [s.split() for s in sav_list],
        columns=["# Uniprot ACC", "position", "a.a.1", "a.a.2"]
    )
    sav_df["position"] = sav_df["position"].astype(int)

    # Attach labels if provided
    if label_list is not None:
        if len(label_list) != len(sav_list):
            raise ValueError("label_list must have the same length as sav_list")
        sav_df["label"] = label_list

    # Merge to extract matching rows
    df_hits = df_am.merge(
        sav_df,
        on=["# Uniprot ACC", "position", "a.a.1", "a.a.2"],
        how="inner"
    )

    # Reconstruct SAV_coords
    df_hits["SAV_coords"] = (
        df_hits["# Uniprot ACC"].astype(str) + " " +
        df_hits["position"].astype(str) + " " +
        df_hits["a.a.1"].astype(str) + " " +
        df_hits["a.a.2"].astype(str)
    )

    # Refined pathogenicity class
    df_hits["refined pathogenicity class"] = df_hits["pathogenicity score"].apply(
        lambda x: "likely_pathogenic" if x >= threshold else "likely_benign"
    )

    return df_hits

def extract_rhapsody(rhapsody_txt, sav_list, label_list=None):
    rows = []

    with open(rhapsody_txt) as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = re.split(r"\s{2,}", line)
            if len(parts) < 9:
                continue

            rows.append({
                "SAV_coords": parts[0],
                "score": float(parts[2]),
                "prob.": float(parts[3]),
                "class": parts[4],
                "PolyPhen2_score": parts[5],
                "PolyPhen2_class": parts[6],
                "EVmutation_score": parts[7],
                "EVmutation_class": parts[8],
            })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="SAV_coords")

    # keep only requested SAVs
    df = df[df["SAV_coords"].isin(sav_list)].copy()

    # add labels if provided
    if label_list is not None:
        df["labels"] = [label_list[sav_list.index(sav)] for sav in df["SAV_coords"]]

    return df