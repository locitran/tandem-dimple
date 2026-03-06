import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.features.Pfam import parse_hmmscan, read_pfam_data
except Exception as e:  # pragma: no cover
    pytest.skip(f"Cannot import Pfam module: {e}", allow_module_level=True)


def _domtbl_row(seq_id: str) -> str:
    # Minimal domtblout-like row compatible with parse_hmmscan column indexing.
    return (
        f"Trypsin PF00089.32 220 {seq_id} - 686 "
        "4.5e-54 184.0 0.0 1 1 1.1e-57 6.4e-54 183.5 0.0 "
        "1 220 445 679 445 679 0.92 Trypsin\n"
    )


def test_read_pfam_data_has_expected_keys():
    pfam_data = read_pfam_data()
    assert "Trypsin" in pfam_data
    assert "ga_dom" in pfam_data["Trypsin"]
    assert "ga_seq" in pfam_data["Trypsin"]


def test_parse_hmmscan_multiple_proteins(tmp_path):
    hmmscan_file = tmp_path / "hmmscan_out"
    hmmscan_file.write_text(_domtbl_row("O00187") + _domtbl_row("O00189"), encoding="utf-8",)

    pfam_data = read_pfam_data()
    result = parse_hmmscan(str(hmmscan_file), pfam_data)

    # Both proteins should be parsed from one file.
    assert "O00187" in result
    assert "O00189" in result

    # Each should contain Trypsin accession PF00089.32.
    assert "PF00089.32" in result["O00187"]
    assert "PF00089.32" in result["O00189"]

    # Simple field check.
    loc = result["O00187"]["PF00089.32"]["locations"][0]
    assert loc["ali_start"] == 445
    assert loc["ali_end"] == 679
    assert loc["score"] == 183.5

# pytest -q test/pfam_test.py
# from src.features.SEQ import SEQfeatures
# acc = 'O00187'
# u  = SEQfeatures(acc, ['O00187 118 R C'])
# u._searchPfam()