from __future__ import annotations

from src.parsers.table import cleanup


def test_normalize_table_pads_rows_and_removes_empty_rows() -> None:
    assert cleanup.normalize_table([
        ["A", "B", "C"],
        ["1", None],
        ["", "", ""],
    ]) == [
        ["A", "B", "C"],
        ["1", "", ""],
    ]


def test_to_markdown_uses_detected_header_row() -> None:
    table = [
        ["Table 1. Results", ""],
        ["Product", "Yield"],
        ["X", "80%"],
    ]

    assert cleanup.to_markdown(table) == (
        "| Product | Yield |\n"
        "| --- | --- |\n"
        "| X | 80% |"
    )


def test_unsplit_twin_columns_folds_symmetric_halves() -> None:
    rows = [
        ["A", "B", "A", "B"],
        ["1", "2", "3", "4"],
    ]

    assert cleanup.unsplit_twin_columns(rows) == [
        ["A", "B"],
        ["1", "2"],
        ["3", "4"],
    ]


def test_unsplit_twin_columns_preserves_left_then_right_reading_order() -> None:
    rows = [
        ["strain", "characteristic", "source", "strain", "characteristic", "source"],
        ["DH5α", "left desc", "Invitrogen", "B16", "right desc", "this study"],
        ["BL21(DE3)", "left desc 2", "Invitrogen", "B17", "right desc 2", "this study"],
    ]

    assert cleanup.unsplit_twin_columns(rows) == [
        ["strain", "characteristic", "source"],
        ["DH5α", "left desc", "Invitrogen"],
        ["BL21(DE3)", "left desc 2", "Invitrogen"],
        ["B16", "right desc", "this study"],
        ["B17", "right desc 2", "this study"],
    ]


def test_unsplit_twin_columns_infers_header_when_right_half_starts_data() -> None:
    rows = [
        ["strains", "characteristics", "source", "B16", "right desc", "this study"],
        ["DH5α", "left desc", "Invitrogen", "B17", "right desc 2", "this study"],
        ["BL21(DE3)", "left desc 2", "Invitrogen", "B18", "right desc 3", "this study"],
    ]

    assert cleanup.unsplit_twin_columns(rows) == [
        ["strains", "characteristics", "source"],
        ["DH5α", "left desc", "Invitrogen"],
        ["BL21(DE3)", "left desc 2", "Invitrogen"],
        ["B16", "right desc", "this study"],
        ["B17", "right desc 2", "this study"],
        ["B18", "right desc 3", "this study"],
    ]


def test_repair_table_structure_drops_empty_columns_and_merges_pm_values() -> None:
    rows = [
        ["strain", "LNT II titer", "", "LNT titer", ""],
        ["B1", "0.37", "± 0.05", "0.22", "± 0.03"],
        ["B2", "1.67", "± 0.08", "0.62", "± 0.06"],
    ]

    repaired, actions = cleanup.repair_table_structure(rows)

    assert repaired == [
        ["strain", "LNT II titer", "LNT titer"],
        ["B1", "0.37 ± 0.05", "0.22 ± 0.03"],
        ["B2", "1.67 ± 0.08", "0.62 ± 0.06"],
    ]
    assert any(action.startswith("merge_plus_minus_columns") for action in actions)


def test_repair_table_structure_merges_word_fragment_columns() -> None:
    rows = [
        ["Field", "Basic", "GEM", "Fine", "grain"],
        ["Organism", "P.pas", "toris", "Mammalia", "nce"],
        ["Compartment", "cyto", "sol", "secreto", "ry"],
    ]

    repaired, actions = cleanup.repair_table_structure(rows)

    assert repaired == [
        ["Field", "BasicGEM", "Finegrain"],
        ["Organism", "P.pastoris", "Mammaliance"],
        ["Compartment", "cytosol", "secretory"],
    ]
    assert any(action.startswith("merge_fragment_columns") for action in actions)


def test_structural_quality_allows_stable_tables_with_descriptive_columns() -> None:
    rows = [
        ["host cells", "characteristics", "medium", "titer", "references"],
        [
            "E.coli",
            "the lgtA and lacY genes were chromosomally integrated and lacZ was deleted",
            "minimal medium",
            "0.22",
            "24",
        ],
        [
            "E.coli",
            "the pathway genes were screened and strengthened for production",
            "Gly, lactose",
            "1.20",
            "26",
        ],
        [
            "B.subtilis",
            "the metabolic pathway was introduced and optimized for synthesis",
            "shake flask",
            "0.80",
            "31",
        ],
    ]

    quality = cleanup.structural_quality(rows)

    assert quality["allows_descriptive_columns"] is True
    assert quality["fallback_to_vision"] is False
    assert quality["fallback_reasons"] == []
