from __future__ import annotations

from src.application.external_briefing import MetadataBriefClassifier
from src.domain.external_papers import make_candidate


def test_rule_classifier_requires_pichia_for_pichia_categories() -> None:
    candidate = make_candidate(
        source="openalex",
        title="Urban transport model for human development",
        abstract="This study discusses transport systems and prediction models.",
        doi="10.1/transport",
    )

    decision = MetadataBriefClassifier().rule_classify(candidate)

    assert decision.category_id == "other"


def test_rule_classifier_matches_y103_pichia_lactoferrin_category() -> None:
    candidate = make_candidate(
        source="pubmed",
        title="Recombinant lactoferrin expression in Pichia pastoris",
        abstract="Pichia pastoris was engineered for heterologous recombinant lactoferrin secretion.",
        doi="10.1/lactoferrin",
    )

    decision = MetadataBriefClassifier().rule_classify(candidate)

    assert decision.category_id == "03"
    assert decision.category_name == "毕赤酵母（乳铁蛋白/骨桥蛋白）"


def test_rule_classifier_does_not_score_generic_pichia_markers_only() -> None:
    candidate = make_candidate(
        source="openalex",
        title="Genome resources for Pichia pastoris",
        abstract="Pichia pastoris is mentioned as an organism in a genome resource note.",
        doi="10.1/generic",
    )

    decision = MetadataBriefClassifier().rule_classify(candidate)

    assert decision.category_id == "other"


def test_rule_classifier_does_not_match_short_keywords_inside_words() -> None:
    candidate = make_candidate(
        source="openalex",
        title="Pichia pastoris precursor production",
        abstract="The precursor is purified after fermentation without discussing organelle transport.",
        doi="10.1/precursor",
    )

    decision = MetadataBriefClassifier().rule_classify(candidate)

    assert "er" not in decision.key_evidence


def test_rule_classifier_does_not_promote_lactoferrin_clinical_use_to_y103() -> None:
    candidate = make_candidate(
        source="pubmed",
        title="Effects of human lactoferrin on the adult gut microbiome",
        abstract=(
            "Human lactoferrin produced by Komagataella phaffii was used in a "
            "randomized trial to evaluate gut microbiome and fecal metabolites."
        ),
        doi="10.1/clinical",
    )

    decision = MetadataBriefClassifier().rule_classify(candidate)

    assert decision.category_id == "other"
