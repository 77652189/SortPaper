from __future__ import annotations

from evals.fielded_lexical_eval import (
    build_document_frequency,
    fielded_score,
    payload_field_texts,
    query_terms,
)


def test_payload_field_texts_keeps_structured_metadata() -> None:
    fields = payload_field_texts({
        "paper_title": "LNT paper",
        "target_products": ["Lacto-N-tetraose"],
        "organisms": ["E. coli"],
        "genes": ["lgtA"],
        "seo_terms": ["UDP-Gal"],
        "content": "generic production text",
    })

    assert "lacto n tetraose" in fields["entities"]
    assert "escherichia coli" in fields["entities"]
    assert "lgta" in fields["entities"]
    assert "udp gal" in fields["entities"]


def test_fielded_score_prefers_entity_match_over_body_only_match() -> None:
    entity_doc = payload_field_texts({
        "target_products": ["Lacto-N-tetraose"],
        "genes": ["lgtA"],
        "content": "short note",
    })
    body_doc = payload_field_texts({
        "content": "Lacto-N-tetraose lgtA mentioned only in a generic body paragraph.",
    })
    docs = [entity_doc, body_doc]
    df = build_document_frequency(docs)
    terms = query_terms("lacto-N-tetraose lgtA")

    assert fielded_score(entity_doc, terms, total_docs=2, df=df) > fielded_score(
        body_doc,
        terms,
        total_docs=2,
        df=df,
    )
