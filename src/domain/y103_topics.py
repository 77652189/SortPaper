from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Y103Topic:
    category_id: str
    name: str
    description: str
    keywords: tuple[str, ...]
    retrieval_query: str
    smoke_profile: str = "evidence"

    @property
    def topic_id(self) -> str:
        return f"Y103-{self.category_id}"


Y103_TOPICS: tuple[Y103Topic, ...] = (
    Y103Topic(
        "01",
        "毕赤酵母（综述）",
        "系统梳理毕赤酵母表达系统、代谢工程、合成生物学、特定产物应用及相关进展的综述。",
        ("pichia", "komagataella", "pastoris", "review", "overview", "expression system", "synthetic biology"),
        "Pichia pastoris Komagataella expression system synthetic biology review",
        "quick",
    ),
    Y103Topic(
        "02",
        "乳铁蛋白/骨桥蛋白（综述）",
        "系统回顾乳铁蛋白或骨桥蛋白结构功能、提取技术、人体合成与功能、市场来源等的综述。",
        ("lactoferrin", "osteopontin", "review", "overview", "milk protein", "bioactive protein"),
        "lactoferrin osteopontin bioactive milk protein review",
        "quick",
    ),
    Y103Topic(
        "03",
        "毕赤酵母（乳铁蛋白/骨桥蛋白）",
        "在毕赤酵母中异源合成乳铁蛋白或骨桥蛋白，包括合成优化、来源形式、产量提升、放大和活性验证。",
        ("pichia", "komagataella", "pastoris", "lactoferrin", "osteopontin", "heterologous", "recombinant"),
        "Pichia pastoris recombinant lactoferrin osteopontin expression yield",
    ),
    Y103Topic(
        "04",
        "乳铁蛋白/骨桥蛋白（翻译后修饰）",
        "研究乳铁蛋白或骨桥蛋白翻译后修饰，包括糖基化、磷酸化及其功能影响和表达系统差异。",
        ("lactoferrin", "osteopontin", "glycosylation", "phosphorylation", "post-translational", "ptm"),
        "lactoferrin osteopontin glycosylation phosphorylation post-translational modification",
    ),
    Y103Topic(
        "05",
        "毕赤酵母（蛋白转运全过程）",
        "围绕毕赤酵母分泌通路、信号肽、ER、高尔基体、囊泡运输、胞外分泌及通路优化的研究。",
        ("pichia", "pastoris", "secretion", "secretory pathway", "er", "golgi", "vesicle", "transport"),
        "Pichia pastoris secretion secretory pathway ER Golgi vesicle transport",
    ),
    Y103Topic(
        "06",
        "毕赤酵母（氧化应激-氧化还原/ER应激）",
        "甲醇代谢氧化应激、重组蛋白表达导致的ER应激、UPR、氧化还原酶和缓解策略。",
        ("pichia", "pastoris", "oxidative stress", "redox", "methanol", "upr", "hac1", "ire1", "ero1", "pdi"),
        "Pichia pastoris oxidative stress redox UPR Hac1 Ire1 recombinant protein expression",
    ),
    Y103Topic(
        "07",
        "毕赤酵母细胞壁相关",
        "毕赤酵母细胞壁结构功能、完整性信号通路，以及细胞壁改造优化表达系统的研究。",
        ("pichia", "pastoris", "cell wall", "cell-wall", "wall integrity", "mannan", "glucan"),
        "Pichia pastoris cell wall integrity mannan glucan protein secretion",
    ),
    Y103Topic(
        "08",
        "毕赤酵母（碳源代谢）",
        "毕赤酵母不同碳源利用、代谢途径、调控机制及提高蛋白产量的代谢工程改造。",
        ("pichia", "pastoris", "carbon source", "methanol", "glycerol", "glucose", "metabolism", "aox"),
        "Pichia pastoris carbon source methanol glycerol glucose AOX metabolism protein production",
    ),
    Y103Topic(
        "09",
        "毕赤酵母（表达元件）",
        "启动子、终止子、5'UTR、表达强度、诱导特性、分泌效率相关表达元件和工程改造。",
        ("pichia", "pastoris", "promoter", "terminator", "aox1", "gap", "5'utr", "expression element"),
        "Pichia pastoris promoter terminator AOX1 GAP 5'UTR expression element",
    ),
    Y103Topic(
        "10",
        "毕赤酵母（信号肽）",
        "信号肽筛选、预测、α-MF优化、信号肽与目标蛋白组合及共/翻译后转运机制。",
        ("pichia", "pastoris", "signal peptide", "alpha-factor", "α-mf", "secretion signal"),
        "Pichia pastoris signal peptide alpha-factor secretion signal recombinant protein",
    ),
    Y103Topic(
        "11",
        "毕赤酵母（密码子优化）",
        "毕赤酵母密码子偏好、密码子优化策略、工具评估及其对转录调控和表达水平的影响。",
        ("pichia", "pastoris", "codon", "codon optimization", "codon usage", "synonymous"),
        "Pichia pastoris codon optimization codon usage recombinant protein expression",
    ),
    Y103Topic(
        "12",
        "毕赤酵母（转录因子）",
        "Mxr1、Mit1、Prm1等转录因子在代谢调控和重组蛋白表达中的功能与机制。",
        ("pichia", "pastoris", "transcription factor", "mxr1", "mit1", "prm1", "transcriptional regulator"),
        "Pichia pastoris transcription factor Mxr1 Mit1 Prm1 metabolic regulation",
    ),
    Y103Topic(
        "13",
        "毕赤酵母（蛋白表达）",
        "毕赤酵母表达抗体片段、工业酶、抗菌肽等重组蛋白的个性化表达策略和瓶颈突破。",
        ("pichia", "pastoris", "recombinant protein", "protein expression", "antibody", "enzyme", "antimicrobial peptide"),
        "Pichia pastoris recombinant protein expression antibody enzyme antimicrobial peptide",
        "quick",
    ),
    Y103Topic(
        "14",
        "毕赤酵母（氨基酸合成）",
        "调控毕赤酵母内源氨基酸代谢以支撑高效重组蛋白表达，包括限速酶、反馈抑制和供应增强。",
        ("pichia", "pastoris", "amino acid", "biosynthesis", "amino-acid", "translation material"),
        "Pichia pastoris amino acid biosynthesis translation material recombinant protein expression",
    ),
    Y103Topic(
        "15",
        "乳铁蛋白/骨桥蛋白（分离纯化）",
        "从毕赤酵母发酵液中高效分离纯化重组乳铁蛋白或骨桥蛋白的方法、原理与技术应用。",
        ("lactoferrin", "osteopontin", "purification", "separation", "chromatography", "fermentation broth"),
        "lactoferrin osteopontin purification separation chromatography fermentation broth",
    ),
    Y103Topic(
        "16",
        "毕赤酵母（间接检测方法）",
        "通过相关指标间接评估毕赤酵母生理状态或生产能力，而非直接测量目标产物。",
        ("pichia", "pastoris", "indirect detection", "indirect assay", "biomarker", "physiological state"),
        "Pichia pastoris indirect assay biomarker physiological state production capacity",
    ),
)


def y103_topic_rows() -> list[dict[str, object]]:
    return [
        {
            "category_id": topic.category_id,
            "topic_id": topic.topic_id,
            "name": topic.name,
            "description": topic.description,
            "keywords": list(topic.keywords),
            "retrieval_query": topic.retrieval_query,
            "smoke_profile": topic.smoke_profile,
        }
        for topic in Y103_TOPICS
    ]


__all__ = ["Y103Topic", "Y103_TOPICS", "y103_topic_rows"]
