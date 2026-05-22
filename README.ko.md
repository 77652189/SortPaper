<div align="center">

# 📄 SortPaper

**학술 논문 파싱, 품질 판정, 시맨틱 검색 파이프라인**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4A90D9)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid--Search-DC382D)](https://qdrant.tech)
[![DeepSeek](https://img.shields.io/badge/DeepSeek V4 Pro%20%26%20v4--pro-4D6BFE)](https://deepseek.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**언어:**
[🇬🇧 English](README.md) &nbsp;|&nbsp;
[🇨🇳 中文](README.zh.md) &nbsp;|&nbsp;
[🇯🇵 日本語](README.ja.md) &nbsp;|&nbsp;
🇰🇷 한국어

</div>

---

## 개요

**SortPaper**는 로컬 우선 학술 논문 처리 파이프라인입니다. PDF에서 구조화된 콘텐츠(텍스트, 표, 이미지)를 추출하고 LLM 심판으로 품질 평가, Qdrant 벡터 DB에 Hybrid Search 저장, Streamlit GUI로 탐색·검색·일괄 가져오기를 제공합니다.

## ✨ 주요 기능

| 기능 | 설명 |
|---|---|
| 📝 텍스트 추출 | PyMuPDF 레이아웃 인식 텍스트 청킹(2단 감지 포함) |
| 📋 표 감지 | pdfplumber + PyMuPDF + camelot 3엔진, 테두리 없는 표 대응 |
| 🖼️ 이미지 캡션 | qwen3-vl-plus 서브그림 독립 식별 및 설명 생성 |
| ⚖️ LLM 심판 | DeepSeek V4 Pro 품질 평가, 합격 청크 재시도 시 자동 스킵 |
| 💾 Qdrant 저장 | Hybrid Search(Dense+Sparse+RRF) + qwen3-rerank 2차 정렬 |
| 📉 강등 저장 | 표 구조 불량 시 'degraded' 보존, 오탐(참고문헌 등)은 폐기 |
| 🔁 스마트 재시도 | 이미지 재시도는 DeepSeek 텍스트 재작성, 표 재시도는 파서 전환 |
| 🖥️ GUI | 원클릭 가져오기·배치 드래그앤드롭·벡터 라이브러리 관리 |
| 📊 품질 평가 | 4단계: 분류 → Map-Reduce 요약 → 청크 컨텍스트 → 저장 |
| 🤖 Agent 검색 | Qwen-plus 자율 함수 호출 다중 라운드 검색·종합 제안 |

## 🏗️ 아키텍처

```
PDF
 │
 ▼
코디네이터
 │
 ├──► 텍스트 워커  ──► Judge (DeepSeek) ──┐
 ├──► 표 워커      ──► Judge (DeepSeek) ──┤──► 병합 ──► Qdrant
 └──► 이미지 워커  ──► Judge (DeepSeek) ──┘
  (qwen3-vl-plus)    ▲                      │
                    └── 재시도(재작성) ──────┘
```

**레이어 구성:**

- **파서 레이어** — PyMuPDFParser, TableParser(3엔진), VisionParser(qwen3-vl-plus)
- **판정 레이어** — LLMJudge(DeepSeek V4 Pro, 섹션 인식)
- **품질 평가 레이어** — PaperQualityEvaluator(분류→Map-Reduce), Reduce는 DeepSeek V4 Pro
- **저장 레이어** — QdrantStore(Hybrid Search + Rerank)
- **오케스트레이션** — LangGraph 병렬 fan-out/fan-in + 합격 스킵 재시도

## 🚀 빠른 시작

**1. 클론 및 의존성 설치**

```bash
git clone https://github.com/77652189/SortPaper.git
cd SortPaper
pip install -r requirements.txt
```

**2. DashScope API 키 설정**

```bash
cp .env.example .env
```

> [DashScope 콘솔](https://dashscope.aliyun.com)에서 API 키를 발급받으세요. ¥20 충전으로 약 10편의 논문을 처리할 수 있습니다.

**3. GUI 실행**

```bash
streamlit run app.py
```

브라우저에서 **http://localhost:8501**을 열어주세요.

## 🖥️ GUI 사용법

1. **PDF 선택** — 임의의 PDF 업로드 또는 내장 샘플 논문 선택
2. **모드 선택:**
   - **빠른 미리보기** — 로컬 파서만, 즉시 결과, API 비용 없음
   - **전체 파이프라인** — Judge 실행, 품질 평가·저장 수동 트리거
   - **원클릭 가져오기** — 파싱→판정→평가→저장, 완전 자동
3. **🚀 파싱 시작 클릭**
4. 탭에서 결과 확인:
   - 📊 개요 · 📝 텍스트 · 🖼️ 이미지 · 📋 표 · 📐 PDF 복원 · 🔍 검색
5. **벡터 라이브러리** — 사이드바에서 등록 논문 목록·논문별 삭제

## 📁 프로젝트 구조

```
SortPaper/
├── app.py                    # Streamlit GUI
├── src/
│   ├── parsers/              # 각종 파서 (PyMuPDF/pdfplumber/camelot/VL)
│   ├── judge/                # LLM 심판 + 논문 품질 평가
│   ├── store/                # Qdrant (Hybrid Search)
│   ├── agent/                # 문헌 검색 Agent
│   └── graph/                # LangGraph 파이프라인
└── data/sample_papers/       # 샘플 PDF
```

## 🛠️ 기술 스택

| 컴포넌트 | 기술 |
|---|---|
| 텍스트 파싱 | PyMuPDF (fitz) |
| 표 파싱 | pdfplumber + PyMuPDF + camelot (3엔진) |
| 이미지 캡션 | qwen3-vl-plus (DashScope) |
| LLM 심판 | DeepSeek V4 Pro |
| 품질 평가 | DeepSeek V4 Pro(분류/Map) + DeepSeek V4 Pro(Reduce) |
| 임베딩 | text-embedding-3-large (OpenAI, 3072d dense) |
| 재정렬 | qwen3-rerank (DashScope) |
| 벡터 저장소 | Qdrant (Hybrid Search: Dense + Sparse + RRF) |
| Agent | Qwen-plus (DashScope Function Calling) |
| 파이프라인 | LangGraph |
| GUI | Streamlit |

---

<div align="center">

Made with ❤️ &nbsp;·&nbsp; [GitHub Issues](https://github.com/77652189/SortPaper/issues)

</div>
