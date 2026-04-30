<div align="center">

# 📄 SortPaper

**학술 논문 파싱, 품질 판정, 시맨틱 검색 파이프라인**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4A90D9)](https://github.com/langchain-ai/langgraph)
[![DashScope](https://img.shields.io/badge/DashScope-qwen--max-FF6A00)](https://dashscope.aliyun.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**언어:**
[🇬🇧 English](README.md) &nbsp;|&nbsp;
[🇨🇳 中文](README.zh.md) &nbsp;|&nbsp;
[🇯🇵 日本語](README.ja.md) &nbsp;|&nbsp;
🇰🇷 한국어

</div>

---

## 개요

**SortPaper**는 로컬 우선 학술 논문 처리 파이프라인입니다. PDF 연구 논문에서 구조화된 콘텐츠(텍스트, 표, 이미지)를 추출하고, LLM 심판으로 각 청크의 품질을 평가하며, 합격한 청크를 FAISS 벡터 스토어에 저장하고, Streamlit GUI를 통해 대화형 탐색 및 시맨틱 검색을 제공합니다.

## ✨ 주요 기능

| 기능 | 설명 |
|---|---|
| 📝 텍스트 추출 | PyMuPDF 기반 레이아웃 인식 텍스트 청킹(2단 레이아웃 감지 포함) |
| 📋 표 감지 | pdfplumber 격자선 모드 파싱 + 오탐 필터링 |
| 🖼️ 이미지 캡션 | Qwen-VL-Max 비전 모델이 자연어 설명 생성 |
| ⚖️ LLM 심판 | qwen-max로 각 청크 품질 평가, 실패 시 최대 3회 재시도 |
| 💾 FAISS 저장소 | 합격 청크를 text-embedding-v3로 임베딩하여 인덱싱 |
| 📉 강등 저장 | 일부 청크가 실패해도 합격한 청크는 반드시 저장 |
| 🖥️ GUI | Streamlit 웹 인터페이스: 업로드, 파싱, 탐색, 시맨틱 검색 |

## 🏗️ 아키텍처

```
PDF
 │
 ▼
코디네이터
 │
 ├──► 텍스트 워커  ──► Judge 텍스트  ──┐
 ├──► 표 워커      ──► Judge 표      ──┤──► 병합 ──► FAISS 저장
 └──► 이미지 워커  ──► Judge 이미지  ──┘
         ▲                             │
         └──── 재시도 (최대 3회) ────────┘
```

**레이어 구성:**

- **파서 레이어** — `PyMuPDFParser`, `TableParser`, `VisionParser`
- **판정 레이어** — `LLMJudge` (qwen-max, 섹션 인식 프롬프트)
- **저장 레이어** — `FAISSStore` (text-embedding-v3, 1024차원)
- **오케스트레이션** — LangGraph 팬아웃/팬인 상태 머신

## 🚀 빠른 시작

**1. 클론 및 의존성 설치**

```bash
git clone https://github.com/77652189/SortPaper.git
cd SortPaper
pip install -r requirements.txt
```

**2. DashScope API 키 설정**

```bash
echo "DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx" > .env
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
   - **빠른 미리보기** — 로컬 파서만 사용, 즉시 결과, API 비용 없음
   - **전체 파이프라인** — Judge + VisionParser + FAISS 실행 (API 할당량 사용)
3. **🚀 파싱 시작 클릭**
4. 다섯 개의 탭에서 결과 확인:
   - 📊 개요 · 📝 텍스트 · 🖼️ 이미지 · 📋 표 · 🔍 시맨틱 검색

## 📁 프로젝트 구조

```
SortPaper/
├── app.py                    # Streamlit GUI
├── main.py                   # CLI 배치 실행
├── src/
│   ├── parsers/              # 각종 파서
│   ├── judge/                # LLM 심판 + 프롬프트
│   ├── store/                # FAISS 저장소 + 텍스트 분할
│   └── graph/                # LangGraph 파이프라인
├── scripts/                  # 검증 및 디버그 스크립트
└── data/sample_papers/       # 샘플 PDF
```

## 🛠️ 기술 스택

| 컴포넌트 | 기술 |
|---|---|
| 텍스트 파싱 | PyMuPDF (fitz) |
| 표 파싱 | pdfplumber |
| 이미지 캡션 | Qwen-VL-Max (DashScope) |
| LLM 심판 | qwen-max (DashScope) |
| 임베딩 | text-embedding-v3 (DashScope) |
| 벡터 저장소 | FAISS |
| 파이프라인 | LangGraph |
| GUI | Streamlit |

---

<div align="center">

Made with ❤️ &nbsp;·&nbsp; [GitHub Issues](https://github.com/77652189/SortPaper/issues)

</div>
