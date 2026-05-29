<div align="center">

# SortPaper

**학술 논문 파싱, 품질 평가, 벡터 저장, 시맨틱 검색**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4A90D9)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid--Search-5B21B6)](https://qdrant.tech)
[![DashScope](https://img.shields.io/badge/DashScope-Embedding%20%7C%20Rerank%20%7C%20Vision-FF6A00)](https://dashscope.aliyun.com)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-Judge%20%7C%20Quality-4D6BFE)](https://deepseek.com)

**언어:**
[English](README.md) &nbsp;|&nbsp;
[中文](README.zh.md) &nbsp;|&nbsp;
[日本語](README.ja.md) &nbsp;|&nbsp;
한국어

</div>

---

## 개요

**SortPaper**는 로컬 우선 논문 처리 도구입니다. PDF에서 텍스트, 표, 이미지를 추출해 `LayoutChunk`로 정리하고, LLM Judge로 품질을 평가한 뒤 논문 단위 분류·요약·metadata를 보강하여 Qdrant에 저장합니다. 이후 시맨틱 검색과 Agent 기반 종합 답변에 사용합니다.

현재 코드베이스는 큰 애플리케이션 파일 중심 구조에서 더 명확한 레이어 구조로 정리되고 있습니다. 목표는 단순 chunk 생성이 아니라, 어떤 논문이 답변을 뒷받침하는지, 증거가 어디에 있는지, 신뢰할 수 있는지를 추적하는 검색 기반을 만드는 것입니다.

## 주요 기능

| 기능 | 설명 |
|---|---|
| 텍스트 파싱 | PyMuPDF 기반 레이아웃 인식 chunk 생성 |
| 표 파싱 | pdfplumber / PyMuPDF / camelot 계열 전략, 영역 감지, 후처리, 품질 판단 |
| 이미지 파싱 | qwen3-vl-plus로 그림 및 서브그림 설명 생성 |
| LLM Judge | chunk 단위 품질 평가, 저가치 chunk 필터링, degraded 결과 보존 |
| 논문 품질 평가 | 분류, Map-Reduce 요약, chunk context, 산물·균주·신뢰도 metadata |
| Qdrant 저장 | chunk 단위 벡터 저장, 논문별 삭제, 중복 확인, payload 업데이트 |
| 시맨틱 검색 | DashScope embedding, Qdrant hybrid search, qwen3-rerank, 품질 metadata 표시 |
| Agent 검색 | Qwen-plus tool calling 기반 다중 라운드 문헌 검색 및 종합 답변 |
| Streamlit UI | 단일 논문 파싱, 원클릭 가져오기, 배치 가져오기, 벡터 라이브러리 관리 |

## 아키텍처

```text
PDF
 |
 v
Streamlit UI
 |
 v
Pipeline Orchestration
 |
 +--> Text Parser  --> LLM Judge --+
 +--> Table Parser --> LLM Judge --+--> Merge --> Quality Eval --> Qdrant
 +--> Image Parser --> LLM Judge --+
                                      |
                                      v
                              Search / Agent Answer
```

| 레이어 | 주요 파일 | 역할 |
|---|---|---|
| UI | `app.py`, `app_ui.py`, `app_sidebar.py` | 화면, 입력, 결과 표시, 벡터 라이브러리 작업 |
| 오케스트레이션 | `app_pipeline.py`, `src/graph/pipeline_graph.py` | 미리보기, 전체 파이프라인, 원클릭 가져오기, 품질 평가 |
| 데이터 모델 | `src/parsers/layout_chunk.py` | 텍스트·표·이미지 공통 chunk 표현 |
| 파서 | `src/parsers/*` | PDF 텍스트, 표, 이미지 추출 |
| 표 모듈 | `src/parsers/table/*` | 영역 감지, 추출, 정리, 중복 제거, fallback, Judge metadata |
| Judge | `src/judge/*` | chunk 품질, 표 품질, 논문 단위 평가 |
| Store | `src/store/qdrant_store.py`, `src/store/chunk_storage.py` | embedding, 저장, 검색, rerank, payload 업데이트 |
| Agent | `src/agent/literature_agent.py` | 검색 tool을 이용한 종합 답변 |

## 빠른 시작

```bash
pip install -r requirements.txt
```

프로젝트 디렉터리에 `.env`를 만듭니다.

```bash
DASHSCOPE_API_KEY=your_dashscope_key
DEEPSEEK_API_KEY=your_deepseek_key
```

선택 설정:

```bash
SORTPAPER_EMBEDDING_PROVIDER=dashscope
OPENAI_API_KEY=your_openai_key
OPENAI_EMBEDDING_BASE_URL=https://api.openai.com/v1
```

Qdrant를 실행합니다.

```bash
docker run -p 6333:6333 qdrant/qdrant
```

UI를 실행합니다.

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`을 엽니다.

## 검색 품질 참고

시맨틱 검색은 이미 가져온 논문 안에서만 답할 수 있습니다. 원 논문이 없으면 rerank가 정상이어도 리뷰, 인용 문단, 인접 주제 논문이 반환될 수 있습니다.

검색 결과가 좋지 않으면 먼저 대상 논문이 벡터 라이브러리에 있는지 확인하세요. 그다음 품질 평가 metadata가 있는지, 반환된 chunk가 원 실험 논문인지 리뷰인지 인용 문단인지 확인하고, 이후 query, hybrid search, rerank, UI filter를 조정합니다.

수동 검색과 Agent 검색은 이제 enhanced chunk recall을 기본으로 사용합니다. 수동 검색은 기본적으로 10개 결과를 보여 주므로 평가된 top10 evidence를 바로 확인할 수 있습니다. 이 indexed lexical backfill은 `search_text`와 저빈도 query term으로 evidence 후보를 보강하며, 매번 전체 라이브러리를 스캔하지 않고 원본 상위 retrieval anchor도 보호합니다. rerank를 쓰지 않는 60-case top10 평가에서 `chunk_hit@10`은 `0.4000`에서 `0.6000`으로, `nearby_chunk_hit@10`은 `0.4000`에서 `0.6333`으로 개선되었고 p50 latency는 약 `561ms`에서 `745ms`로 증가했습니다. UI 기본 qwen3-rerank 경로에서는 `chunk_hit@10`이 `0.6667`, `nearby_chunk_hit@10`이 `0.7000`, p50 latency가 약 `1621ms`입니다. 너무 넓은 query나 지연 시간이 민감한 경우 UI에서 끌 수 있습니다.

Agent synthesis는 상위 5개 hit 논문에서 paper-local deeper evidence를 score-rank로 먼저 추가하고, 그다음 nearby chunk를 추가합니다. 이 보강은 tool search ranking 자체를 바꾸지 않습니다. 5개 context 예산과 논문별 보강 한도 3개 설정에서 현재 context eval은 `context_chunk_hit@10`을 `0.5000`에서 `0.6333`으로, `context_nearby_hit@10`을 `0.5667`에서 `0.6667`로 개선했습니다.

## 프로젝트 구조

```text
SortPaper/
+-- app.py
+-- app_sidebar.py
+-- app_ui.py
+-- app_pipeline.py
+-- app_utils.py
+-- app_config.py
+-- src/
|   +-- agent/
|   +-- graph/
|   +-- judge/
|   +-- parsers/
|   |   +-- table/
|   |   +-- layout_chunk.py
|   +-- store/
|       +-- qdrant_store.py
|       +-- chunk_storage.py
+-- tests/
+-- data/
```

## 테스트

```bash
pytest -q
```

## 쿼리 재작성과 멀티 쿼리 리콜

SortPaper에는 evidence chunk 검색을 개선하기 위한 실험적 쿼리 재작성 및 멀티 쿼리 리콜 경로가 포함되어 있습니다.

- 쿼리 재작성은 DeepSeek V4 Flash를 사용해 중국어, 한국어, 영어 또는 구어체 질문을 짧은 영어 과학 검색 query로 정규화합니다.
- 멀티 쿼리 리콜은 원본 query, 정규화 query, 소수의 variants를 함께 검색한 뒤 결과를 병합합니다.
- 현재 병합 전략은 원본 query의 상위 결과와 raw tail 후보를 우선하며, variants는 여러 route에서 일치하거나 같은 앵커 논문에 속할 때 주로 하위 구간을 보충합니다.
- 수동 검색과 Agent 검색 UI에서 명시적으로 켤 수 있지만, 현재 기본값은 꺼짐입니다.

smoke20 평가 기준으로 보호식 멀티 쿼리 리콜은 lexical baseline을 더 이상 악화시키지 않지만, 안정적인 향상은 아직 입증되지 않았고 지연 시간은 증가합니다.

```text
lexical baseline smoke20:
chunk_hit@10 = 0.4545
nearby_chunk_hit@10 = 0.5455
elapsed_ms_p50 = 713ms

multi-query protected smoke20:
chunk_hit@10 = 0.4545
nearby_chunk_hit@10 = 0.5455
elapsed_ms_p50 = 3357ms
```

평가 명령:

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 --strategy standard --lexical-backfill --multi-query --out reports/retrieval_eval_multi_query_lexical60_top10.json
```

자세한 내용은 `evals/QUERY_REWRITE.md`를 참고하세요.
