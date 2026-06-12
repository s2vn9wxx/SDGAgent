# nodes/marketing_strategist.py
from core.state import State
from core.llm_config import llm, get_message_content
from core.retriever import rag_chain

def marketing_strategist(state: State) -> State:
    analysis = state.get("analysis_result", "")
    
    # 최근 5개의 대화 내역을 취합하여 검색 쿼리 재작성용 컨텍스트로 활용
    history_msgs = []
    for msg in state["messages"][-5:]:
        # 메시지 타입에 따라 사장님 / 에이전트 구분 표시
        role = "사장님" if msg.type == "human" else "에이전트"
        history_msgs.append(f"{role}: {get_message_content(msg)}")
    history_str = "\n".join(history_msgs)

    # 1. RAG용 검색 쿼리 고도화 (Query Rewrite)
    rewrite_prompt = f"""
당신은 성동구 식당/카페 사장님들을 위한 '마케팅 검색 쿼리 작성기'입니다.
아래의 [대화 내역]과 [데이터 분석 결과]를 바탕으로, RAG 벡터 데이터베이스에서 가장 효과적인 마케팅 솔루션 및 성공 사례를 찾기 위한 최적의 검색 쿼리를 1개 작성하세요.

[대화 내역]
{history_str}

[데이터 분석 결과]
{analysis}

[규칙]
- 오직 검색용 쿼리(문장 또는 키워드 나열)만 출력하세요. 다른 설명이나 앞뒤 마크다운(```)은 절대 추가하지 마세요.
"""
    rewrite_res = llm.invoke(rewrite_prompt)
    query = get_message_content(rewrite_res).strip()
    
    # 2. 고도화된 쿼리로 RAG 검색 및 전략 도출
    strategy, rag_debug, rag_query, rag_refs = rag_chain(query, analysis)
    
    # 3. 결과 상태 업데이트 및 최종 포맷 조립
    state["marketing_strategy"] = strategy
    state["final_answer"] = f"""🎯 [데이터 기반 진단 보고서]
{analysis}

💡 [맞춤형 마케팅 전략 제안]
{strategy}

📚 [참고 자료 및 출처]
{rag_refs}"""

    return state
