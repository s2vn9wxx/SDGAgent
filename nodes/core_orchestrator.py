# nodes/core_orchestrator.py

import re
import json
from core.state import State
from core.llm_config import llm
from core.retriever import rag_chain

def core_orchestrator(state: State) -> State:
    last_msg = state["messages"][-1].content
    analysis = state.get("analysis_result", "")

    if not state.get("store_code"):
        m = re.search(r"\b[A-Z0-9]{10}\b", last_msg.upper())
        state["store_code"] = m.group(0) if m else ""

    # 데이터가 없으면 분석가에게, 데이터가 있으면 최종 답변 생성 후 종료
    # 추가 정보 필요 시 human_proxy로 보냄
    prompt = f"""
당신은 '성동구 마케팅 수석 전략가'입니다.
사장님 질문: {last_msg}
데이터 상태: {"분석 완료" if analysis else "분석 전"}

[임무]
- 데이터 분석이 필요하면: "business_analyst"
- 사장님께 추가 정보를 물어야 하면: "human_proxy"
- 분석 결과가 충분하여 결론을 내릴 수 있다면: "finish"

반드시 JSON으로만 응답하세요:
{{"decision": "내용", "message": "지시사항 또는 질문"}}
"""
    res = llm.invoke(prompt)
    data = json.loads(res.content.replace("```json", "").replace("```", ""))
    
    state["next_step"] = data["decision"]
    state["next_step_details"] = data["message"]

    # RAG 기반 최종 답변 생성
    if data["decision"] == "finish":
        strategy, *_ = rag_chain(last_msg, analysis)
        state["final_answer"] = f"🎯 [진단]\n{analysis}\n\n💡 [전략]\n{strategy}"
        
    return state