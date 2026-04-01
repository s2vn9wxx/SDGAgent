# nodes/business_analyst.py

import re
from core.state import State
from core.llm_config import pandas_agent

def business_analyst(state: State) -> State:
    # 전략가로부터 전달받은 구체적인 분석 미션 또는 사용자 질문
    mission = state.get("next_step_details", state["messages"][-1].content)
    store_code = state.get("store_code", "")

    # llm_config.py에서 정의한 df 순서에 맞춘 정밀 가이드
    prompt = f"""
당신은 '성동구 외식업 데이터 분석 전문가'입니다.
주어진 4개의 데이터프레임(df1~df4)을 활용하여 가맹점번호 [{store_code}]에 대한 [분석 미션]을 수행하세요.

[데이터프레임 상세 지도]
1. df1 (가맹점 마스터): 가맹점명, 주소, 상권, 업종, 개설/폐업일, 브랜드이름, 별점, 카테고리평가
2. df2 (매출 성과): 매출금액/건수 구간, 운영개월수, 객단가 구간, 취소율, 배달매출비율
3. df3 (고객 세그먼트): 유니크 고객 수, 성별/연령별 비중(10종), 재방문/신규 비중, 유입지 비중(거주/직장/유동인구)
4. df4 (상권 경쟁력): 동일 업종/상권 내 매출 순위 비율, 동일 업종/상권 내 해지(폐업) 가맹점 비중

[분석 미션]
{mission}

[수행 규칙]
- 데이터프레임 간 조인(Merge)이 필요할 경우 '가맹점구분번호'를 기준으로 수행하세요.
- 모든 분석 결과에는 반드시 구체적인 '수치'나 '구간'을 명시하세요 (예: "매출 순위 상위 10% 이내", "재방문 비중 30% 미만" 등).
- 마케팅 전략(Action Plan)은 절대 제안하지 마세요. 오직 데이터 기반의 '현상 진단'과 '인사이트'만 제공합니다.
- 데이터가 없거나 조회가 안 될 경우 "데이터 부재로 확인 불가"라고 답변하세요.

[보고 형식]
- 요약: 한 줄 요약
- 상세 분석: 수치 기반의 관찰 결과
- 특이사항: 동일 상권/업종 대비 눈에 띄는 차이점 또는 시사점
"""

    try:
        if pandas_agent is None:
            state["analysis_result"] = "데이터베이스 에이전트 초기화 실패"
            return state

        result = pandas_agent.invoke(prompt)
        state["analysis_result"] = result["output"]
    except Exception as e:
        # 파싱 에러 방지용 예외 처리
        error_str = str(e)
        if "Could not parse LLM output" in error_str:
            # Pydantic 파싱 에러 시 원문 텍스트 추출 시도
            match = re.search(r"`(.*?)`", error_str, re.DOTALL)
            state["analysis_result"] = match.group(1).strip() if match else "데이터 분석 결과 파싱 실패"
        else:
            state["analysis_result"] = f"분석 중 오류 발생: {error_str}"
    
    return state