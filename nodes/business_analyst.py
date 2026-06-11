# nodes/business_analyst.py

import re
from core.state import State
# from core.llm_config import pandas_agent
import pandas as pd
from core.llm_config import llm, df_store, df_customer, df_market, df_sales, get_message_content

def business_analyst(state: State) -> State:
    mission = state.get("next_step_details", get_message_content(state["messages"][-1]))
    store_code = state.get("store_code", "")

    # 데이터가 로드되지 않았거나 비어 있는 경우 조기 처리
    if df_store.empty:
        state["analysis_result"] = "⚠️ 현재 로드된 데이터셋이 없어 상세한 데이터 수치 분석을 진행할 수 없습니다. 데이터셋(data/full_data.xlsx) 업로드 상태를 확인해 주세요."
        return state

    code_gen_prompt = f"""
당신은 '성동구 외식업 데이터 분석 전문가'입니다.
데이터프레임(df_store, df_sales, df_customer, df_market)을 활용하여 가맹점번호 [{store_code}]에 대한 파이썬 분석 코드를 작성하세요.

[데이터프레임 상세 지도]
1. df_store: 가맹점 마스터 정보 (가맹점구분번호 기준)
2. df_sales: 매출 성과
3. df_customer: 고객 세그먼트 및 비중
4. df_market: 상권 경쟁력

[분석 미션]
{mission}

[수행 규칙]
- 반드시 파이썬 코드만 출력하세요. (마크다운 ```python 코드 블록으로 감싸서 작성)
- 데이터프레임 조인은 '가맹점구분번호'를 기준으로 합니다.
- 최종 분석 결과물은 반드시 `analysis_output`이라는 변수에 문자열로 저장하세요.
- 외부 라이브러리(os, sys 등)는 절대 사용하지 마세요.
"""

    try:
        raw_response = get_message_content(llm.invoke(code_gen_prompt))
        
        # 정규표현식을 이용해 마크다운 파이썬 코드 블록만 정확하게 추출
        match = re.search(r"```python\s*(.*?)\s*```", raw_response, re.DOTALL)
        code = match.group(1).strip() if match else raw_response.strip()

        # 위협 차단
        forbidden = ['os.', 'sys.', 'subprocess', 'import ', 'eval(', 'exec(']
        if any(word in code for word in forbidden):
            raise PermissionError("보안 위협이 탐지된 코드는 실행할 수 없습니다.")
        
        # 환경 격리 및 df1~df4 앨리어스 지원
        local_vars = {
            "df_store": df_store, "df_sales": df_sales, "df_customer": df_customer, "df_market": df_market,
            "df1": df_store, "df2": df_sales, "df3": df_customer, "df4": df_market,
            "pd": pd
        }
        exec(code, {"__builtins__": {}}, local_vars)

        extracted_data = local_vars.get("analysis_output", "분석 데이터 추출 실패")
        report_prompt = f"""
당신은 데이터 분석 보고서 작성가입니다. 아래의 [분석 데이터]를 바탕으로 보고서를 작성하세요.

[분석 데이터]
{extracted_data}

[보고 형식]
- 요약: 한 줄 요약
- 상세 분석: 수치 기반의 관찰 결과
- 특이사항: 동일 상권/업종 대비 눈에 띄는 차이점 또는 시사점

규칙: 마케팅 전략은 제안하지 말고 데이터 기반의 현상 진단만 제공합니다.
"""
        final_report = get_message_content(llm.invoke(report_prompt))
        state["analysis_result"] = final_report
    except Exception as e:
        state["analysis_result"] = f"데이터 분석 중 오류 발생: {str(e)}"
    
    return state