# core/llm_config.py
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# =========================
# Loading API Key
# =========================
load_dotenv()
API_KEY = os.getenv("API_KEY")

# =========================
# LLM / Embeddings
# =========================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", google_api_key=API_KEY
)

# =========================
# Loading Data + Pandas Agent
# =========================
try:
    raw_df = pd.read_excel("data/최종_데이터셋_v2.xlsx")

    df_store = raw_df[['가맹점구분번호', '가맹점주소', '가맹점명', '브랜드구분코드', '가맹점지역', '업종', '상권', '개설일', '폐업일', '브랜드이름', '별점', '카테고리평가']].drop_duplicates()

    df_sales = raw_df[['가맹점구분번호', '기준년월', '가맹점 운영개월수 구간', '매출금액 구간', '매출건수 구간', '객단가 구간', '취소율 구간', '배달매출금액 비율']]

    df_customer = raw_df[['가맹점구분번호', '기준년월', '유니크 고객 수 구간', '남성 20대이하 고객 비중', '남성 30대 고객 비중', '남성 40대 고객 비중', '남성 50대 고객 비중', '남성 60대이상 고객 비중', '여성 20대이하 고객 비중', '여성 30대 고객 비중', '여성 40대 고객 비중', '여성 50대 고객 비중', '여성 60대이상 고객 비중', '재방문 고객 비중', '신규 고객 비중', '거주 이용 고객 비율', '직장 이용 고객 비율', '유동인구 이용 고객 비율']]

    df_market = raw_df[['가맹점구분번호', '기준년월', '동일 업종 매출금액 비율', '동일 업종 매출건수 비율', '동일 업종 내 매출 순위 비율', '동일 상권 내 매출 순위 비율', '동일 업종 내 해지 가맹점 비중', '동일 상권 내 해지 가맹점 비중']]

    pandas_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=[df_store, df_sales, df_customer, df_market], # df1, df2, df3, df4로 인식됨
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )
    print("✅ 데이터 로드 및 Pandas Agent 초기화 완료!")

except Exception as e:
    print(f"❌ 데이터 로드 실패 (경로 또는 파일명을 확인하세요): {e}")
    pandas_agent = None