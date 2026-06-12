import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import RedisSaver
from langchain_core.messages import HumanMessage


from core.state import State
from nodes.core_orchestrator import core_orchestrator
from nodes.business_analyst import business_analyst
from nodes.marketing_strategist import marketing_strategist

REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_ENDPOINT = os.getenv("REDIS_ENDPOINT")

builder = StateGraph(State)

builder.add_node("core_orchestrator", core_orchestrator)
builder.add_node("business_analyst", business_analyst)
builder.add_node("marketing_strategist", marketing_strategist)
builder.add_node("human_proxy", lambda x: x) # 대기 노드

builder.add_edge(START, "core_orchestrator")

# 조건부 엣지: 지휘관의 판단에 따라 분기
builder.add_conditional_edges(
    "core_orchestrator",
    lambda x: x["next_step"],
    {
        "business_analyst": "business_analyst",
        "human_proxy": "human_proxy",
        "marketing_strategist": "marketing_strategist",
        "finish": "marketing_strategist"
    }
)

# 자식 노드들은 일을 마치면 무조건 다시 지휘관에게 돌아옴 (Hub-and-Spoke)
builder.add_edge("business_analyst", "core_orchestrator")
builder.add_edge("human_proxy", "core_orchestrator")
builder.add_edge("marketing_strategist", END)

def run_loop(agents, thread_config):
    while True:
        current_state = agents.get_state(thread_config)

        # 사장님 답변 대기 중인 경우
        if current_state.next and "human_proxy" in current_state.next:
            print(f"\n💬 [에이전트]: {current_state.values.get('next_step_details')}")
            user_input = input("[사장님 응답]: ")
            if user_input.lower() in ["q", "exit"]: break

            agents.update_state(
                thread_config,
                {"messages": [HumanMessage(content=user_input)]},
                as_node="human_proxy"
            )
            agents.invoke(None, config=thread_config)

        # 결과가 나왔거나 새로 시작하는 경우
        else:
            if current_state.values.get("final_answer"):
                print(f"\n✅ [최종 답변]\n{current_state.values['final_answer']}")

            user_input = input("\n[사장님 질문]: ")
            if user_input.lower() in ["q", "exit"]: break

            # 새 질문 시작 전에 이전 대화 회차의 분석 및 제안 상태 초기화
            agents.update_state(
                thread_config,
                {
                    "analysis_result": "",
                    "marketing_strategy": "",
                    "final_answer": ""
                }
            )

            agents.invoke({"messages": [HumanMessage(content=user_input)]}, config=thread_config)

if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": "boss_01"}}
    print("\n🚀 에이전트 가동 (종료: q)")

    # Redis Saver 환경 변수 검증 및 연결 처리 (실패 시 MemorySaver로 자동 폴백)
    checkpointer = None
    redis_cm = None

    if REDIS_ENDPOINT:
        try:
            if REDIS_PASSWORD:
                connection_string = f"redis://{REDIS_PASSWORD}@{REDIS_ENDPOINT}"
            else:
                connection_string = f"redis://{REDIS_ENDPOINT}"
            
            # RedisSaver 연결 컨텍스트 매니저 수동 진입
            redis_cm = RedisSaver.from_conn_string(connection_string)
            checkpointer = redis_cm.__enter__()
            checkpointer.setup() # RediSearch 인덱스 초기화
            print("✅ RedisSaver 연결 성공! 대화 내역이 레디스에 저장됩니다.")
        except Exception as e:
            print(f"⚠️ Redis 연결 실패: {e}. MemorySaver로 폴백합니다.")
            if redis_cm is not None:
                try:
                    redis_cm.__exit__(None, None, None)
                except:
                    pass
            checkpointer = MemorySaver()
    else:
        print("ℹ️ Redis 환경변수(REDIS_ENDPOINT)가 설정되지 않았습니다. MemorySaver를 사용합니다.")
        checkpointer = MemorySaver()

    # 에이전트 컴파일 및 실행 루프 시작 (노드 실행 에러가 Redis 오류로 오보되는 것을 방지하기 위해 분리)
    try:
        agents = builder.compile(checkpointer=checkpointer, interrupt_before=["human_proxy"])
        run_loop(agents, thread_config)
    finally:
        # 정상 가동되던 Redis 커넥션 종료 처리
        if redis_cm is not None and not isinstance(checkpointer, MemorySaver):
            try:
                redis_cm.__exit__(None, None, None)
            except:
                pass