import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import RedisSaver
from langchain_core.messages import HumanMessage

from core.state import State
from nodes.core_orchestrator import core_orchestrator
from nodes.business_analyst import business_analyst

REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_ENDPOINT = os.getenv("REDIS_ENDPOINT")

builder = StateGraph(State)

builder.add_node("core_orchestrator", core_orchestrator)
builder.add_node("business_analyst", business_analyst)
builder.add_node("human_proxy", lambda x: x) # 대기 노드

builder.add_edge(START, "core_orchestrator")

# 조건부 엣지: 지휘관의 판단에 따라 분기
builder.add_conditional_edges(
    "core_orchestrator",
    lambda x: x["next_step"],
    {
        "business_analyst": "business_analyst",
        "human_proxy": "human_proxy",
        "finish": END
    }
)

# 자식 노드들은 일을 마치면 무조건 다시 지휘관에게 돌아옴 (Hub-and-Spoke)
builder.add_edge("business_analyst", "core_orchestrator")
builder.add_edge("human_proxy", "core_orchestrator")

with RedisSaver.from_conn_string("redis://" + REDIS_PASSWORD + "@" + REDIS_ENDPOINT) as checkpointer:
    agents = builder.compile(checkpointer=checkpointer, interrupt_before=["human_proxy"])

if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": "boss_01"}}
    print("\n🚀 에이전트 가동 (종료: q)")

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

            agents.invoke({"messages": [HumanMessage(content=user_input)]}, config=thread_config)