from utils import *
from orchestrator import orchestrator,llm_call,synthesizer,assign_workers
from State import State
from Agents import Healtcare_agent,Tellicom_agent,E_commurce_agent,Bank_finance_agent,solution_providing_agent
from Human_feedback_operations import human_node, user_giving_clification_on_question,feedback_by_user_agent
from LLM_Routing import should_continue, llm_call_router,route_decision,decision_by_feedback_agent
from person_support_needed import Provide_solution_based_on_feedback,person_support_needed
from langgraph.checkpoint.memory import MemorySaver


# POSTGRES_URI= "postgresql://postgres:[YOUR-PASSWORD]@db.qibwthjmjsghjhcgbqaq.supabase.co:5432/postgres"
# Build workflow
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("Tellicom_agent", Tellicom_agent)
router_builder.add_node("E_commurce_agent", E_commurce_agent)
router_builder.add_node("Healtcare_agent", Healtcare_agent)
router_builder.add_node("Bank_finance_agent", Bank_finance_agent)
router_builder.add_node("llm_call_router", llm_call_router)
router_builder.add_node("human_feedback", human_node)
router_builder.add_node("user_giving_clification_on_question",user_giving_clification_on_question)
router_builder.add_node("solution_providing_agent",solution_providing_agent)
router_builder.add_node("feedback_by_user_agent",feedback_by_user_agent)
# router_builder.add_node("decision_by_feedback_agent",decision_by_feedback_agent)
# router_builder.add_node("Final_report",Final_report)
router_builder.add_node("Final_report", orchestrator)
router_builder.add_node("llm_call", llm_call)
router_builder.add_node("synthesizer", synthesizer)

router_builder.add_node("Provide_solution_based_on_feedback",Provide_solution_based_on_feedback)
router_builder.add_node("person_support_needed",person_support_needed)


# Add edges to connect nodes
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "Bank_finance_agent": "Bank_finance_agent",
        "Tellicom_agent": "Tellicom_agent",
        "E_commurce_agent": "E_commurce_agent",
        "Healtcare_agent":"Healtcare_agent",
        "human_feedback":"human_feedback"
    },
)
router_builder.add_edge("Bank_finance_agent", "user_giving_clification_on_question")
router_builder.add_edge("Tellicom_agent", "user_giving_clification_on_question")
router_builder.add_edge("E_commurce_agent","user_giving_clification_on_question")
router_builder.add_edge("Healtcare_agent","user_giving_clification_on_question")


router_builder.add_conditional_edges(
        "human_feedback",
        should_continue,
    {  # Name returned by route_decision : Name of next node to visit
        "Bank_finance_agent": "Bank_finance_agent",
        "Tellicom_agent": "Tellicom_agent",
        "E_commurce_agent": "E_commurce_agent",
        "Healtcare_agent":"Healtcare_agent"
    },
    )
router_builder.add_edge("user_giving_clification_on_question","solution_providing_agent")
router_builder.add_edge("solution_providing_agent","feedback_by_user_agent")
# router_builder.add_edge("feedback_by_user_agent","decision_by_feedback_agent")
router_builder.add_edge("person_support_needed",END)
router_builder.add_edge("Provide_solution_based_on_feedback","feedback_by_user_agent")
# router_builder.add_edge("Final_report",END)
router_builder.add_conditional_edges(
    "Final_report", assign_workers, ["llm_call"]
)
router_builder.add_edge("llm_call", "synthesizer")
router_builder.add_edge("synthesizer", END)


router_builder.add_conditional_edges(
    "feedback_by_user_agent",
    decision_by_feedback_agent,
    {
        "person_support_needed":"person_support_needed",
        "Provide_solution_based_on_feedback":"Provide_solution_based_on_feedback",
        "Final_report":"Final_report"
    },
    )


checkpointer = MemorySaver()  
router_workflow = router_builder.compile(checkpointer=checkpointer)
