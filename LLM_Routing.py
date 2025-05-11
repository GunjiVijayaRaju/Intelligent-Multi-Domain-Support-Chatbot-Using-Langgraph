from State import State
from LLM import llm
from Routing import Route
from utils import *

router = llm.with_structured_output(Route)

def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    # Run the augmented LLM with structured output to serve as routing logic
    decision = router.invoke(
        [
            SystemMessage(
                        content="""Given the following user query, first check if it mentions specific domain-related keywords (such as 'bank', 'ecommerce', 'tellicom', 'healthcare'). If the query mentions any of these keywords, classify it into the corresponding domain: Bank_Finance, Telecom, E_Commerce, or Healthcare. If no keywords are mentioned or it’s unclear, return 'human_feedback'."""
            ),
            HumanMessage(content=state['input']),
        ]
    )
    print("Decision llm============>",decision.step)
    return {"decision": decision.step}

def decision_by_feedback_agent(state:State) :

    if state["feedback_by_user"] == "solved":
        # decision_by_feedback = "Final_report"
        return "Final_report"
    elif state["feedback_by_user"] ==  "not solved":
        # decision_by_feedback = "Provide_solution_based_on_feedback"
        return "Provide_solution_based_on_feedback"
    elif state["feedback_by_user"] == "person_support_needed":
        # decision_by_feedback = "person_support_needed"   
        return "person_support_needed"

def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "Bank_finance":
        return "Bank_finance_agent"
    elif state["decision"] == "Tellicom":
        return "Tellicom_agent"
    elif state["decision"] == "E_commerce":
        return "E_commurce_agent"
    elif state["decision"] == "Healthcare":
        return "Healtcare_agent"
    elif state["decision"] == "human_feedback":
        return "human_feedback"
    
def should_continue(state: State):
    if state["decision"] == "Bank_finance":
        return "Bank_finance_agent"
    elif state["decision"] == "Tellicom":
        return "Tellicom_agent"
    elif state["decision"] == "E_commerce":
        return "E_commurce_agent"
    elif state["decision"] == "Healthcare":
        return "Healtcare_agent"
