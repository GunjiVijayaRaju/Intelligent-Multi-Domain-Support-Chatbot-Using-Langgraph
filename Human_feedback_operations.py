from State import State
from LLM import llm
from utils import *

def human_node(state: State):
    value = interrupt("Choose the one option bank or ecommerce or tellicom or healthcare: ?")

    if 'bank' in value:
        decision = "Bank_finance"
    elif 'ecommerce' in value:
        decision = "E_commurce"
    elif 'tellicom' in value:
        decision = "Tellicom"
    elif 'healthcare' in value:
        decision = "Healthcare"

    return {
        "some_text": value,
        "decision": decision  # Update the decision after human feedback
    }

def user_giving_clification_on_question(state:State):

    # value="answers for questions"
    value = interrupt("Enter the solution for the question ?")

    if value:
        return {"answers_givien_by_user":value}

def feedback_by_user_agent(state:State):
    
    value_1 = interrupt("Enter the feedback wether it is solved or not solved or need person support ?")
  
    result = llm.invoke(f"Try to understand the user's intention behind their feedback. If the user conveys that their problem is resolved using words like 'solved', 'fixed', or similar, return only 'solved'. If the user indicates that the issue is still present or unresolved, return only 'not solved'. If the user is asking for further assistance, guidance, or requesting customer support, return only 'person_support_needed'. Do not provide any additional information or explanations—strictly return one of these three values. Feedback by user: {value_1}")
    print("Result given by llm====================>",value_1,result.content)
    
    
    return {"feedback_by_user": str(result.content).lower()}