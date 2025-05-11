from State import State
from LLM import llm

def Bank_finance_agent(state: State):
    """Based on the input just asks some clarification question on it"""

    result = llm.invoke(f"Based on the input just asks some clarification question on it{state['input']}")
    return {"output": result.content}


def Healtcare_agent(state: State):
    """Based on the input just asks some clarification question on it"""

    result = llm.invoke(f"Based on the input just asks some clarification question on it{state['input']}")
    return {"output": result.content}


def E_commurce_agent(state: State):
    """Based on the input just asks some clarification question on it"""

    result = llm.invoke(f"Based on the input just asks some clarification question on it{state['input']}")
    return {"output": result.content}

def Tellicom_agent(state: State):
    """Based on the input just asks some clarification question on it"""

    result = llm.invoke(f"Based on the input just asks some clarification question on it{state['input']}")
    return {"output": result.content}

def solution_providing_agent(state:State):

    """Based on the user give answers for the question you need to provide the sloution for that"""

    print(f"input : {state['input']}")

    result = llm.invoke(f"Based on the answers given by user for clarification question you need to provide a clear solution to user initial Query given by user :{state['input']}, Clarification question asked by you {state['output']} and Answers given by user for thiose clarification question{state['answers_givien_by_user']} finally give me only solution that's it clearly with more explanation ")

    
    
    return {"solution_providing_output": result.content}