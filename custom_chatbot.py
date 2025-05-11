from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List
import operator
from langchain_groq import ChatGroq
from langgraph.types import interrupt, Command


groq_api_key="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"

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )


class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

from langgraph.constants import Send



# Worker state
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


# Augment the LLM with schema for structured output
planner = llm.with_structured_output(Sections)


# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["Healthcare", "E_commurce", "Tellicom","Bank_finance","human_feedback"] = Field(
        None, description="The next step in the routing process"
    )


# Augment the LLM with schema for structured output
router = llm.with_structured_output(Route)

count = 1
# State
class State(TypedDict):
    input: str
    decision: str
    output: str
    some_text: str
    answers_givien_by_user:str
    solution_providing_output:str
    feedback_by_user:str
    # decision_by_feedback:str
    support_person_details:str
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel
    final_report: str  # Final report



# Nodes
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

# Nodes
def orchestrator(state: State):
    """Orchestrator that generates a plan for the report."""
    report_sections = planner.invoke([
        SystemMessage(content="Generate a structured report based on the given details."),
        HumanMessage(content=f"User's Query: {state['input']}\nClarification: {state['output']}\nUser's Responses: {state['answers_givien_by_user']}\nSolution: {state['solution_providing_output']}")
    ])
    return {"sections": report_sections.sections}


def llm_call(state: WorkerState):
    """Worker writes a section of the report"""

    # Generate section
    section = llm.invoke(
        [
            SystemMessage(
                content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
            ),
            HumanMessage(
                content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
            ),
        ]
    )

    # Write the updated section to completed sections
    return {"completed_sections": [section.content]}


def synthesizer(state: State):
    """Synthesize full report from sections"""

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}


# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off section writing in parallel via Send() API
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


def person_support_needed(state:State):

    create_tables()
    ticket_id = add_ticket("UserX", state["decision"] , state["input"])
    print("PERSON DETAILS TICKETS ---------------->",assign_ticket(ticket_id))
    return {"support_person_details":assign_ticket(ticket_id)}


def Provide_solution_based_on_feedback(state:State):

    result = llm.invoke(f"Based on the user's initial question: {state['input']}, and the solution provided: {state['solution_providing_output']}, now consider the user's feedback: {state['feedback_by_user']}. Using this information, generate a refined solution that directly addresses the user's feedback, ensuring clarity, completeness, and relevance.")

    return {"solution_providing_output":result.content}
    
def Final_report(state:State):

    result = llm.invoke(f"""Generate a comprehensive report or documentation based on the user's initial query and the clarification process. Ensure that the document is structured in a way that is easy for the user to understand, using clear explanations and simple language.
        User's Initial Query: {state['input']}
    Clarification Question Asked: {state['output']}
    User's Responses to Clarification Questions: {state['answers_givien_by_user']}
    Solution Provided Based on the Clarification: {state['solution_providing_output']}
    Using the above information, create a well-structured document that:

    Clearly outlines the problem statement based on the initial query.
    Summarizes the clarification process, including the key questions asked and the user’s responses.
    Presents a detailed and well-explained solution based on the clarification.
    Provides additional insights, best practices, or recommendations if relevant.
    Ensure that the document is clear, user-friendly, and easy to understand. Use simple language, avoid complex jargon, and structure the content in a way that guides the user step by step. The output should be well-organized, concise, and suitable for documentation purposes. No extra information—strictly focus on the given details while maintaining clarity and user comprehension. """)

    return {"support_person_details":result.content}


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

    value = interrupt("Enter the solution for the question ?")
    if value:
        return {"answers_givien_by_user":value}
    

def solution_providing_agent(state:State):

    """Based on the user give answers for the question you need to provide the sloution for that"""

    result = llm.invoke(f"Based on the answers given by user for clarification question you need to provide a clear solution to user initial Query given by user :{state['input']}, Clarification question asked by you {state['output']} and Answers given by user for thiose clarification question{state['answers_givien_by_user']} finally give me only solution that's it clearly with more explanation ")
    
    return {"solution_providing_output": result.content}


def feedback_by_user_agent(state:State):
    
    value_1 = interrupt("Enter the feedback wether it is solved or not solved or need person support ?")
  
    result = llm.invoke(f"Try to understand the user's intention behind their feedback. If the user conveys that their problem is resolved using words like 'solved', 'fixed', or similar, return only 'solved'. If the user indicates that the issue is still present or unresolved, return only 'not solved'. If the user is asking for further assistance, guidance, or requesting customer support, return only 'person_support_needed'. Do not provide any additional information or explanations—strictly return one of these three values. Feedback by user: {value_1}")
    print("Result given by llm====================>",value_1,result.content)
    
    
    return {"feedback_by_user": str(result.content).lower()}
    
def decision_by_feedback_agent(state:State) :

    if state["feedback_by_user"] == "solved":
        return "Final_report"
    elif state["feedback_by_user"] ==  "not solved":
        return "Provide_solution_based_on_feedback"
    elif state["feedback_by_user"] == "person_support_needed":  
        return "person_support_needed"
   


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


# Conditional edge function to route to the appropriate node
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


import sqlite3
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import os
from langchain_groq import ChatGroq

groq_api_key="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"

# os.environ['GEMINI_API_KEY']="AIzaSyBpB-NLQ50R0ftU9B-cNVEPzS0pY96OtMY"

model = SentenceTransformer("all-MiniLM-L6-v2")



# --- DATABASE SETUP ---
def create_tables():
    conn = sqlite3.connect("support.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        field TEXT,
        skills TEXT,
        completed_tasks TEXT,
        open_tickets INTEGER DEFAULT 0,
        availability TEXT
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT,
        field TEXT,
        issue_description TEXT,
        assigned_to INTEGER,
        status TEXT,
        created_at TEXT
    )""")
    conn.commit()
    conn.close()

# --- ADD SUPPORT PERSON ---
def add_person(name, email, field, skills, completed_tasks, availability="yes"):
    conn = sqlite3.connect("support.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO persons (name, email, field, skills, completed_tasks, open_tickets, availability)
        VALUES (?, ?, ?, ?, ?, 0, ?)""",
        (name, email, field, skills, completed_tasks, availability))
    conn.commit()
    conn.close()

# --- ADD SUPPORT TICKET ---
def add_ticket(user_name, field, issue_description):
    conn = sqlite3.connect("support.db")
    cursor = conn.cursor()
    created_at = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO tickets (user_name, field, issue_description, assigned_to, status, created_at)
        VALUES (?, ?, ?, NULL, 'open', ?)""",
        (user_name, field, issue_description, created_at))
    ticket_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return ticket_id

# --- ENRICH TICKET VIA LLM ---
def enrich_ticket_with_tags(description):
    prompt = f"""
You are a support assistant. Normalize the user issue and extract support skill tags.
Return JSON in this format only return this strictly i am telling don't inlcude any other extra matter or details i need onlt this details that's it:
{{
  "normalized_issue": "...",
  "tags": ["tag1", "tag2", ...]
}}

User Issue: {description}
    """
        
    model=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")
    res = model.invoke(prompt)
    content=res.content
    print("Resukt---------------------->",content)  
    import json
    return json.loads(content)
    # except Exception as e:
        # print("❌ LLM tagging failed:", e)
        # return {"normalized_issue": description, "tags": []}

# --- ASSIGN TICKET ---
def assign_ticket(ticket_id):
    conn = sqlite3.connect("support.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    ticket = cursor.fetchone()
    if not ticket:
        # print("❌ Ticket not found.")
        return "❌ Ticket not found."

    _, user_name, field, issue_description, _, _, _ = ticket
    enriched = enrich_ticket_with_tags(issue_description)
    issue_tags = enriched['tags']
    summary = enriched['normalized_issue']

    print(f"\n🎫 Ticket ID: {ticket_id}")
    print(f"   → Tags: {issue_tags}")
    print(f"   → Normalized: {summary}")

    # Get available persons
    cursor.execute("""
        SELECT id, name, email, skills, completed_tasks, open_tickets 
        FROM persons 
        WHERE field = ? AND availability = 'yes'
    """, (field,))
    available_persons = cursor.fetchall()

    if not available_persons:
        # print("❌ No available agents.")
        return "❌ No available agents."

    # Build embeddings for candidates
    index = faiss.IndexFlatL2(384)
    person_map = {}

    for idx, person in enumerate(available_persons):
        id_, name, email, skills, completed_tasks, open_tickets = person
        combined_text = f"{skills}, {completed_tasks}"
        embedding = model.encode(combined_text)
        index.add(np.array([embedding], dtype=np.float32))
        person_map[idx] = {
            "id": id_, "name": name, "email": email,
            "open_tickets": open_tickets
        }

    query_embedding = model.encode(", ".join(issue_tags) or summary)
    k = min(5, len(person_map))
    D, I = index.search(np.array([query_embedding], dtype=np.float32), k)

    best_candidate = None
    best_score = float("inf")
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        person = person_map[idx]
        adjusted_score = D[0][rank] + 0.5 * person['open_tickets']

        print(f"\n🔍 Candidate {rank + 1}:")
        print(f"   → Name: {person['name']}, Email: {person['email']}")
        print(f"   → Tickets: {person['open_tickets']}, Distance: {D[0][rank]:.2f}, Score: {adjusted_score:.2f}")

        if adjusted_score < best_score:
            best_score = adjusted_score
            best_candidate = person

    if best_candidate:
        # print(f"\n✅ Assigned To: {best_candidate['name']} and you can contact him through {best_candidate['email']}")
        return f"\n✅ Assigned To: {best_candidate['name']} and you can contact him through {best_candidate['email']}"
        cursor.execute("UPDATE tickets SET assigned_to = ? WHERE id = ?", (best_candidate['id'], ticket_id))
        cursor.execute("UPDATE persons SET open_tickets = open_tickets + 1 WHERE id = ?", (best_candidate['id'],))
        conn.commit()
    else:
        # print("❌ No suitable agent found.")
        return "No suitable agent found."

    conn.close()




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
