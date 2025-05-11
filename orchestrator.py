from utils import *
from State import State,Sections,WorkerState
from LLM import llm

# class Section(BaseModel):
#     name: str = Field(
#         description="Name for this section of the report.",
#     )
#     description: str = Field(
#         description="Brief overview of the main topics and concepts to be covered in this section.",
#     )


# class Sections(BaseModel):
#     sections: List[Section] = Field(
#         description="Sections of the report.",
#     )


# # Worker state
# class WorkerState(TypedDict):
#     section: Section
#     completed_sections: Annotated[list, operator.add]


# Augment the LLM with schema for structured output
planner = llm.with_structured_output(Sections)

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