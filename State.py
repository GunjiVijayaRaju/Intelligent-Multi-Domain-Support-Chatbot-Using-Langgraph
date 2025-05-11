from typing_extensions import TypedDict
from typing import Annotated, List
import operator
from utils import *

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


# Worker state
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

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