from utils import *

class Route(BaseModel):
    step: Literal["Healthcare", "E_commurce", "Tellicom","Bank_finance","human_feedback"] = Field(
        None, description="The next step in the routing process"
    )