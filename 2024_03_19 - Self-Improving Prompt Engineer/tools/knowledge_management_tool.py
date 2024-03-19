from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from enum import Enum
from typing import Optional


class Category(str, Enum):
    Food_Allergy = "Allergy"
    Food_Like = "Like"
    Food_Dislike = "Dislike"
    Family_Attribute = "Attribute"


class Action(str, Enum):
    Create = "Create"
    Update = "Update"
    Delete = "Delete"


class AddKnowledge(BaseModel):
    knowledge: str = Field(
        ...,
        description="Knowledge",
    )
    knowledge_old: Optional[str] = Field(
        None,
        description="If updating or deleting record, the complete, exact phrase that needs to be modified",
    )
    category: Category = Field(
        ...,
        description="Category",
    )
    action: Action = Field(
        ...,
        description="Action",
    )


def handle_action(
    knowledge: str,
    category: str,
    action: str,
    knowledge_old: str = "",
) -> dict:
    print("Handling Knowledge: ", knowledge, knowledge_old, category, action)


tool_name = "Knowledge_Modifier"

tool = StructuredTool.from_function(
    func=handle_action,
    name=tool_name,
    description="Add, update, or delete a bit of knowledge",
    args_schema=AddKnowledge,
)
