from typing import TypedDict, Annotated, Sequence
import operator
import random
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from agents.memory_extractor import (
    memory_extractor_runnable,
    memory_extractor_with_feedback_runnable,
)
from agents.memory_reviewer import memory_reviewer_runnable
from agents.action_assigner import action_assigner_runnable
from agents.category_assigner import category_assigner_runnable
from pydantic.v1 import BaseModel
from typing import List, Union


class Memory(BaseModel):
    knowledge: str


class MemoryWithAction(BaseModel):
    knowledge: str
    action: str
    old_memory: str


class MemoryComplete(BaseModel):
    knowledge: str
    action: str
    category: str
    old_memory: str


class AgentState(TypedDict):
    # The list of previous messages in the conversation
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The snippet of conversation to analyze
    original_conversation: Sequence[BaseMessage]
    # The memory analysis chain
    memory_analysis: Annotated[Sequence[BaseMessage], operator.add]
    # The list of memories
    memories: List[Union[Memory, MemoryWithAction, MemoryComplete]]
    # The list of existing memories
    existing_memories: List[Memory]


def should_retry_memory_extractor(state):
    last_message = state["memory_analysis"][-1]
    # If there are no tool calls, then we finish
    if (
        last_message.content == "AI analysis is perfect."
        or len(state["memory_analysis"]) > 6
    ):
        return "continue"
    # Otherwise, we continue
    else:
        return "retry"


def call_memory_extractor(state):
    if state["memory_analysis"]:
        # Run the memory extractor runnable
        input = {
            "messages": state["original_conversation"],
            "previous_memory_analysis": state["memory_analysis"],
        }
        extracted_memories = memory_extractor_with_feedback_runnable.invoke(input)
    else:
        # Run the memory extractor runnable
        input = {"messages": state["original_conversation"]}
        extracted_memories = memory_extractor_runnable.invoke(input)

    # Extract the memories from the output
    memories = [memory["knowledge"] for memory in extracted_memories.get("memories")]

    # Create an extraction message
    extraction_iteration = (
        1
        if state.get("memory_analysis") is None
        else len(state["memory_analysis"]) // 2 + 1
    )
    extraction_message = f"Memory extraction results from attempt #{extraction_iteration}: {', '.join(repr(memory) for memory in memories)}"

    # Overwrite the current memories; save the message in the chain history of memory attempts; and append this message to the total message log as well
    return {
        "messages": [HumanMessage(content=extraction_message)],
        "memory_analysis": [HumanMessage(content=extraction_message)],
        "memories": memories,
    }


def call_memory_reviewer(state):
    # Randomly reorder the memory list
    random.shuffle(state["memories"])

    # Prepare data for the reviewer runnable
    ai_analysis = "\n".join(
        f"Memory {i+1} of {len(state['memories'])}: {memory}"
        for i, memory in enumerate(state["memories"])
    )
    inputs_to_review = {
        "messages": state["original_conversation"],
        "ai_analysis": ai_analysis,
    }
    # Run the reviewer runnable
    review_results = memory_reviewer_runnable.invoke(inputs_to_review)
    # Convert the messages to dictionaries
    review_results_dict = [dict(message) for message in review_results.content]
    # Find the first tool call
    first_tool_call = next(
        (
            tool_call
            for tool_call in review_results_dict
            if tool_call["type"] == "tool_use"
        ),
        None,
    )

    # Either return the criticism or state that the message was perfect
    new_message = HumanMessage(content="AI analysis is perfect.")
    if first_tool_call is not None and not first_tool_call["input"]["is_perfect"]:
        # Create an extraction message
        extraction_iteration = (
            1
            if state.get("memory_analysis") is None
            else len(state["memory_analysis"]) // 2 + 1
        )
        extraction_message = f"Memory extraction analysis for attempt #{extraction_iteration}: {first_tool_call['input']['criticism']}"
        new_message = HumanMessage(content=extraction_message)

    return {"messages": [new_message], "memory_analysis": [new_message]}


def call_action_assigner(state):
    inputs = {
        "existing_memories": state["existing_memories"],
        "new_memories": state["memories"],
    }

    memories_with_actions = action_assigner_runnable.invoke(inputs)
    new_message = f"Added actions: {memories_with_actions}"

    return {"messages": [new_message], "memories": memories_with_actions["memories"]}


def call_category_assigner(state):
    inputs = {"memories": state["memories"]}

    complete_memories = category_assigner_runnable.invoke(inputs)

    new_message = f"Added categories: {complete_memories}"
    return {"messages": [new_message], "memories": complete_memories["memories"]}


# Initialize a new graph
graph = StateGraph(AgentState)

# Define the Nodes we will cycle between
graph.add_node("memory_extractor", call_memory_extractor)
graph.add_node("memory_reviewer", call_memory_reviewer)
graph.add_node("action_assigner", call_action_assigner)
graph.add_node("category_assigner", call_category_assigner)

# Set the Starting Edge
graph.set_entry_point("memory_extractor")

# Define the Conditional Edges
graph.add_conditional_edges(
    "memory_reviewer",
    should_retry_memory_extractor,
    {"retry": "memory_extractor", "continue": "action_assigner"},
)

# Define the Normal Edges that should always be called after another
graph.add_edge("memory_extractor", "memory_reviewer")
graph.add_edge("action_assigner", "category_assigner")
graph.add_edge("category_assigner", END)

# We compile the entire workflow as a runnable
memory_reflection_graph = graph.compile()
