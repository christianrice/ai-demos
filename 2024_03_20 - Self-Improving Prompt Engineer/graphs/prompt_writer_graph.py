import json
import copy
from langchain_core.messages import ToolMessage, FunctionMessage
from langgraph.prebuilt import ToolInvocation
from typing import TypedDict, Sequence, List
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from tools.write_prompt_openai import prompt_engineer_runnable
from tools.write_prompt_anthropic import prompt_engineer_anthropic_runnable
from agents.prompt_engineer_manager import prompt_controller_runnable, tool_executor
from tools.run_eval import process_eval_dataset


class PromptParts(TypedDict):
    opener: str
    instructions: str
    chain_of_thought: str
    closer: str


class PromptModification(TypedDict):
    what_changed: str
    previous_value: str
    new_value: str
    results: str
    decision: str
    accuracy: float


class AgentState(TypedDict):
    # The list of previous messages in the conversation
    messages: Sequence[BaseMessage]
    # The current iteration of the prompt
    prompt: PromptParts
    # Change log
    prompt_change_log: List[PromptModification]
    # Highest accuracy
    highest_accuracy: float


# Define the function that determines whether to continue or not
def should_continue(state):
    last_message = state["messages"][-1]
    # If there are no tool calls, then we finish
    if "tool_calls" not in last_message.additional_kwargs:
        return "end"
    # Otherwise, we continue
    else:
        return "continue"


# Define the function that determines whether the test is complete or not
def should_run_another_test(state):
    if state["highest_accuracy"] < 0.98:
        return "continue"
    else:
        return "end"


# Define the function that calls the prompt controller
def call_prompt_controller(state):
    messages = state["messages"]
    response = prompt_controller_runnable.invoke({"messages": messages})
    return {"messages": messages + [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    temp_prompt_change_log = (
        copy.deepcopy(state["prompt_change_log"])
        if state["prompt_change_log"] is not None
        else []
    )
    # We know the last message involves at least one tool call
    last_message = messages[-1]

    # We loop through all tool calls and append the message to our message log
    for tool_call in last_message.additional_kwargs["tool_calls"]:
        action = ToolInvocation(
            tool=tool_call["function"]["name"],
            tool_input=json.loads(tool_call["function"]["arguments"]),
            id=tool_call["id"],
        )

        # We call the tool_executor and get back a response
        if action.tool == "Prompt_Writer":
            input = copy.deepcopy(state["prompt"])
            input["prompt_history"] = temp_prompt_change_log[-4:]

            use_anthropic = False

            if use_anthropic:
                for _ in range(3):
                    try:
                        prompt_modification = prompt_engineer_anthropic_runnable.invoke(
                            input
                        )
                        break  # If the code runs successfully, exit the loop
                    except Exception as e:
                        print(f"Attempt failed with error: {e}")
                        continue

                what_changed = None
                new_value = None

                for item in prompt_modification["response"]:
                    if "prompt_part" in item:
                        what_changed = item["prompt_part"]
                    elif "new_value" in item:
                        new_value = item["new_value"]
            else:
                for _ in range(3):
                    try:
                        prompt_modification = prompt_engineer_runnable.invoke(input)
                        what_changed = prompt_modification["prompt_part"]
                        new_value = prompt_modification["new_value"]
                        break  # If the code runs successfully, exit the loop
                    except Exception as e:
                        print(f"Attempt failed with error: {e}")
                        continue

            what_changed = what_changed.strip().lower()

            change = {
                "what_changed": what_changed,
                "previous_value": state["prompt"][what_changed],
                "new_value": new_value,
                "results": "",
                "decision": "Discarded change",
                "accuracy": 0.0,
            }

            temp_prompt_change_log.append(change)

            response = "New prompt written."
        else:
            response = tool_executor.invoke(action)

        # We use the response to create a FunctionMessage
        function_message = ToolMessage(
            content=str(response), name=action.tool, tool_call_id=tool_call["id"]
        )

        # Add the function message to the list
        messages.append(function_message)
    return {"messages": messages, "prompt_change_log": temp_prompt_change_log}


def call_tester(state):
    messages = state["messages"]
    prompt = copy.deepcopy(state["prompt"])
    temp_prompt_change_log = copy.deepcopy(state.get("prompt_change_log", []))

    # Set the prompt input to the current prompt with the new change to test
    input = copy.deepcopy(prompt)
    change = state["prompt_change_log"][-1]
    what_changed = change["what_changed"].strip().lower()
    input[what_changed] = change["new_value"]

    print("----------------")
    print("----------------")
    print(f"Testing the prompt with the following revision:")
    print(f"Section changed: {what_changed}")
    print(f"New value:")
    print(f"{change['new_value']}")
    print("----------------")

    # Run the test
    confusion_matrix, accuracy, inaccurate_responses = process_eval_dataset(
        "./data/eval_dataset.jsonl", input
    )
    print(f"Test complete. Accuracy: {accuracy}")
    print(confusion_matrix)

    # Create a message to report the accuracy
    new_message = FunctionMessage(
        content=f"Tested the new prompt. It had an accuracy of {accuracy}",
        name="Tester",
    )
    messages.append(new_message)

    if temp_prompt_change_log:
        # Update the most recent entry with the inaccurate responses
        temp_prompt_change_log[-1]["results"] = inaccurate_responses
        temp_prompt_change_log[-1]["accuracy"] = accuracy
    else:
        # Handle the case where there is no prompt history
        print("No prompt history to update with inaccurate responses.")

    # Update the highest accuracy if necessary
    highest_accuracy = (
        state["highest_accuracy"] if state.get("highest_accuracy") is not None else 0.0
    )
    temp_prompt = copy.deepcopy(state["prompt"])
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        temp_prompt[what_changed] = change["new_value"]
        temp_prompt_change_log[-1]["decision"] = "Accepted change"
        print("Accepted change, here is the new prompt:")
        print(temp_prompt)
        print("\n")
    else:
        print("Rejected change")
        print("\n")

    # Return the updated state with the modified prompt history
    return {
        "messages": messages,
        "prompt": temp_prompt,
        "prompt_change_log": temp_prompt_change_log,
        "highest_accuracy": highest_accuracy,
    }


# Initialize a new graph
graph = StateGraph(AgentState)

# Define the two "Nodes"" we will cycle between
graph.add_node("prompt_controller", call_prompt_controller)
graph.add_node("action", call_tool)
graph.add_node("test", call_tester)

# Define all our Edges

# Set the Starting Edge
graph.set_entry_point("prompt_controller")

# We now add Conditional Edges
graph.add_conditional_edges(
    "prompt_controller",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

graph.add_conditional_edges(
    "test",
    should_run_another_test,
    {
        "continue": "prompt_controller",
        "end": END,
    },
)

# We now add Normal Edges that should always be called after another
graph.add_edge("action", "test")

# We compile the entire workflow as a runnable
app = graph.compile()
