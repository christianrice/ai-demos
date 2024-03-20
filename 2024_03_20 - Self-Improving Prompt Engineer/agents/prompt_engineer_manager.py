# Set up the tools to execute them from the graph
from langgraph.prebuilt import ToolExecutor

from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.utils.function_calling import convert_to_openai_function
from tools.write_prompt_openai import tool_prompt_writer

# Set up the agent's tools
agent_tools = [tool_prompt_writer]

tool_executor = ToolExecutor(agent_tools)

system_prompt_initial = """
Your job is to return an incredible prompt to the user that will help them to get the desired output from the model.

You have access to a prompt writer, and you will then see the results of that prompt after they are run against some tests.

You will not stop until you have hit 10 iterations or an accuracy threshold of 0.98 or higher.

Either request something from the prompt writer (you can only request one writer a time, never request multiple simultaneously), or return DONE to finish the task.
"""

# Get the prompt to use - you can modify this!
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt_initial),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Choose the LLM that will drive the agent
llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    streaming=True,
    temperature=0.0,
)

# Create the tools to bind to the model
tools = [convert_to_openai_function(t) for t in agent_tools]

prompt_controller_runnable = prompt | llm.bind_tools(tools)
