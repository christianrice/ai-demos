from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.utils.function_calling import convert_to_openai_function
from tools.knowledge_management_tool import tool as knowledge_updater_tool

SYSTEM_PROMPT = """
{opener}

{instructions}

{chain_of_thought}

Here are the existing bits of information that we have about the family:
```
{memories}
```

Call the right tools to save the information. If you identiy multiple pieces of information, make sure you batch all the tool calls simultaneously. You only have one chance to call tools.

{closer}

Analyze the following message:
"""

# Get the prompt to use - you can modify this!
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
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
agent_tools = [knowledge_updater_tool]
tools = [convert_to_openai_function(t) for t in agent_tools]

generate_prompt_output_runnable = prompt | llm.bind_tools(tools)
