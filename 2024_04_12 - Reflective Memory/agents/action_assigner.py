from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from typing import List
from pydantic.v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

system_prompt_initial = """
Your job is to determine what to do with a list of memories extracted from a chat history.

You will compare a list of new memories to an existing list. You will need to determine if the new memories are new information, updates to old information, or if they should result in deleting information that is not correct.

The options for actions are as follows:

1. CREATE a new piece of information
2. UPDATE an existing piece of information
3. DELETE an existing piece of information

When you receive a message, you perform a sequence of steps consisting of:

1. Internally analyze and understand all the existing memories
2. Internally compare the new memories to the existing list
3. For each piece of new knowledge, determine if this is new knowledge, an update to old knowledge that now needs to change, or should result in deleting information that is not correct. It's possible that a food you previously wrote as a dislike might now be a like, or that a family member who previously liked a food now dislikes it - those examples would require an update.

Here are the existing bits of information that we have about the family.

```
{existing_memories}
```

Return the information fragments in the following format:

{format_instructions}

I will tip you $20 if you are perfect, and I will fine you $40 if you miss any important information or change any incorrect information.

Take a deep breath, think step by step, and then analyze the following memories:

{new_memories}
"""


class Memory(BaseModel):
    knowledge: str = Field(description="If this failed, provide your explanation")
    action: str = Field(
        description="The action to take on this memory: either CREATE, UPDATE, or DELETE"
    )
    old_memory: str = Field(
        description="If updating or deleting, include the original text from the old memory to update or delete"
    )


class Memories(BaseModel):
    memories: List[Memory] = Field(description="List of memories")


parser = JsonOutputParser(pydantic_object=Memories)

# Get the prompt to use - you can modify this!
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt_initial),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Choose the LLM that will drive the agent
llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    # model="gpt-4-0125-preview",
    streaming=True,
    temperature=0.0,
)

action_assigner_runnable = prompt | llm | parser
