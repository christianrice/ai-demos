from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)

system_prompt_initial = """
Your job is to assign a category to each memory in a list of new memories.

The options for categories are as follows:

1. ALLERGY. This is only assigned when it is a serious allergy, and not just a like, a dislike, or a sensitivity. Allergies are potentially life-threatening.
2. LIKE. This is assigned to foods the family likes (e.g. likes pasta) - These are important to know because they can help you plan meals, but are not life-threatening.
3. DISLIKE. This is assigned to foods the family dislikes (e.g. doesn't eat mussels or rarely eats beef) - These are important to know because they can help you plan meals, but are not life-threatening.
4. ATTRIBUTE. These are immutable facts about the family that may impact weekly meal planning (e.g. lives in Austin; has a husband and 2 children; has a garden; likes big lunches, etc.)

When you receive a message, you perform a sequence of steps consisting of:

1. Internally analyze and understand each of the memories in the list provided
2. Internally assign a category to each memory
3. Return the list of memories with the new categories you have determined. Do not change any of the original text of the memories, only add a category assignation.

Here are the memories for you to assign categories to:

```
{memories}
```

Return the information fragments in the following format:

{format_instructions}

I will tip you $20 if you are perfect, and I will fine you $40 if you miss any important information or change any incorrect information.

Take a deep breath, and think step by step.
"""

class Memory(BaseModel):
    memory: str = Field(description="If this failed, provide your explanation")
    action: str = Field(description="The action to take on this memory: either CREATE, UPDATE, or DELETE")
    category: str = Field(description="The category for this action: either ALLERGY, LIKE, DISLIE, OR ATTRIBUTE")
    old_memory: str = Field(description="If updating or deleting, include the original text from the old memory to update or delete")

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

category_assigner_runnable = prompt | llm | parser