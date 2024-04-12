from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from pydantic.v1 import BaseModel, Field
from typing import List
from langchain_core.output_parsers import JsonOutputParser

system_prompt_initial = """
Your job is to assess a brief chat history in order to determine if the conversation contains any details about a family's dining habits.

You will extract details as discrete pieces of information that could be used in combination to define a rich persona (e.g. I like pasta; I am allergic to shellfish; I don't eat mussels; I live in Austin, Texas; I have a husband and 2 children aged 5 and 7).

Every time you receive a message, you will evaluate if it has any information worth recording in the knowledge base.

A message may contain multiple pieces of information that should be saved separately.

You are only interested in the following categories of information:

1. The family's food allergies (e.g. a dairy or soy allergy) - These are important to know because they can be life-threatening. Only log something as an allergy if you are certain it is an allergy and not just a dislike.
2. Foods the family likes (e.g. likes pasta) - These are important to know because they can help you plan meals, but are not life-threatening.
3. Foods the family dislikes (e.g. doesn't eat mussels or rarely eats beef) - These are important to know because they can help you plan meals, but are not life-threatening.
4. Attributes about the family that may impact weekly meal planning (e.g. lives in Austin; has a husband and 2 children; has a garden; likes big lunches, etc.)

When you receive a message, you perform a sequence of steps consisting of:

1. Analyze the most recent Human message for information. You will see multiple messages for context, but we are only looking for new information in the most recent message.
2. For each discrete piece of information, determine the following two pieces of information:
2.a. Who is this information relevant to? (e.g. I, Wife, Husband, Daughter, Family, etc)
2.b. What is the condensed bit of knowledge? Do not include extraneous information. (For example, "I like pasta it's so delicious!" should be condensed to "I like pasta"; "I'm a vegetarian, I don't eat meat" should be condensed to "I'm a vegetarian")
3. Combine those two pieces of information into a single piece of information to be stored for future reference in the format: [person(s) this is relevant to] [fact to store] (e.g. Husband doesn't like tuna; I am allergic to shellfish; etc)
4. Review to make sure that EVERY BIT of helpful information is stored in the smallest amount of text possible. You can make changes to your response as many times as you need to get it perfect.

I will tip you $20 if you are perfect, and I will fine you $40 if you miss any important information or change any incorrect information.

Return the information fragments in the following format:

{format_instructions}

Take a deep breath, think step by step, and then analyze the following messages:
"""

system_prompt_close = """
You previously attempted this task. Incorporate the feedback below to generate better results:

{previous_memory_analysis}
"""


class Memory(BaseModel):
    knowledge: str = Field(
        description="Condensed bit of knowledge to be saved for future reference in the format: [person(s) this is relevant to] [fact to store]"
    )


class Memories(BaseModel):
    memories: List[Memory] = Field(description="List of memories")


parser = JsonOutputParser(pydantic_object=Memories)

prompt_without_previous_analysis = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt_initial),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(format_instructions=parser.get_format_instructions())


prompt_with_previous_analysis = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt_initial),
        MessagesPlaceholder(variable_name="messages"),
        SystemMessagePromptTemplate.from_template(system_prompt_close),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Choose the LLM that will drive the agent
llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    # model="gpt-4-0125-preview",
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}},
)

memory_extractor_runnable = prompt_without_previous_analysis | llm | parser
memory_extractor_with_feedback_runnable = prompt_with_previous_analysis | llm | parser
