from langchain_anthropic import ChatAnthropic
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.pydantic_v1 import BaseModel, Field


class GenerateCritique(BaseModel):
    """Critique the AI analysis"""

    is_perfect: bool = Field(
        ..., description="True if the analysis is perfect, False otherwise"
    )
    criticism: str = Field(
        ...,
        description="What did the AI get wrong? If the analysis is perfect, explain why it is perfect",
    )


reviewing_system_prompt = """
Your job is to compare a set of extracted memories to the original message history. Is anything missing or incorrect? You are very thorough and detail-oriented, so I trust you to catch any mistakes.

Here is the AI's analysis of the original message history that you will analyze, delimited by triple backticks below:

```
{ai_analysis}
```

You are interested in making sure the AI captured all the key data that might relate to the following categories of information:

1. The family's food allergies (e.g. a dairy or soy allergy) - These are important to know because they can be life-threatening. Only log something as an allergy if you are certain it is an allergy and not just a dislike.
2. Foods the family likes and dislikes (e.g. likes pasta) - These are important to know because they can help you plan meals, but are not life-threatening.
3. Attributes about the family that may impact weekly meal planning (e.g. lives in Austin; has a husband and 2 children; has a garden; likes big lunches, etc.)

Your job is just to determine if the AI included EVERY critical piece of information that could be pertinent for the categories above from the messages, and that each is correct.

Follow these steps:

1. Silently read each of the memories that the AI extracted in its original analysis
2. Silently read the original message history and determine the memories that you would extract
3. Go through each memory that you would extract one at a time to see if there is a corresponding match in the AI analysis
4. Silently finalize your analysis: Did everything you identify have a match? Was there anything the AI identified that you did not?
5. Provide your critique in the format requested of you below

Here are some examples of what you might say in different scenarios:

```
Example 1:

If the AI's analysis is:

Memory 1: Family is all vegetarian
Memory 2: Family eats out 3 times a week.
Memory 3: Daughter is allergic to soy.

And the original sentence is: We eat out 3 times a week, but my wife doesn't eat meat.

Then you should call GenerateCritique with the following parameters:
is_perfect: false
criticism: Incorrectly stated that the family is all vegetarian, but only only the wife is vegetarian. Also incorrectly included that the daughter is allergic to soy, but that was not mentioned in the original message.
```

```
Example 2:

If the AI's analysis is:
Memory 1: Daughter is allergic to dairy.

And the original sentence is:
My youngest daughter is lactose intolerant.

Then you should call GenerateCritique with the following parameters:
is_perfect: false
crticism: Incorrectly stated that the daughter is allergic to dairy, it should have been more specific and said the youngest daughter is lactose intolerant.
```

```
Example 3:
If the AI's analysis is:
Memory 1: Family likes pasta.

And the original sentence is:
We like eating pasta.

Then you should call GenerateCritique with the following parameters:
is_perfect: true
criticism: NA
```

Take a deep breath, think step by step, and then share your analysis of how the AI performed in relation to the following analysis and original message history. Use the GenereateCritique tool to provide your analysis.

Here is the AI analysis and original message history to analyze:
"""

# Get the prompt to use - you can modify this!
reviewing_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(reviewing_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Choose the LLM that will drive the agent
llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    # model="claude-3-sonnet-20240229",
    temperature=0.0,
)

llm_with_tools = llm.bind_tools([GenerateCritique])

memory_reviewer_runnable = reviewing_prompt | llm_with_tools
