import os
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser


SYSTEM_PROMPT_EXPECTED_RESPONSE = """
Your job is to compare an actual response vs an expected response and evalute if the response is correct.

The wording may be a little different, but the meaning should be the same and should not omit any important details.

You need to evaluate 3 different aspects:
1. Did the agent correctly identify the new knowledge?
2. Did the agent correctly categorize the new knowledge?
3. Did the agent correctly determine if it should add, update, or delete the knowledge?

Here is the expected output:

```
{expected_output}
```

And here is the actual output:

```
{actual_output}
```

Remember: words do not need to be identical, but the meaning for each should be the same.

Now take a deep breath and compare the expected vs actual ouptut.

If the actual output matches the expected output across all 3 dimensions for every entry, then this should be evaluated as a pass. Otherwise, it is a failure and you should provide an incredibly brief explanation of what went wrong. Only describe what went wrong, not went right.

For example:
- Failed at Category: incorrectly assigned 'I avoid dairy' as an Allergy instead of a Dislike
- Failed at Action: incorrectly Created 'I like tacos' when it should have Updated 'I dislike tacos' to a Like

{format_instructions}
"""

SYSTEM_PROMPT_BAD_RESPONSE = """
Your job is to compare an actual response vs an incorrect response and evalute if the response is correct.

The wording may be a little different, but the meaning should be the same and should not omit any important details.

You need to evaluate 3 different aspects:
1. Did the agent correctly identify the new knowledge?
2. Did the agent correctly categorize the new knowledge?
3. Did the agent correctly determine if it should add, update, or delete the knowledge?

And here is the actual output:

```
{actual_output}
```

Here is the expected output:

```
{expected_output}
```

Remember: the words do not need to be identical, but the meaning for each should be the same.

Now take a deep breath and compare the expected vs actual ouptut.

If the actual output is different, respond with DIFFERENT. Otherwise, respond with PERFECT.
"""


# Define your desired data structure.
class EvalResult(BaseModel):
    eval: bool = Field(description="Was this perfect? Respond True or False")
    failure_reason: str = Field(description="If this failed, provide your explanation")


parser = JsonOutputParser(pydantic_object=EvalResult)

# Get the prompt to use\
prompt_expected = ChatPromptTemplate.from_messages(
    [SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_EXPECTED_RESPONSE)]
).partial(format_instructions=parser.get_format_instructions())

prompt_bad = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template(SYSTEM_PROMPT_BAD_RESPONSE),
    ]
)

# Choose the LLM that will drive the agent
llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0.0,
)

evaluate_expected_output_runnable = prompt_expected | llm | parser
evaluate_bad_output_runnable = prompt_bad | llm
