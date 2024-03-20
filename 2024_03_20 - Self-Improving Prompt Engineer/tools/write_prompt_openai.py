from langchain.tools import StructuredTool
from typing import List
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


PROMPT_WRITER_PROMPT = """
You are an expert Prompt Engineer. Your job is to improve upon different aspects of a prompt in order to get the output of an LLM to be as close to the desired output as possible.

I will provide you with 4 areas of a prompt to experiment with. I will also provide you with a history of changes to those prompts, as well as the evaluation results of those prompts. The evaluation results will focus on failure cases, and will provide you with a sense of what is not working with each change.

The ultimate goal of the prompt you are working on is to get an LLM to extract information from a chat history related to a family's eating habits.

Your job is to analyze the history of changes, and then propose a change to one aspect of the prompt.

Here are the 4 parts of the prompt:

Part 1:
Opener - This is the first part of the prompt, and it should set the stage for the task at hand.

Here is the current opener:
```
{opener}
```

Part 2:
Instructions - This is the second part of the prompt, and it should clearly explain the outcome of this exercise.

Here are the current instructions:
```
{instructions}
```

Part 3:
Chain of Thought - This is the third part of the prompt, and it should guide the LLM through the steps they need to take to get to the desired end result.

Here are the current chain of thought instructions:
```
{chain_of_thought}
```

Part 4:
Closer - This is the final part of the prompt, and it should wrap up the task at hand. Use interesting techniques to encourage the prompt to perform well.

Here is the current closer:
```
{closer}
```

Here are the past changes to the prompt and the result of the evaluations for each, sorted from oldest to newest runs. You should absolutely consider each when making your change:

```
{prompt_history}
```

Follow this process:
1. Analyze the prompt history and the evaluation results.
2. Consider which part of the prompt has the most potential to improve.
3. Create a new version of the part of the prompt you have decided to address. It should try to overcome the failures from previous attempts to modify the prompt. Be creative and do not limit yourself at all to the existing prompt, you can write whatever you want to make this work.

Share the new variant to pass directly into the next test.

Take a deep breath, think step, by step, and begin!

{format_instructions}
"""


# Define your desired data structure.
class Modification(BaseModel):
    prompt_part: str = Field(
        description="The prompt part being changed. Must be one of: opener, instructions, chain_of_thought, closer"
    )
    new_value: str = Field(description="The new value for the prompt part")


parser = JsonOutputParser(pydantic_object=Modification)

prompt = PromptTemplate(
    template=PROMPT_WRITER_PROMPT,
    input_variables=[
        "opener",
        "instructions",
        "chain_of_thought",
        "closer",
        "prompt_history",
    ],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Choose the LLM that will drive the agent
llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    # model="gpt-4-0125-preview",
    temperature=0.8,
    model_kwargs={"response_format": {"type": "json_object"}},
)

prompt_engineer_runnable = prompt | llm | parser


def write_prompt(prompt_history: List[str]) -> str:
    print(prompt_history)
    return "Wrote prompt"


class BlankSchema(BaseModel):
    pass


tool_prompt_writer = StructuredTool.from_function(
    func=write_prompt,
    name="Prompt_Writer",
    description="This tool simply calls the same function every time",
    args_schema=BlankSchema,
)
