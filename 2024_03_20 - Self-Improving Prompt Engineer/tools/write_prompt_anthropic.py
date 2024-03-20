from langchain.tools import StructuredTool
from typing import List
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.output_parsers import XMLOutputParser


PROMPT_WRITER_PROMPT_ANTHROPIC = """
You are an expert Prompt Engineer. Your job is to dramatically improve upon one aspect of a prompt in order to improve the accuracy of the prompt's output.

I will provide you with 4 areas of a prompt, and you should choose the one with the most potential to improve and return your new variant on that prompt. I will also provide you with a history of changes to those prompts, as well as the evaluation results of those prompts so you can learn from previous attempts to modify the prompt. The evaluation results will focus on failure cases, and will provide you with a sense of what is not working with each change.

The ultimate goal of the prompt you are working on is to get an LLM to extract information from a chat history related to a family's eating habits.

Your job is to analyze the history of changes, and then propose a change to one aspect of the prompt that you think will result in a better performing prompt.

Here are the 4 parts of the prompt:

Part 1:
Opener - This is the first part of the prompt, and it should set the stage for the task at hand.

Here is the current opener:

<opener>
{opener}
</opener>

Part 2:
Instructions - This is the second part of the prompt, and it should clearly explain the outcome of this exercise.

Here are the current instructions:
<instructions>
{instructions}
</instructions>

Part 3:
Chain of Thought - This is the third part of the prompt, and it should guide the LLM through the steps they need to take to get to the desired end result.

Here are the current chain of thought instructions:
<chain_of_thought>
{chain_of_thought}
</chain_of_thought>

Part 4:
Closer - This is the final part of the prompt, and it should wrap up the task at hand. Use interesting techniques to encourage the prompt to perform well.

Here is the current closer:
<closer>
{closer}
</closer>

Here are the past changes to the prompt, and some of the evaluation results that you should consider when making your change:

<prompt_history>
{prompt_history}
</prompt_history>

Now it's time for you to make your change. Choose one of the 4 parts of the prompt, and then provide your new variant on that part.

The output should be formatted as a XML file with the following two tags:

- Prompt part you are changing in <prompt_part></prompt_part> tags. Must be one of: opener, instructions, chain_of_thought, closer
- The new value you are proposing in <new_value></new_value> tags. This is where you should put your unique variatiant on the prompt part you are changing. Be imaginative, but do your best to make the LLM perform well.

Wrap your answer in <response></response> tags.

DO NOT INCLUDE ANY RESPONSE OTHER THAN THE XML TAGS. ANYTHING ELSE WILL BE CONSIDERED AN INVALID RESPONSE.

Take a deep breath, think step, by step, and begin!
"""

# Please structure your response in the following XML tags:


parser = XMLOutputParser(tags=["prompt_part", "new_value"])

prompt = PromptTemplate(
    template=PROMPT_WRITER_PROMPT_ANTHROPIC,
    input_variables=[
        "opener",
        "instructions",
        "chain_of_thought",
        "closer",
        "prompt_history",
    ],
)

llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=1.0,
)

prompt_engineer_anthropic_runnable = prompt | llm | parser
