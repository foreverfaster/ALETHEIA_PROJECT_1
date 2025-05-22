#.env files store key-value pairs (like API keys or configuration settings), and load_dotenv() loads these into the runtime environment (os.environ) so that they can be accessed programmatically.
#LangChain — a framework that allows LLMs to act like agents by calling tools, making decisions, and producing intermediate reasoning steps.
from dotenv import load_dotenv

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
#ChatOpenAI is a wrapper that allows OpenAI models to be used seamlessly in LangChain pipelines with consistent APIs, especially for chat-based interfaces.

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent,AgentExecutor #can create any agent
from tools import search_tool, wiki_tool, save_tool

#define asimple Python calss which will specify the type of content 
#that we want our LLM to generate

load_dotenv() #load the environment var. file i.e .env which we crreated 
#this line is possible because of from dotenv import load_dotenv
#setting up a llm  for the agent to use .
#load .env content like api keys and config settings to access it in our program

os.environ["OPENAI_API_KEY"] = os.getenv("TOGETHER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.together.xyz/v1"

class HarshasResearch(BaseModel): #this class is inherited from basemodel i.e pydantic
    
    topic:str
    summary:str
    sources:list[str]
    tools_used:list[str]
    #can have n nos. of fileds nested objects with incresing complexity
    #eventually these are passed to the LLM


llm=ChatOpenAI(model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free")
parser=PydanticOutputParser(pydantic_object=HarshasResearch)
#can use other ways to set up this parser like json , here v r using pydantic
#tis parser tkes the output of LLM and parse it into class Hrsha(BModel)
#so as we could use it normal py object

prompt= ChatPromptTemplate.from_messages(
        [
        (
            "system", #info to LLM
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query} {name}"), #It could have multiple queries,cums from the user
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions()
          #the above line is important
          )

tools=[search_tool,wiki_tool,save_tool] 
agent=create_tool_calling_agent( #a function
    llm=llm,
    prompt=prompt,
    tools=tools

)
#61:a custom or LangChain utility function that:Creates an Agent — a specialized interface around an LLM (e.g., GPT-4) that can interact with tools and reason step by step.
#62: This is your language model instance (e.g., ChatOpenAI(), OpenAI()), already initialized.
#63: : A system prompt or template that defines how the agent should behave (e.g., be a tutor, give short answers, use tools, etc.).
#64: ou're passing in an empty list — meaning this agent has no tools to call, such as search engines, calculators, or custom functions.
# Analogy: You’re giving the agent a brain (llm) and a personality (prompt), but no arms or gadgets (tools=[]), so it must think its way through everything.

#75:You’re now wrapping the agent in an executor — which is what actually runs the agent on real input. Think of AgentExecutor like a task manager that:
#Handles running the agent,
#Tracks steps (if enabled with verbose), and
#Passes tools to it (if any).

#agent=agent: This is the agent you just created.
#tools=[]: Still no tools. Even though the executor can pass tools, you’re keeping it empty.
#verbose=True: Tells LangChain to print intermediate reasoning steps, like thought chains, tool selection (if any), and final answer generation.

#The Agent is a smart person. The AgentExecutor is their assistant who records everything they say and manages their tasks.

agent_executor=AgentExecutor(agent=agent,tools=tools, verbose=True)
query=input("What can i help you research ?")
raw_response=agent_executor.invoke({"query":query,"name":"Harshvardhan"})
print(raw_response)

#response=llm.invoke("How to get  of the past ")
#print(response.content)

#PLEASE check the differnce between shell , powershell , bash, cmd , etc in notes

#27. BaseModel is a class from a Python library called Pydantic.
#29. These are fields (or attributes) of your HarshasResearch object.
#37. ChatOpenAI:This is a class provided by LangChain that wraps chat-based LLMs (like GPT, Claude, or DeepSeek).
#It creates a standard interface so you can talk to any LLM in the same way — no matter the backend (OpenAI, Anthropic, Together, etc.).
# model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
#tells LangChain which specific model to use.
#38. PydanticOutputParse:his is a LangChain class that’s specifically designed to convert the output generated by a language model into a structured Python object based on a Pydantic model.
#Pydantic is a Python library that helps you define data structures and automatically checks that the data is correct (e.g., type, required fields).
#Without this PydanticOutputParser, the output from the language model would just be plain text. It would be messy and unstructured. You would have to manually extract the parts you care about. But with the PydanticOutputParser, it ensures that: The output matches the structure defined in HarshasResearch (topic, summary, sources, tools used).  It automatically converts the raw text from the LLM into a clean, usable Python object that you can access like:
#A parser is a tool or function that takes messy, unstructured input (usually text) and turns it into structured, machine-readable data.


# A CODE TO JUST LOOKM THE RESPONSE NOT TTHE THINKEINH PROCESSS:WE USE PARSER TO PARSE THE PARSE TTHIS CONTENT
structured_response = parser.parse(raw_response["output"])
print(structured_response)

try:
    structured_response=parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing",e,"Raw Response-", raw_response)
