import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from tools.tool import get_profile_url_tavily

load_dotenv()

def lookup(name: str)-> str:

    llm = ChatOllama(model="mistral")

    summary = """ given the full name {name_of_person} from I want you to get it me a link to their LinkedIn profile's page. Your answer should only contain a URL"""

    prompt_template = PromptTemplate(input_variables=["name_of_person"], template=summary)

    tools_for_agent = [
        Tool(
            name = "Crawl Google 4 linkedin profile page",
            func= get_profile_url_tavily,
            description="useful for when you need to get the LinkedIn Page URL",
        )
    ]
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)
    
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url

if __name__ == "__main__":
    linkedin_url = lookup("Eden Marco")
    print(linkedin_url)
