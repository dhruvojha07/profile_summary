# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from typing import Tuple
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import summary_parser, Summary

def ice_break_with(name: str) -> Tuple[Summary, str]:
    linkedin_username = linkedin_lookup_agent(name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username, mock=True)

    summary_template = """
        given the LinkedIn information {information} about a person from I want you to create:
        1. a short summary of the person
        2. two interesting facts about them
    \n {format_instructions}
    """
    summary_prompt_template = PromptTemplate(
        input_variables="information", template=summary_template, 
        partial_variables={"format_instructions":summary_parser.get_format_instructions()})
    llm = ChatOllama(model="llama3.1")

    chain  = summary_prompt_template | llm | summary_parser
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/eden-marco/", mock=True)

    res: Summary = chain.invoke(input= {"information": linkedin_data})
    return res, linkedin_data.get("profile_pic_url")



if __name__ == "__main__":
    load_dotenv()
    print("IceBreaker")
    ice_break_with(name="")

    