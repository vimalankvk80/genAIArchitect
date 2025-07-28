
import os
import requests
from dotenv import load_dotenv


load_dotenv()

from langfuse.callback import CallbackHandler
callback_handler = CallbackHandler()

from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(
    deployment_name="myllm", 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0,
    callbacks=[callback_handler])

from pydantic import BaseModel, Field
from typing import List
class SentimentAnalysis(BaseModel):
    # model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    company_name: str
    stock_code: str
    newsdesc: str
    sentiment: str = Field(..., decription="Positive/Negative/Neutral")
    people_names: List[str]
    places_names: List[str]
    other_companies_referred: List[str]
    relateed_industries: List[str]
    market_implications: str
    confidence_score: float

class StockTicker(BaseModel):
    company_name: str
    stock_code: str

from langchain_core.output_parsers import PydanticOutputParser
stock_parser = PydanticOutputParser(pydantic_object=StockTicker)
parser = PydanticOutputParser(pydantic_object=SentimentAnalysis)

from langchain_core.prompts import PromptTemplate
sentiment_prompt = PromptTemplate(
    template="""
    You are an AI Stock Analyst, Your job is to,

    Analyze the following news articles about a company and return 
    sentiment and extract names entiries.

    Company Name: {company_name}
    Stock Code: {stock_code}
    News Sumary: {newsdesc}

    Return a JSON with this structure:
    {format_instructions}
    """,
    input_variables=["company_name", "stock_code", "newsdesc"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

from langchain_core.runnables import RunnableLambda

# Company Lookup
lookup_table = {
        "Apple Inc": "AAPL",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "Meta": "META"
    }

def lookup_stock_symbol(company_name: str):
    return lookup_table.get(company_name, "N/A")

get_stock_code = RunnableLambda(lambda x: lookup_stock_symbol(x))

stock_code_prompt = PromptTemplate(
    template="""
    You are Stock Ticker Identifier.
    
    Given the company name below, return the official stock ticker symbol in Structured JSON Format

    Company Name: {company_name}

    Respond with only valid JSON (no markdown, no explanation):
    {format_instructions}
""",
input_variables= ["company_name"],
partial_variables={"format_instructions": stock_parser.get_format_instructions()}
)

get_llm_stock_code = stock_code_prompt | llm

#
# Prompt Category
#
classifier_prompt = PromptTemplate(
    input_variables=["newsdesc"],
    template="""
    Classify the news below as one of: financial, tech, generic.
    News: {newsdesc}
    Category:
"""
)

classifier_chain = classifier_prompt | llm

prompt_finance = PromptTemplate(
    input_variables=["company_name", "stock_code", "newsdesc"],
    template="""
    You are a financial analyst. Analyze the financial news.
    Company: {company_name}
    Stock Code: {stock_code}
    News: {newsdesc}
    {format_instructions}
""",
partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=SentimentAnalysis).get_format_instructions()}
)

prompt_tech = PromptTemplate(
    input_variables=["company_name", "stock_code", "newsdesc"],
    template="""
    You are a Tech Industry analyst. Analyze the Tech news.
    Company: {company_name}
    Stock Code: {stock_code}
    News: {newsdesc}
    {format_instructions}
""",
partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=SentimentAnalysis).get_format_instructions()}
)

prompt_generic = PromptTemplate(
    input_variables=["company_name", "stock_code", "newsdesc"],
    template="""
    Analyze the generic business news.
    Company: {company_name}
    Stock Code: {stock_code}
    News: {newsdesc}
    {format_instructions}
""",
partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=SentimentAnalysis).get_format_instructions()}
)

from langchain.chains.llm import LLMChain

finance_chain = LLMChain(llm=llm, prompt=prompt_finance)
tech_chain = LLMChain(llm=llm, prompt=prompt_tech)
generic_chain = LLMChain(llm=llm, prompt=prompt_generic)
# router_chain = LLMChain(llm=llm, prompt=cl)

destination_chains = {
    "financial": finance_chain,
    "tech": tech_chain,
    "generic": generic_chain
}

# Define LLM Chain Map
prompt_infos_1 = {
    "financial": prompt_finance | llm | PydanticOutputParser(pydantic_object=SentimentAnalysis),
    "tech": prompt_tech | llm | PydanticOutputParser(pydantic_object=SentimentAnalysis),
    "generic": prompt_generic | llm | PydanticOutputParser(pydantic_object=SentimentAnalysis)
}

prompt_infos = [
    {
        "name": "financial",
        "description": "prompt for a financial analyst",
        "prompt_template": prompt_finance,
    },
    {
        "name": "tech",
        "description": "prompt for a Tech analyst",
        "prompt_template": prompt_tech,
    },
    {
        "name": "generic",
        "description": "prompt for a generic business",
        "prompt_template": prompt_generic,
    }
]


from langchain.chains.router.multi_prompt import MultiPromptChain
# multi_prompt_chain = MultiPromptChain.from_prompts(llm, 
#                                                    prompt_infos,
#                                                    # default_chain=prompt_infos_1["generic"],
#                                                    #router_chain=classifier_chain,
#                                                    #input_key="newsdesc"
#                                                    )

# multi_prompt_chain = MultiPromptChain(
#     name="multi_prompt",
#     destination_chains=destination_chains,
#     router_chain=classifier_chain,
#     default_chain=generic_chain
#     )



import xml.etree.ElementTree as et

def fetch_news_from_google_rss_feed(symbol: str):
    news_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}"
    news_url = f"https://news.google.com/rss/search?q={symbol}"
    try:

        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url=news_url, timeout=10)
        if response.status_code == 200:
            root = et.fromstring(response.content)
            items = root.findall(".//item")
            headlines = [item.find("title").text.strip() for item in items[:5] if item.find("title") is not None]
            return "\n".join(headlines) if headlines else "No Headlines found."
        else:
            return f"Failed to fetch news: HTTP {response.status_code}" 

    except Exception as exc:
        return f"Error Fetching the news: {str(exc)}"
    
get_news = RunnableLambda(lambda x: {
    "company_name": x["company_name"],
    "stock_code": x["stock_code"],
    "newsdesc": fetch_news_from_google_rss_feed(x["stock_code"])
})


# sentiment chain
sentiment_chain = sentiment_prompt | llm | parser

# Entity linking via Wikipedia analyzer
def link_entity_to_wikipedia(name: str):
    try:
        wiki_url = f"https://en.wikipedia.org/w/api.php"
        response = requests.get(url=wiki_url, params={
            "action": "opensearch",
            "search": name,
            "limit": 1,
            "namespace": 0,
            "format": "json"
        },
        timeout=5)
        data = response.json()
        return data[3][0] if len(data) > 3 and data[3] else ""
    except Exception as exc:
        return ""
    
def enrich_with_entity_links(result_dict: dict):
    linked = {
        "linked_people": {name: link_entity_to_wikipedia(name) for name in result_dict.get("people_names", [])},
        "linked_companies": {name: link_entity_to_wikipedia(name) for name in result_dict.get("other_companies_referred", [])}
    }
    result_dict.update(linked)
    return result_dict

from langfuse import Langfuse
langfuse = Langfuse()

def stock_market_sentiment_chain(company_name: str, llm_ticker: bool = True, is_multi_prompt: bool = True):
    
    # Find the Ticker
    if  llm_ticker:
        new_stick_code = get_llm_stock_code.invoke(company_name)
        news_input = json.loads(new_stick_code.content)
        print("LLM Sticker Code : ", news_input)
    else:
        stock_code = get_stock_code.invoke(company_name, config={"callbacks": [callback_handler]})
        print("stock_code: ", stock_code)
        news_input = {"company_name": company_name, "stock_code": stock_code}

    # Get News
    news_desc = get_news.invoke(news_input, config={"callbacks": [callback_handler]})
    print("news_desc : ", news_desc)

    # Find the Sentiment.
    # if is_multi_prompt:
    #     # sentiment_result = multi_prompt_chain.invoke(input=news_desc, config={"callbacks": [callback_handler]})
    #     sentiment_result = multi_prompt_chain.run(news_desc)
    # else:
    sentiment_result = sentiment_chain.invoke(input=news_desc, config={"callbacks": [callback_handler]})

    print("sentiment result : ", sentiment_result)
    
    # Find the Entity
    entity = sentiment_result.model_dump() 
    result = enrich_with_entity_links(entity)
    # return {"company_name": company_name, "stock_code": stock_code, "news_desc": news_desc}
    return result

import json

if __name__ == "__main__":
    
    # print(" New Sticker Code : ", json.dumps(new_stick_code.content, indent=2))

    result = stock_market_sentiment_chain("Amazon")
    # result = fetch_news_from_yahoo_rss_feed("MSFT")
    print(result)
    # print(json.dumps(result.dict(), indent=2))







