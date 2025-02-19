from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from langchain_google_genai import ChatGoogleGenerativeAI
import openai

import os
from dotenv import load_dotenv
load_dotenv()


GOOGLE_API_KEY = "AIzaSyCqNzDqQ6grXOKAdLIkOKjcD0AIqApNcGg"

web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Alway include sources"],
    show_tools_calls=True,
    markdown=True,

)

word_search_agent=Agent(
    name="Word Search Agent",
    role="Search for 5 basic words and their definition that are used in finance and should be helpful for people of rural India.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Alway include sources"],
    show_tools_calls=True,
    markdown=True,

)

finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=["Use bullet points and numbers to write the current news regrading the finance market of India. Make a 50 word document so that it can be easily understood by rural people of India "],
    show_tool_calls=True,
    markdown=True

)

multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["First show some news regrading the financial domain"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Show me today's news on rural market of India",stream=True)

