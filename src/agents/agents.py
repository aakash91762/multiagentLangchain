from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

# Model Initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0,
)

# 1st Agent : Search Agent
def build_search_agent():
    return create_agent(
        model= llm,
        tools=[web_search],
        system_prompt="""
        You are a web research reader.

        Rules:
        1. Select a URL only from the supplied search results.
        2. You MUST call scrape_url.
        3. Do not invent URLs.
        4. Do not answer from your own knowledge.
        5. If scrape_url returns SCRAPE_ERROR, report the failure.
        6. Never present search-result knowledge as scraped content.
        """
    )
    

# 2nd Agent : Reader Agent
def build_reader_agent():
    return create_agent(
        model= llm,
        tools=[scrape_url]
    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()




#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()