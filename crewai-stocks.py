# import das libs 

import json                         
import os                           
from datetime import datetime       

import yfinance as yf               
from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st  # pip install streamlit

# from IPython.display import Markdown


# AGENTE 1: preço histórico de uma ação --------------------------------------------------

# AGENTE 1: FERRAMENTA

# criação da função para a ferramenta:
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08") 
    return stock

# criação da ferramenta:
yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Fetches stock prices for {ticket} from the last year about a specific company from Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket)
)


# IMPORTANDO OPENAI LLM - GPT:::
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']  # variável de ambiente que se chama OPENAI_API_KEY
llm = ChatOpenAI(model="gpt-3.5-turbo")


# AGENTE 1: AGENTE, busca o preço da ação.

stockPriceAnalyst = Agent(
    role="Senior stock price Analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory=""" You're a higly experienced in analyzing the price of an specific stock and make predictions about its future price.""",
    verbose=True,
    llm = llm,
    max_iter=5,                 
    memory=True,
    tools=[yahoo_finance_tool],  
    allow_delegation=False
)

# AGENTE 1: TASK

getStockPrice = Task(
    description="Analyze the stock {ticket} price history and create a trend analyses for up, down or sideways",
    expected_output=""" Specify the current trend stock price - up, down or sideways.
    eg. stock='AAPL, price UP'
    """,
    agent = stockPriceAnalyst       
)


# AGENTE 2: notícias --------------------------------------------------

# AGENTE 2: FERRAMENTA

search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)

# AGENTE 2: AGENTE, agente de notícias

newsAnalyst = Agent(
    role="Stock news Analyst",
    goal="Create a short summary of the market news related to the stock {ticket} company. Specify the current trend - up, down or sideways with the news context. For each requested stock specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.",
    backstory=""" 
    You're a higly experienced in analyzing the market trends and news and have tracked assets for more than 10 years. 
    You're also a master level analyst in the traditional markets and have a deep understanding of human psychology. 
    You understand news, their titles and information, but you look at those with a healthy dose of skepticism. 
    You consider also the source of the news articles.
    """,
    verbose=True,
    llm = llm,
    max_iter=10,                    
    memory=True,
    tools=[search_tool],            
    allow_delegation=False
)

# AGENTE 2: TASK

getNews = Task(
    description="""Take the stock and always include BTC to it (if not requested). Use the search tool to search each one individually.

    The current date is {datetime.now()}.

    Compose the results into a helpful report.
    """,
    expected_output="""A summary of the overall market and one sentence summary for each requested asset (stock and BTC).
    Include a fear/greed score for each asset (stock and BTC) based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
    agent = newsAnalyst             
)


# AGENTE 3: análise --------------------------------------------------

# AGENTE 3: AGENTE, agente de análise

stockAnalystReport = Agent(
    role="Senior Stock Analyst Writer",
    goal="Analyze the trends price and news and write an insightfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend.",
    backstory="""You're widely accepted as the best stock analyst in the market. 
    You understand complex concepts and create compelling stories and narratives that resonate with wider audiences.
    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analysis. 
    You're able to hold multiple opinions when analyzing anything.
    """,
    verbose=True,
    llm = llm,
    max_iter=5,
    memory=True,
    allow_delegation=True   
)

# AGENTE 3: TASK

writeAnalysis = Task(
    description=""" Use the stock price trend and the stock news report to create an analysis and write the newsletter about the {ticket} company that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analysis of stock trend and news summary.
    """ ,
    expected_output=""" An eloquent 3 paragraphs newsletter formated as markdown in a easy readable manner. 
    It should contain:
    - 3 bullets executive summary
    - Introduction: set the overall picture and spike up the interest 
    - main part: provides the meat of the analysis including the news summary and fear/greed scores 
    - summary: key facts and concrete future trend prediction - up, down or sideways.
    """,
    agent=stockAnalystReport,           
    context = [getStockPrice, getNews]  # contextos de quais tasks esta task (writeAnalysis) vai se basear
)


# CRIAÇÃO DO GRUPO DOS AGENTES: --------------------------------------------------
# 
# -> from crewai import Agent, Task, Crew
# -> o Crew permite a criação do grupo dos agentes de ia
# -> crew = Crew()
# 
# -> from crewai import Agent, Task, Crew, Process
# -> Process permite a especificação do processo de execução que o CREW vai seguir
# -> process= Process.hierarchical

"""
verbose = the verbosity level for loggin during execution
process = the process FLOW the crew follows -> eg. sequential process, hierarchical process
        - sequential: executes tasks sequentially, ensuring tasks are completed in an ORDERLY PROGRESSION.
        - hierarchical: 
            >> organiza as tasks de maneira hierárquica, para que ele consiga delegar e executar as tarefas baseando-se na estrutura da cadeia de comando.
            >> a manager language model (manager_llm) or a custom manager agent (manager_agent) must be specified in the CREW to enable the hierarchical process;; facilitating the creation and management of tasks by the manager

            [!!!] o agente de análise (stockAnalystReport) está delegando tarefas para os outros agentes, então o processo hierárquico é o que será usado.
"""

crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystReport],
    tasks = [getStockPrice, getNews, writeAnalysis],
    verbose = 2,
    process= Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm = llm,
    max_iter = 15
)


# EXECUÇÃO: --------------------------------------------------
"""
results = crew.kickoff(inputs={'ticket': 'AAPL'})

list(results.keys())
# output: ['final_output', 'tasks_outputs']

results['final_output']
# output: resultado final gerado pelo stockAnalystReport

len(results['tasks_output'])
# output: 3
# ou seja, existem 3 tasks

# from IPython.display import Markdown
# Markdown(results['final_output'])
# output: resultado gerado pelo stockAnalystReport em formato markdown
"""
# EXECUÇÃO VIA STREAMLIT: --------------------------------------------------

# cria um sidebar no navegador:
with st.sidebar:    
    st.header('Enter the Stock to Research')
    
    # formulário:
    with st.form(key='research_form'):                                  # formulário
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label="Run Research")

# submit_button provavelmente eh boolean
if submit_button:   # se o submit_button for pressionado ou o usuário apertar enter:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results = crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of your research:")
        st.write(results['final_output'])

"""
RODAR: streamlit run crewai-stocks.py
"""

# -------------------------------------------------------------------------