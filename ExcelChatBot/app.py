import streamlit as st
import pandas as pd

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

st.title("Excel ChatBot")
st.subheader("Stack used: LangChain Agent, Streamlit, OpenAI LLM - by https://github.com/jaglinux", divider='rainbow')

uploaded_file = st.file_uploader("Choose a file", type=['csv','xlsx'])
if uploaded_file is None:
    df = pd.read_csv("titanic.csv")
    st.write("Default file uploaded, titanic.csv")
else:
    # Can be used wherever a "file-like" object is accepted:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
st.dataframe(df, height=5)

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
if question := st.chat_input("Ask Question to the csv/xlsx provided"):
    response = agent.invoke(question)
    print(response['output'])
    st.chat_message("user").markdown(question)
    st.chat_message("assistant").markdown(response['output'])