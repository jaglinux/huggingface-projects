import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader

default_url = "https://rocm.docs.amd.com/en/latest/what-is-rocm.html"
st.title("URL Loader")

url = st.text_input("Provide URL ", default_url)
if "url_dict" not in st.session_state:
    st.session_state.url_dict = {}
if url not in st.session_state.url_dict:
    loader = WebBaseLoader(url)
    st.session_state.url_dict[url] = loader.load()

llm = ChatOpenAI(temperature=0.1)
prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question : {input}
    """)
chain = prompt | llm

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask Question to the URL provided"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chain.invoke({
    "context" : [st.session_state.url_dict[url]],
    "input" : prompt
    })
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response.content)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})

