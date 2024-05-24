from collections import defaultdict

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

default_url = "https://rocm.docs.amd.com/en/latest/what-is-rocm.html"
st.title("URL Loader")
st.subheader("Stack used: LangChain, FaissDB for RAG, Streamlit, OpenAI LLM - by https://github.com/jaglinux", divider='rainbow')

embeddings = OpenAIEmbeddings()

url = st.text_input("Provide URL ", default_url)
if "url_dict" not in st.session_state:
    st.session_state.url_dict = {}
if url not in st.session_state.url_dict:
    loader = WebBaseLoader(url)
    documents = loader.load()
    st.session_state.url_dict[url] = defaultdict(dict)
    st.session_state.url_dict[url]['documents'] =  documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)
    print(db.index.ntotal)
    url_hash = "faiss_index" + str(abs(hash(url)))
    db.save_local(url_hash)
    st.session_state.url_dict[url]['FAISS_db'] = url_hash

llm = ChatOpenAI(temperature=0.1)
prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question : {input}
    """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages[-2:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if question := st.chat_input("Ask Question to the URL provided"):
    # Display user message in chat message container
    st.chat_message("user").markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    db = FAISS.load_local(st.session_state.url_dict[url]['FAISS_db'],
                              embeddings, allow_dangerous_deserialization=True)
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    retriever = db.as_retriever(search_kwargs={"k": 2})
    chain = create_retrieval_chain(retriever, document_chain)

    response = chain.invoke({
    "input" : question
    })
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response["answer"])
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

