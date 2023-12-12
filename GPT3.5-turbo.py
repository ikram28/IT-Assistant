import os
from  dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.redis import Redis
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st 


load_dotenv()

st.set_page_config(page_title="IT Assistant", page_icon="🤖")
st.title("🤖    IT Assistant")



llm = ChatOpenAI(temperature = 0, openai_api_key=os.getenv("OPENAI_API_KEY"))


embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# initialize redis

rds = Redis.from_existing_index(
    embeddings, redis_url=os.getenv("redis_url"), index_name=os.getenv("index_name")
)

retriever = rds.as_retriever()



tool = create_retriever_tool(
    retriever,
    "search_IT",
    "Searches and returns documents regarding the IT service management."
)
tools = [tool]

agent_executor = create_conversational_retrieval_agent(llm, tools)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        st.write(" 🧠 Thinking...")
        result = agent_executor({"input": prompt})
        response = result["output"]
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})








   