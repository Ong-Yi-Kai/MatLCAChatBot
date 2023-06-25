import os
import openai
from sentence_transformers import SentenceTransformer
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import ( SystemMessagePromptTemplate, HumanMessagePromptTemplate,
                                ChatPromptTemplate, MessagesPlaceholder )
from langchain.llms import OpenAI
from streamlit_chat import message
import json
from utils import *

# Frontend BS
st.set_page_config(page_title="LCA GPT", page_icon="♻", layout="wide")

with st.sidebar:
    st.header("Life Cycle Analysis Chatbot")
    st.markdown("# About 🙌")
    st.markdown(
        "LCA-GPT allows you to talk to version of chatGPT \n"
        "that has access to many papers regarding LCA \n"
        )

# Credentials /  keys
os.environ['OPENAI_API_KEY'] = st.secrets.openai.api_key


pinecone.init(api_key=st.secrets.pinecone.api_key, environment=st.secrets.pinecone.env)
index = pinecone.Index(st.secrets.pinecone.index_name)

llm = OpenAI(model_name="text-davinci-003")         # model used to come up with response
model = SentenceTransformer('all-MiniLM-L6-v2')     # model used to find matches and to encode initial pdfs



# conversation state to store history of conservation
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context,
and if the answer is not contained within the text below, say 'I don't know'""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# creating user interface
st.title("LCA Chatbot")
# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            # Make our QA-Model smarter via refining Queries and finding matches with utility functions
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query, model, index)
            # print(context)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
