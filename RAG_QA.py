import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter


MODEL = "llama3.1"
DATA = './data/ArtOfWar/ArtOfWar_intro.pdf'
template = """
Answer the question based on the context below. If you can't answer the question, reply "I don't know".

Context: {context}

Question: {prompt}
"""


loader = PyPDFLoader(DATA)
pages = loader.load_and_split()
embeddings = OllamaEmbeddings(model=MODEL)


@st.cache_resource
def get_vector_store(_documents, _embeddings):
    vector_store = DocArrayInMemorySearch.from_documents(
        documents=_documents,
        embedding=_embeddings
    )
    return vector_store
vector_store = get_vector_store(pages, embeddings)
retriever = vector_store.as_retriever()


model = Ollama(model=MODEL)


prompt = PromptTemplate.from_template(template=template)
chain = (
    {
        "context": itemgetter("prompt") | retriever,
        "prompt": itemgetter("prompt")
    }
    | prompt
    | model
)


st.title(f"RAG QA with {MODEL}")


if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input()


if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append(
        {
            'role': 'user',
            'content': prompt
        }
    )
    with st.chat_message("assistant"):
        response = st.write_stream(chain.stream(
            {
                "prompt": prompt
            }
        ))
    st.session_state.messages.append({'role': 'assistant', 'content': response})
