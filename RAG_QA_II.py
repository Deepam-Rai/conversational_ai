import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from operator import itemgetter
from langchain_chroma import Chroma


MODEL = "llama3.1"
DATA = './data/ArtOfWar'
DATA_STORE = "./indices"
template = """
Answer the question based on the context below. If you can't answer the question, reply "I don't know".

Context: {context}

Question: {prompt}
"""


loader = PyPDFDirectoryLoader(DATA)
documents = loader.load_and_split()
embeddings = OllamaEmbeddings(model=MODEL)


@st.cache_resource
def create_index_from_documents(
    _all_docs,
    _embeddings,
    persist_directory,
    collection_name
):
    print(f"Creating index for collection {collection_name}...")
    vectordb = Chroma.from_documents(
        documents=_all_docs,
        embedding=_embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(f"Finished creating index.")
    return vectordb


@st.cache_resource
def get_retriever(persist_directory, _embeddings, collection_name):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=_embeddings, collection_name=collection_name)
    return vectordb.as_retriever()


collection_name = DATA.replace(".","").replace("\\","").replace("/","")
create_index_from_documents(documents, embeddings, DATA_STORE, collection_name)
retriever = get_retriever(DATA_STORE, embeddings, collection_name)


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
