import streamlit as st
from typing import Optional, List
import requests


class CustomOpenAI():
    def __init__(self, base_url: str, model: str, api_key: str = None, temperature: float = None):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def __call__(self, messages: List, stop: Optional[List[str]] = None) -> str:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            },
            headers=headers
        )
        response.raise_for_status()
        completion = response.json()
        return completion["choices"][0]["message"]["content"]


llm = CustomOpenAI(
    base_url="http://localhost:9000/v1",
    model="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
    api_key="lm-studio",
)

st.title('Basic Streamlit')


if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'system', 'content': 'Always answer concisely.'}]

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input()


if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    with st.spinner(text="..."):
        response = llm(st.session_state.messages)
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})
