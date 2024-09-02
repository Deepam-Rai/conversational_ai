import streamlit as st
from typing import Optional, List
from openai import OpenAI


class CustomOpenAI():
    def __init__(self, base_url: str, model: str, api_key: str = None, temperature: float = None, stream: bool = False):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.stream = stream
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def __call__(self, messages: List, stop: Optional[List[str]] = None, stream: bool = True) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=stream or self.stream
        )
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


llm = CustomOpenAI(
    base_url="http://localhost:9000/v1",
    model="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
    api_key="lm-studio",
)

st.title('Basic Streamlit II')


if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'system', 'content': 'Answer concisely but with adequate details.'}]

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input()


if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    with st.chat_message('assistant'):
        response = st.write_stream(llm(messages=st.session_state.messages, stream=True))
    st.session_state.messages.append({'role':'assistant', 'content':response})
