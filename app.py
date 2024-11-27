# powershell: streamlit run app.py --server.port 8051
# -*- coding: utf-8 -*-

#TOFIX: UnboundLocalError: cannot access local variable 'response' where it is not associated with a value

import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

class ChatLLM:

    def __init__(self):
        # Model
        self.model = ChatOllama(model = "gemma2:2b", temperature = 3)

        # Prompt
        self.template = """ Please provide a short and concise answer to the given question in English.

            Question:{question}
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)

        # Chain Linkers
        self._chain = (
            {"question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def invoke(self, user_input):
        response = self._chain.invoke(user_input)
        return response
    
    def format_doc(self, docs):
        return '\n\n'.join([d.page_content for d in docs])
    

class ChatWeb:
    def __init__(self, llm, page_title="Iskandar Chatbot", page_icon=":robot:"):
        self.page_title = page_title
        self.page_icon = page_icon
        self.llm = llm

    def print_messages(self):
        if "messages" not in st.session_state and len(st.session_state["messages"]) > 0:
            for chat_message in st.session_state["messages"]:
                st.chat_message(chat_message.role).write(chat_message.content)

    def run(self):
        # web page primary environment setup
        st.set_page_config(page_title=self.page_title, page_icon=self.page_icon)
        st.title(self.page_title)

        # reset the log messages
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # print the messages
        self.print_messages()

        # user input, chatbot response
        response = ""
        if user_input := st.chat_input("Please input any question."):
            # user input message
            st.chat_message("user").write(f"{user_input}")
            st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
            response = self.llm.invoke(user_input)

        # chatbot response
        with st.chat_message("assistant"):
            msg_assistant = response
            st.write(msg_assistant)
            st.session_state["messages"].append(ChatMessage(role="assistant", content=msg_assistant))             


if __name__ == "__main__":
    llm = ChatLLM()
    web = ChatWeb(llm = llm)
    web.run()