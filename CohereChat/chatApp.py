import streamlit as st
from dotenv import load_dotenv
import os
from langchain.chains import ConversationChain
from langchain_community.llms import Cohere
from langchain_community.chat_models import ChatCohere
from langchain.memory import ConversationBufferMemory


def main():
    load_dotenv()

    if os.getenv("COHERE_API_KEY") is None or os.getenv("COHERE_API_KEY") =="":
        print("COHERE_API_KEY is not set. Please add your key to env")
        exit(1)
    else:
        print('API key is set')

    llm = ChatCohere(model='command',temperature=0.75)
    conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory(),verbose=True)

    print("Hello, I am Sethyne!")

    while True:
        user_input =input(">")
        ai_response =conversation.predict(input=user_input)

        print("\nAssistant:\n",ai_response)
        


if __name__=='__main__':
    main()
