import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.llms import Cohere
from langchain_community.chat_models import ChatCohere
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def main():
    load_dotenv()

    if os.getenv("COHERE_API_KEY") is None or os.getenv("COHERE_API_KEY") =="":
        print("COHERE_API_KEY is not set. Please add your key to env")
        exit(1)
    else:
        print('API key is set')




        chat = ChatCohere(model='command',temperature=0.75)

        messsages = [
            SystemMessage(content="You are a helpful assistant"),

        ]

        while True:
            user_input =input(">")
            messsages.append(HumanMessage(content=user_input))

            ai_response =chat(messsages)

            messsages.append(AIMessage(content=ai_response.content))

            print("\nAssistant:\n",ai_response.content)


if __name__=='__main__':
    main()
