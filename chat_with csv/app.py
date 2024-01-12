import streamlit as st
#from langchain.agents import create_csv_agent
from dotenv import load_dotenv
from langchain_community.llms import Cohere
from langchain_experimental.agents import create_csv_agent



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with CSV")
    st.header("Ask you csv:")

    user_csv = st.file_uploader("Upload your file here")

    if user_csv is not None:
        user_question = st.text_input("Ask question about csv: ")

        llm = Cohere(temperature=0) # return exact value in the csv
        agent =create_csv_agent(llm,user_csv,verbose=True)

        if user_question is not None and user_question !="":
            st.write(f"Your question was: {user_question}")
            response = agent.run(user_question)
            st.write(response)




if __name__=='__main__':
    main()