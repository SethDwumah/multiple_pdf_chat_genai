import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_cohere import ChatCohere,CohereEmbeddings
from htmlTemplate import css, user_template, bot_template
#from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import os, getpass


# Define the API key directly

#GOOGLE_API_KEY= "AIzaSyBGbRBj8BTvbjcl1q88U2b_9bkIwRM_xSY"

COHERE_API_KEY = 'BrfGKg2jyraO29bD1iaKqKD3i60cdDcZ7e0mBbCu'

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)
    #embeddings = embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="AIzaSyBGbRBj8BTvbjcl1q88U2b_9bkIwRM_xSY")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatCohere(model="command-r-plus", max_tokens=800, temperature=0.6, cohere_api_key=COHERE_API_KEY,stream=True)
    #llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",max_output_tokens=564,temperature=0.6,google_api_key= "AIzaSyBGbRBj8BTvbjcl1q88U2b_9bkIwRM_xSY")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

# Function to handle user input and generate responses
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Conversation is not initialized.")
        return

    try:
        response = st.session_state.conversation.invoke({'question': user_question})
        if response is not None:
            st.session_state.chat_history = response.get('chat_history', [])
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.error("No response received from the conversation chain.")
    except Exception as e:
        st.error(f"An error occurred: {e}")


# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.chat_input("Ask a question about documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Processing complete!")

if __name__ == '__main__':
    main()

