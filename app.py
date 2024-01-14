import streamlit as st
import tiktoken
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatCohere
from htmlTemplate import css, user_template, bot_template
from langchain_community.llms import Cohere



def get_pdf_text(pdf_docs):
    text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap =200,
        length_function=len

    )
    chunks = text_splitter.split_text(text)
    return chunks
def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key='KLxxk4D5YgzHbisanSwQe5nWIBCuLIUC6gCbxAyF')

    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI()
    llm=ChatCohere(model="command", max_tokens=556, temperature=0.6,cohere_api_key='KLxxk4D5YgzHbisanSwQe5nWIBCuLIUC6gCbxAyF')
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",model_kwargs={"temperature":0.6,"max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain =ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response =st.session_state.conversation({'question':user_question})
    st.session_state.chat_history=response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2== 0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None


    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about documents:")

    if user_question:
        handle_userinput(user_question)

   

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):

            # Placeholder for processing logic
            # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
            # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                
            # create vector store
                vectorstore = get_vectorstore(text_chunks)
            # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("Processing complete!")
    

if __name__ == '__main__':
    main()