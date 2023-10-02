import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import base64
import os

def generate_chat_transcript():
    if 'chat_history' not in st.session_state or not st.session_state.chat_history:
        return None

    chat_str = ""
    for i, message in enumerate(st.session_state.chat_history):
        role = "User" if i % 2 == 0 else "Bot"
        chat_str += f"{role}: {message.content}\n\n"
    
    return chat_str

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def handle_userinput(user_question):
    with st.spinner("Waiting for API response..."):
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, str):
        b64 = base64.b64encode(object_to_download.encode()).decode()
    else:
        b64 = base64.b64encode(object_to_download).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if not user_question or "conversation" not in st.session_state:
        return
    
    with st.spinner("Waiting for API response..."):
        
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# Download Chat Transcript
    transcript = generate_chat_transcript()
    if transcript:
        b64 = base64.b64encode(transcript.encode()).decode()
        href = f'<a href="data:text/plain;charset=utf-8;base64,{b64}" download="chat_transcript.txt">Download Chat Transcript</a>'
        st.markdown(href, unsafe_allow_html=True)

        

def pdf_chat_main():
    load_dotenv()

    # Sidebar for API key input saved as environment variable
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:secret:", type="password")
    st.sidebar.button("Submit")
    st.sidebar.warning("Please enter your OpenAI API key to use the app.")
    st.sidebar.info("Don't have an API key? Get one [here](https://beta.openai.com/).")

    # Check if the API key is entered
    if api_key:
    # Manually set the OPENAI_API_KEY environment variable
        os.environ["OPENAI_API_KEY"] = api_key
    # Show a toast message saying API key is successfully entered
        st.sidebar.success("API key successfully submitted!")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDF Files :open_file_folder:")

    # Display description
    if not st.session_state.get("api_request_made", False):
        st.markdown("""
            Welcome to chat with your PDFs. Here you can upload a PDF file and ask questions about that document,
            summarize the document into a couple of sentences, or ask specific questions. 
            AI-MATE will use the PDF uploaded and answer any questions to the best of its ability. 
            AI-MATE doesn't store any information, everything is uploaded on your local machine.
            """)
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            st.toast("Document Uploaded")
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)

            # create vector store
            vectorstore = get_vectorstore(text_chunks)

            # create conversation chain
            st.session_state.conversation = get_conversation_chain(
                vectorstore)
            st.session_state.api_request_made = True  # Set flag to hide description
    user_question = st.text_input("Ask a question about your documents:")

    # Chat now button
    chat_button = st.button("Chat now")

    if chat_button and user_question:
        st.session_state.api_request_made = True  # Set flag to hide description
        handle_userinput(user_question)

    
    


if __name__ == '__main__':
    pdf_chat_main()
