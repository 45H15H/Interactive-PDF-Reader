import streamlit as st
import tempfile
import os
import base64
from PyPDF2 import PdfReader
import time


# Page configuation
st.set_page_config(
    page_title="Interactive Reader",
    page_icon="📚",
    layout="wide"
)

# Form to enter API key
with st.sidebar:
    with st.form(key='api_form'):
        st.markdown("""
        Enter your Gemini API key :red[*]
        """)
        api_key = st.text_input("Enter your Gemini API key:", type='password', key = 'token', label_visibility='collapsed')
        st.form_submit_button("SUBMIT",
                            #   disabled=not api_key,
                              use_container_width=True)
        st.caption(
        "To use this app, you need an API key. "
        "You can get one [here](https://ai.google.dev/)."
        )

        if not (api_key.startswith('AI') and len(api_key) == 39):
            st.warning('Please enter your credentials!', icon = '⚠️')
        else:
            st.success("Proceed to use the app!", icon = '✅')

        st.subheader('Parameters')

        chunk_size = st.sidebar.slider(':blue[Chunk Size]', min_value=50, max_value=1000, value = 200, step = 10, help = 'Determines the size of each chunk that the text will be split into.' , disabled=not api_key)
        chunk_overlap = st.sidebar.slider(':blue[Chunk Overlap]', min_value=0, max_value=100, value=20, step=10, help = 'This parameter determines the number of tokens that will overlap between each chunk.', disabled=not api_key)

# LangChain Training
# LLM
from langchain_google_genai import GoogleGenerativeAI as genai

# Document Loader
from langchain_community.document_loaders import PyPDFLoader

# Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Tokenizer
from transformers import GPT2TokenizerFast

# Embedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Vector DataBase
from langchain_community.vectorstores import Chroma


col1, col2 = st.columns(spec=[0.55, 0.45], gap="medium")

with col1:
    # Subheader
    st.header("Interactive Reader :open_book:")

    with st.expander("Your Documents"):
        uploaded_files = st.file_uploader(label="Upload your PDF here and click on 'Process'",
                               type=["pdf"],
                               accept_multiple_files=True,
                               disabled=not api_key
                               )

    question = st.text_input(label="Ask a question from the PDF:", value=None, max_chars=None, key="question", type="default", on_change=None, disabled=not uploaded_files, label_visibility="visible")

   
    if st.button("Process") and uploaded_files is not None:
        for uploaded_file in uploaded_files:
            # Save the uploaded file to a temporary location
            fd, path = tempfile.mkstemp()
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(uploaded_file.getvalue())
            # Now 'path' variable holds the path of the uploaded file on your server
            # st.write(f'File saved to: {path}')
            # Load content from the PDFs
            reader = PdfReader(path)
            pdf_data = ""
            for page_number in range(len(reader.pages)):
                page = reader.pages[page_number]
                pdf_data += page.extract_text()        
        
            # Large Language Model
            chat_llm = genai(model="gemini-pro", google_api_key=api_key, convert_system_message_to_human=True)

            # Split by chunks
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            def count_tokens(text: str) -> int:
                return len(tokenizer.encode(text))

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=20,
                length_function=count_tokens  # It uses len() by default.
            )

            texts = text_splitter.create_documents([pdf_data])
            pdf_chunks = text_splitter.split_documents(texts)
            # st.write("PDF Data - Now you have {0} chunks".format(len(pdf_chunks)))

            # Quick data visualization to ensure chunking was successful
            # import pandas as pd
            # import matplotlib.pyplot as plt

            # Create a list of token counts
            # token_counts = [count_tokens(chunk.page_content) for chunk in pdf_chunks]

            # Create a DataFrame from the token counts
            # df = pd.DataFrame({'Token Count': token_counts})

            # Create a histogram of the token count distribution
            # st.bar_chart(df)

            # Get embidding model
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=api_key
            )

            db_chroma = Chroma.from_documents(pdf_chunks, embeddings)

            from langchain.chains.question_answering import load_qa_chain

            query = question
            matches = db_chroma.similarity_search(query, k=5)
            # st.write(matches)

            chain = load_qa_chain(chat_llm, chain_type="stuff")

            response = chain.run(input_documents=matches, question=query)

            # stream response
            def stream_data():
                for word in response.split(" "):
                    yield word + " "
                    time.sleep(0.02)

            st.write_stream(stream_data)

            

with col2:
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            fd, path = tempfile.mkstemp()
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(uploaded_file.getvalue())

            # Now 'path' variable holds the path of the uploaded file on your server
            # st.write(f'File saved to: {path}')

            # Convert the uploaded file to a data URL
            with open(path, "rb") as f:
                data_url = base64.b64encode(f.read()).decode('utf-8')
                st.markdown(f'<iframe src="data:application/pdf;base64,{data_url}" width="500" height="720" type="application/pdf"></iframe>', unsafe_allow_html=True)

            # Delete the temporary file when done
            os.remove(path)




