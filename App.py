import streamlit as st
import tempfile
import os
import base64
# from PyPDF2 import PdfReader
import time

import base64
import io
import fitz
from PIL import Image

from openai import OpenAI

endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

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

        if not api_key:
            st.warning('Please enter your credentials!', icon = '⚠️')
        else:
            st.success("Proceed to use the app!", icon = '✅')

        st.subheader('Parameters')

        chunk_size = st.sidebar.slider(':blue[Chunk Size]', min_value=50, max_value=1000, value = 200, step = 10, help = 'Determines the size of each chunk that the text will be split into.' , disabled=not api_key)
        chunk_overlap = st.sidebar.slider(':blue[Chunk Overlap]', min_value=0, max_value=100, value=20, step=10, help = 'This parameter determines the number of tokens that will overlap between each chunk.', disabled=not api_key)

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

            pdf_document = fitz.open(path)
            page = pdf_document.load_page(1 - 1)  # input is one-indexed
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")

            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Large Language Model
            client = OpenAI(
                base_url=endpoint,
                api_key=api_key,
            )

            query = question

            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that describes images in details.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": query,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                },
                            },
                        ],
                    },
                ],
                model=model_name,
            )

            # stream response
            def stream_data():
                for word in response.choices[0].message.content.split(" "):
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




