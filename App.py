import streamlit as st
import tempfile
import os
import base64
import time

import base64
import io
import fitz
from PIL import Image

from openai import OpenAI

# Initialize session state for chat history if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

# Page configuration
st.set_page_config(
    page_title="Interactive Reader",
    page_icon="📚",
    layout="wide"
)

# Form to enter API key
with st.sidebar:
    with st.form(key='api_form'):
        st.markdown("""
        Enter your OpenAI API key :red[*]
        """)
        api_key = st.text_input("Enter your Gemini API key:", type='password', key = 'token', label_visibility='collapsed')
        st.form_submit_button("SUBMIT",
                            #   disabled=not api_key,
                              use_container_width=True)
        st.caption(
            "To use this app, you need an API key. "
            "You can get one [here](https://github.com/marketplace/models)."
        )

        if not api_key:
            st.warning('Please enter your credentials!', icon = '⚠️')
        else:
            st.success("Proceed to use the app!", icon = '✅')

    st.subheader('Parameters')

    temp = st.sidebar.slider(':blue[Temperature]', min_value=0.0, max_value=1.0, value = 0.5, step = 0.01, help = 'Controls randomness in the response, use lower to be more deterministic.' , disabled=not api_key)
    m_tokens = st.sidebar.slider(':blue[Max Tokens]', min_value=100, max_value=4096, value=4096, step=10, help = 'Limit the maximum output tokens for the model response.', disabled=not api_key)
    t_p = st.sidebar.slider(':blue[Top P]', min_value=0.01, max_value=1.0, value=1.0, step=0.01, help = 'Limit the maximum output tokens for the model response.', disabled=not api_key)

col1, col2 = st.columns(spec=[0.55, 0.45], gap="medium")

if 'total_pages' not in st.session_state:
    st.session_state.total_pages = 0

with col1:
    # Subheader
    st.header("Interactive Reader :open_book:")

    with st.expander("Your Documents"):
        uploaded_files = st.file_uploader(
            label="Upload your PDF here and click on 'Process'",
            type=["pdf"],
            accept_multiple_files=True,
            disabled=not api_key
        )
        
    # Get total pages when file is uploaded
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                st.session_state.total_pages = doc.page_count
            uploaded_file.seek(0)  # Reset file pointer
    
    if st.session_state.total_pages > 0:
        page_numbers = st.multiselect(
            "Select page numbers",
            options=list(range(1, st.session_state.total_pages + 1)),
            default=[1],
            help=f"Select page numbers between 1 and {st.session_state.total_pages}"
        )
    else:
        st.info("Upload a PDF to view pages")
    
    st.divider()

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

# Chat input
    if prompt := st.chat_input("Ask a question about the document", disabled=not uploaded_files):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message('user'):
            st.write(prompt)
        
        if uploaded_files:
            images = []
            for uploaded_file in uploaded_files:
                fd, path = tempfile.mkstemp()
                with os.fdopen(fd, 'wb') as tmp:
                    tmp.write(uploaded_file.getvalue())

                pdf_document = fitz.open(path)
                for page_number in page_numbers:
                    page = pdf_document.load_page(page_number - 1)
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    images.append(base64_image)

                # Initialize OpenAI client
                client = OpenAI(
                    base_url=endpoint,
                    api_key=api_key,
                )

                # Prepare conversation history
                conversation_history = [msg for msg in st.session_state.messages]

                # Generate response
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that describes images and maintains conversation context.",
                        },
                        *conversation_history,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                *[
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image}",
                                            "detail": "low"
                                        },
                                    }
                                    for image in images
                                ],
                            ],
                        },
                    ],
                    model=model_name,
                    temperature=temp,
                    max_tokens=m_tokens,
                    top_p=t_p
                )

                # Display response
                assistant_response = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                with st.chat_message("assistant"):
                    def stream_response():
                        for word in assistant_response.split():
                            yield word + " "
                            time.sleep(0.02)
                    # Display streaming response with markdown formatting
                    response_placeholder = st.empty()
                    full_response = ''

                    for chunk in stream_response():
                        full_response += chunk
                        response_placeholder.markdown(full_response)



            

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
