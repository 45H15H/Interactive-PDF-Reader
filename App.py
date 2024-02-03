import streamlit as st
import tempfile
import os


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

col1, col2 = st.columns(spec=[0.5, 0.5], gap="medium")

with col1:
    # Subheader
    st.header("Interactive Reader :open_book:")

    st.text_input(label="Ask a question from the PDF:", value=None, max_chars=None, key="question", type="default", on_change=None, disabled=False, label_visibility="visible")



    with st.container(border=True):
        st.subheader("Your Documents")
        uploaded_files = st.file_uploader(label="Upload your PDF here and click on 'Process'",
                               type=["pdf"],
                               accept_multiple_files=True
                            #    disabled=not api_key
                               )

import base64

def display_pdf(file_path):
  with open(file_path, "rb") as f:
    pdf_data = f.read()
  base64_encoded_pdf = base64.b64encode(pdf_data).decode("utf-8")
  pdf_display = f'<iframe src="data:application/pdf;base64,{base64_encoded_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
  st.markdown(pdf_display, unsafe_allow_html=True)
with col2:

    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            fd, path = tempfile.mkstemp()
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(uploaded_file.getvalue())

            # Now 'path' variable holds the path of the uploaded file on your server
            st.write(f'File saved to: {path}')

            # Convert the uploaded file to a data URL
            with open(path, "rb") as f:
                data_url = base64.b64encode(f.read()).decode('utf-8')
                st.markdown(f'<iframe src="data:application/pdf;base64,{data_url}" width="600" height="800" type="application/pdf"></iframe>', unsafe_allow_html=True)

            # Delete the temporary file when done
            os.remove(path)


