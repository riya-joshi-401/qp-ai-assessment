from dotenv import load_dotenv
load_dotenv()
from core.ai_assistant_base import AIAssistant
import streamlit as st

def main():

    st.title("QuestionPro DocChat")

    # File uploader allows user to add a file
    uploaded_file = st.file_uploader("Upload your document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    query = st.text_input("Ask a question")

    # button to process the input
    if st.button("Submit"):
        if uploaded_file is not None:
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        else:
            st.error("Please upload a file.")

        # get answer
        with AIAssistant() as obj:
            answer,top_3 = obj.contextual_chatbot(uploaded_file.name,query)
            print(top_3)

        # display output
        st.write("#### Answer")
        st.text_area("",value=answer, height=200)

        # display top 3 answers on the sidebar
        if any(top_3):
            st.sidebar.header("Top 3 Answers !")
            for i in range(3):
                st.sidebar.write(f"{i}: {top_3[i].page_content}")


if __name__ == "__main__":
    main()
