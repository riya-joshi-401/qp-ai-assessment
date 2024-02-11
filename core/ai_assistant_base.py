from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate

class AIAssistant:

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def document_loader(self,file_path):

        const_path='test_data/'

        # reading the document
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(const_path+file_path)
            docs = loader.load()
        elif file_path.endswith('.docx') or file_path.endswith('.doc'):
            loader = Docx2txtLoader(const_path+file_path)
            docs = loader.load()
        elif file_path.endswith('.txt'):
            loader = TextLoader(const_path+file_path)
            docs = loader.load()

        # splitting the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
        doc_chunks = text_splitter.split_documents(docs)

        return doc_chunks

    def contextual_chatbot(self,file_path,query):

        # document splitter
        doc_chunks = self.document_loader(file_path)

        # embedding
        embeddings = OpenAIEmbeddings()

        # vectorstore DB
        db = FAISS.from_documents(doc_chunks, embeddings)

        # retrieving top 3 results
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # adding basic instructions in prompt
        with open('core/prompt.txt', "r") as file:
            custom_template = file.read()

        sys_prompt = PromptTemplate(
            template=custom_template, input_variables=["chat_history", "question"]
        )

        # creating Conversational Q&A chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-1106'),
            retriever=retriever,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": sys_prompt}
        )

        # initializing chat history
        chat_history = []

        # invoke qa chain
        result = qa_chain.invoke({"question": query, "chat_history": chat_history})

        # append current question , answer in chat history
        chat_history.append((query, result["answer"]))

        # return final answer with top 3 most similar chunks
        return result['answer'] , result['source_documents']








