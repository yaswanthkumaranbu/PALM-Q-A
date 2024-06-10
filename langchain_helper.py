from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from langchain.schema import Document
import csv

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Create Google Palm LLM model
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
# # Initialize instructor embeddings using the Hugging Face model
embeddings = HuggingFaceEmbeddings()

vectordb_file_path = "faiss_index"



# Custom CSV loader to include role in metadata
class CustomCSVLoader(CSVLoader):
    def load(self):
        documents = []
        with open(self.file_path, 'r', encoding=self.encoding) as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                prompt = row.get(self.source_column)
                response = row.get('response')
                role = row.get('role')
                document = Document(
                    page_content=f"prompt: {prompt}\nresponse: {response}",
                    metadata={'source': prompt, 'role': role, 'row': i}
                )
                documents.append(document)
        return documents

# Function to get user role
def get_user_role(user_id):
    # Dummy data for demonstration
    user_roles = {
        1: 'employee',
        2: 'manager'
    }
    return user_roles.get(user_id, 'employee')

# Function to load and filter data based on role
def load_and_filter_data(user_role):
    loader = CustomCSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    data = loader.load()
    filtered_docs = [doc for doc in data if doc.metadata.get('role') == user_role]
    return filtered_docs

# Function to get data for a logged-in user
def get_data_for_user(user_id):
    user_role = get_user_role(user_id)
    filtered_data = load_and_filter_data(user_role)
    return filtered_data




def create_vector_db():
    # Load data from FAQ sheet
    user_id = 2  # Example user ID
    user_data = get_data_for_user(user_id)
    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=user_data,
                                    embedding=embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))