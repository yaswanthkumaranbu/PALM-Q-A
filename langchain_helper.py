from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import csv
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)

# Create Google Palm LLM model
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
# Initialize embeddings using the Hugging Face model
embeddings = HuggingFaceEmbeddings()

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
        1: 'CEO',
        2: 'VICE PRESIDENT'
    }
    return user_roles.get(user_id, 'CEO')

# Function to load all data
def load_data():
    loader = CustomCSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    return loader.load()

# Function to create a single vector database with role-based indexing
def create_vector_db():
    data = load_data()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local("faiss_index")

# Function to get QA chain with role-based filtering
def get_qa_chain(user_id):
    user_role = get_user_role(user_id)
    vectordb_file_path = "faiss_index"

    # docs=list(vectordb.docstore._dict.values())
    knowledge_base=FAISS.load_local(vectordb_file_path,embeddings)
    d=knowledge_base.docstore._dict
    del_list=[]
    for key,doc in d.items():
        if doc.metadata['role']!=user_role:
            del_list.append(key)
    knowledge_base.delete(del_list)
    # docs=list(knowledge_base.docstore._dict.values())
    # print(docs)

    retriever = knowledge_base.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context from a document relevant to the 'role' and a question, generate an answer based on this context only.
In the answer, copy and paste the exact answer from the 'response' section, or state 'I don't know' if no answer is found in this document. 
Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        return_source_documents=True,
                                        input_key="query",
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

# if __name__ == "_main_":
#     create_vector_db()
#     user_id = 1  # Example user ID
#     chain = get_qa_chain(user_id)
#     result = chain({"question": "Do you have a JavaScript course?"})
#     print(result)