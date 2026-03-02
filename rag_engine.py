import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables (API Keys)
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits the massive text into smaller, overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Converts text chunks into vector embeddings and saves them locally."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def process_user_query(user_question):
    """Handles the user query using modern LCEL (LangChain Expression Language)."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the local vector database
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform Similarity Search to get relevant documents
    docs = new_db.similarity_search(user_question, k=4)
    
    # Extract text from documents to pass as context
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # Modern LCEL Prompt Template
    prompt_template = """
    You are DocuMind AI, an enterprise Retrieval-Augmented Generation assistant.
    Answer the user's question as detailed as possible using ONLY the provided context. 
    If the answer is not in the provided context, DO NOT guess or hallucinate. Just politely state: 
    "I cannot find the answer to this in the uploaded documents."
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Detailed Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Initialize Google Gemini
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # Build the modern LCEL Chain (Prompt -> LLM -> String Output)
    chain = prompt | model | StrOutputParser()
    
    # Invoke the chain
    response = chain.invoke({
        "context": context_text,
        "question": user_question
    })
    
    return response