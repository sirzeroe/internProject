from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# 1. Reconnect to the Local Embedding Model & Database
print("Loading Database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. Connect to Local Ollama
print("Waking up Gemma 2B...")
llm = Ollama(model="gemma:2b")

# 3. Design the Strict RAG Prompt
template = """You are a Senior IT Support Agent. Use the following historical ticket logs to answer the user's question. 
If the answer is not contained in the logs, you must say "I do not have enough information in the historical tickets to answer that." 
Do not guess or hallucinate.

Historical Tickets:
{context}

User Question: {question}

Resolution:"""
prompt = PromptTemplate.from_template(template)

def ask_the_intern(question):
    print(f"\nSearching database for: '{question}'...")
    
    docs = vectorstore.similarity_search(question, k=6)
    context = "\n\n".join([doc.page_content for doc in docs])
    formatted_prompt = prompt.format(context=context, question=question)
    
    print("\n--- The Intern is typing ---\n")
    response = llm.invoke(formatted_prompt)
    print(response)
    print("\n---------------------------\n")

if __name__ == "__main__":
    print("\n--- IT Support Assistant Ready (Type 'exit' to quit) ---")
    while True:
        user_input = input("How can I help you? > ")
        
        if user_input.lower() == 'exit':
            print("Shutting down the Assistant. Good luck with the interview prep!")
            break
            
        ask_the_intern(user_input)