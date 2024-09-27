from langchain.prompts import PromptTemplate
prompt_template = """
You are an AI assistant. Use the following context to answer questions. 
Use the provided query and chat history to improve the accuracy and relevance of your answers.

Context: {context}

history: {history}

Query: {input}

Provide a clear and concise answer based on the given information.
"""

# PROMPT = PromptTemplate(
#     template=prompt_template, 
#     input_variables=["context", "question",]
#     )

contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""