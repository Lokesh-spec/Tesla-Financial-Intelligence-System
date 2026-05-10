from langchain_openai import ChatOpenAI


def build_prompt(context, question, query_type):
    if query_type == "comparison":
        instruction = """Compare data across years strictly from context. 
        If data for any year is missing, say so. Never make up numbers."""
    elif query_type == "summary":
        instruction = """Provide a structured summary with key points. 
        Cover revenue, margins, and major business highlights if available."""
    else:
        instruction = """Answer the specific question concisely and precisely."""
    return f"""
                You are a financial analyst. {instruction}

                Answer ONLY using the provided context.

                If numerical values are given:
                - Convert them into human-readable format (millions or billions)
                - Clearly mention units

                If answer is not in context, say "I don't know".

                Context:
                {context}

                Question:
                {question}
            """


def generate_answer(context, question, query_type):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = build_prompt(context, question, query_type)
    response = llm.invoke(prompt)

    return response.content