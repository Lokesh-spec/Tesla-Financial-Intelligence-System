from langchain_openai import ChatOpenAI


def build_prompt(context, question):
    return f"""
                You are a financial analyst.

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


def generate_answer(context, question):
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    prompt = build_prompt(context, question)
    response = llm.invoke(prompt)

    return response.content