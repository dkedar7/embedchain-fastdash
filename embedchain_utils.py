import os
from string import Template
from embedchain import App
from embedchain.config import LlmConfig

def generate_response(web_page_urls, youtube_urls, pdf_urls, text, query, chat_history):
    "Use Embedchain's LLM capabilities to generate a response using the given query and chat history"

    # Define a custom app
    app = App()

    # Add documents
    try:
        if web_page_urls:
            [app.add(url) for url in web_page_urls]

        if youtube_urls:
            [app.add(url) for url in youtube_urls]

        if pdf_urls:
            [app.add(url) for url in pdf_urls]

        if text:
            app.add(text)

    except Exception as e:
        print(str(e))
        answer_suffix = "I couldn't analyze some sources. If you think this is an error, please try again later or make a suggestion [here](https://github.com/dkedar7/embedchain-fastdash/issues)."


    # Define app configurations
    PROMPT = Template(
    f"""Use the given context to answer the question at the end. Display the answer and use inline numbered citations to cite your sources. Display as markdown text.
    If the given context doesn't contain the answer, say "The given documents don't contain the answer."
    Our previous conversation is given below.

    Context: $context

    Conversation history: { f"{os.linesep} {os.linesep}".join([f"Me: {conv[0]}{os.linesep}You: {conv[1]}" for conv in chat_history[:-2]])}

    Question: $query

    Answer:"""
    )

    query_config = LlmConfig(
    template=PROMPT, number_documents=5, max_tokens=2000, model="gpt-4"
    )

    answer = app.query(query, config=query_config)

    return answer