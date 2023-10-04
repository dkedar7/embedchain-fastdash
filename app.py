import os
from fast_dash import FastDash, dcc, dmc, Chat
from flask import session

from embedchain_utils import generate_response

# Define components
openai_api_key_component = dmc.PasswordInput(
    placeholder="API Key",
    description="Get yours at https://platform.openai.com/account/api-keys",
    required=True,
)

web_page_urls_component = dmc.MultiSelect(
    description="Include all the reference web URLs",
    placeholder="Enter URLs separated by commas",
    searchable=True,
    creatable=True,
)

text_component = dmc.Textarea(
    placeholder="Write your query here",
    autosize=True,
    minRows=4,
    description="Any additional information that could be useful",
)

query_component = dmc.Textarea(
    placeholder="Write your query here",
    autosize=True,
    minRows=4,
    required=True,
    description="Write your query here",
)

answer_component = dcc.Markdown(
    style={"text-align": "left", "padding": "1%"}, link_target="_blank"
)


def explore_your_knowledge_base(
    openai_api_key: openai_api_key_component,
    web_page_urls: web_page_urls_component,
    youtube_urls: web_page_urls_component,
    pdf_urls: web_page_urls_component,
    text: text_component,
    query: text_component,
) -> Chat:
    """
    Input your sources and let GPT4 find answers. Built with Fast Dash.
    This app uses embedchain.ai, which abstracts the entire process of loading and chunking datasets, creating embeddings, and storing them in a vector database.
    Embedchain itself uses Langchain and OpenAI's ChatGPT API.
    """
    answer_suffix = ""

    if not openai_api_key:
        answer = "Did you forget adding your OpenAI API key? If you don't have one, you can get it [here](https://platform.openai.com/account/api-keys)."

    elif not query:
        answer = "Did you forget writing your query in the query box?"
    
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Get chat history from Flask session
        chat_history = session.get("chat_history", [])

        # Generate a response
        answer = generate_response(web_page_urls, youtube_urls, pdf_urls, text, query, chat_history)

        # Save chat history back to the session cache
        chat_history.append([query, answer])
        session["chat_history"] = chat_history

        answer = f"""{answer}

        {answer_suffix}
        """

    chat = dict(query=query, response=answer)

    return chat


# Build app (this is all it takes!). Fast Dash understands what it needs to do.
app = FastDash(
    explore_your_knowledge_base,
    github_url="https://github.com/dkedar7/embedchain-fastdash",
)
server = app.server
server.config["SECRET_KEY"] = "Some key"

if __name__ == "__main__":
    app.run()