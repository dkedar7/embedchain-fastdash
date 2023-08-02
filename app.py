import os
from fast_dash import FastDash, Fastify, dcc, dmc

from embedchain import App
from embedchain.config import QueryConfig
from string import Template

# Define app configurations
PROMPT = Template(
    """Use the given context to answer the question at the end.
If you don't know the answer, say so, but don't try to make one up.
At the end of the answer, also give the sources as a bulleted list.
Display the answer as markdown text.

Context: $context

Query: $query

Answer:"""
)
query_config = QueryConfig(template=PROMPT, number_documents=5, max_tokens=2000, model="gpt-4")

# Define components
openai_api_key_component = dmc.PasswordInput(
        placeholder="API Key",
        description="Get yours at https://platform.openai.com/account/api-keys",
        required=True
    )

web_page_urls_component = dmc.MultiSelect(
        description="Include all the reference web URLs",
        placeholder="Enter URLs separated by commas",
        searchable=True,
        creatable=True,
    )

text_component = dmc.Textarea(placeholder="Write your query here", autosize=True, minRows=4, description="Any additional information that could be useful")

query_component = dmc.Textarea(placeholder="Write your query here", autosize=True, minRows=4, required=True, description="Write your query here")

answer_component = dcc.Markdown(style={"text-align": "left", "padding": "1%"}, link_target="_blank")


def explore_your_knowledge_base(
    openai_api_key: openai_api_key_component,
    web_page_urls: web_page_urls_component,
    youtube_urls: web_page_urls_component,
    pdf_urls: web_page_urls_component,
    text: text_component,
    query: text_component,
) -> answer_component:
    """
    Input your sources and let GPT4 find answers. Built with Fast Dash.
    This app uses embedchain.ai, which abstracts the entire process of loading and chunking datasets, creating embeddings, and storing them in a vector database.
    Embedchain itself uses Langchain and OpenAI's ChatGPT API.
    """
    os.environ["OPENAI_API_KEY"] = openai_api_key

    try:
        app = App()

        if not openai_api_key:
            return "Did you forget adding your OpenAI API key? If you don't have one, you can get it [here](https://platform.openai.com/account/api-keys)."

        if not query:
            return "Did you forget writing your query in the query box?"

        if web_page_urls:
            [app.add("web_page", url) for url in web_page_urls]

        if youtube_urls:
            [app.add("youtube_video", url) for url in youtube_urls]

        if pdf_urls:
            [app.add("pdf_file", url) for url in pdf_urls]

        if text:
            app.add_local("text", text)

        answer = app.query(query, query_config)

    except Exception as e:
        answer = "Oops, something went wrong! Please try again later or make a suggestion [here](https://github.com/dkedar7/embedchain-fastdash/issues)."

    return answer

# Build app (this is all it takes!). Fast Dash understands what it needs to do. 
app = FastDash(explore_your_knowledge_base, github_url="https://github.com/dkedar7/embedchain-fastdash")
server = app.server

if __name__ == "__main__":
    app.run()