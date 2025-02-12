import json

from bs4 import BeautifulSoup
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from llms.agent_detail.extract_html.utils import DEFAULT_REMOVE_TAGS, remove_html_tags


async def extract_html_metadata(
    url: str,
    html: str,
    llm: BaseChatModel,
) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    head = soup.head
    body = soup.body

    if head is None or body is None:
        return {}

    title = head.title.string if head.title else "<no title>"
    meta_description = soup.find("meta", attrs={"name": "description"})
    description = (
        "<no description>"
        if meta_description is None
        else meta_description.get("content", "<no description>")
    )

    remove_html_tags(
        body,
        DEFAULT_REMOVE_TAGS,
    )

    content = ""

    for chunk in body.stripped_strings:
        content += chunk
        content += " "

        if 1024 <= len(content):
            break

    metadata = {
        "url": url,
        "title": title,
        "description": description,
        "content": content.strip(),
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                """
You are tasked with analyzing web page metadata and generating refined metadata based on the given information. You will be provided with the following input:

<metadata>
{metadata}
</metadata>

Your goal is to deduce and generate refined page metadata by analyzing the provided information. Pay close attention to the URL structure, title, description, and content preview to infer additional details about the web page.

Generate the following required metadata fields:
1. Category (e.g., post, article, news, product, notice, etc.)
2. Platform (e.g., Naver Cafe, Steam, ArcaLive, Reddit, etc.)

In addition to these required fields, you should generate any other relevant metadata that you can deduce from the given information. These may include, but are not limited to:
- Product name (if applicable)
- Price (if applicable)
- Published date
- Author or source
- Language
- Main topic or theme
- Content type (e.g., text, video, image gallery)
- Target audience

Provide your analysis and generated metadata in valid JSON, as the following format:

{{
  "field_name_in_snake_case": "[field value in string]",
  "field_name_in_snake_case": "[field value in string]",
  ...
}}

For example:

<example>
{{
  "category": "product",
  "platform": "amazon",
  "product_name": "Wireless Bluetooth Headphones",
  "price": "$49.99",
  "language": "english",
  "released_at": "2025-01-06T13:35:45.475Z",
  "color": "black",
  "bluetooth_version": "5.0"
}}
</example>

Be creative and thorough in your analysis, but ensure that all deduced information is reasonably supported by the provided input. If you're unsure about a particular field, you may omit it.

Remember, you are allowed to provide as many metadata fields as you can deduce, but do not exceed 10 fields in total (including the required category and platform fields).

Prefer lowercase for enum values such as category, platform, etc, as they should be de-duplicated.

Never include explanation or comment in your response.
                """.strip(),
            ),
            ("ai", "{prefix}"),
        ]
    )
    messages = prompt.format_messages(metadata=metadata, prefix="{")
    response = await llm.ainvoke(messages)
    refined_metadata = (
        json.loads(response.content)
        if response.content.startswith("{")
        else json.loads(f"{'{'}{response.content}")
    )

    return {
        "url": url,
        "title": title,
        "description": description,
        **refined_metadata,
    }
