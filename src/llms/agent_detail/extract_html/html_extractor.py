import asyncio

import aiohttp
from langchain_core.language_models import BaseChatModel

from llms.agent_detail.extract_html.extract_html_content import extract_html_content
from llms.agent_detail.extract_html.extract_html_metadata import extract_html_metadata


async def extract_html(url: str, llm: BaseChatModel) -> tuple[str, dict[str, str]]:
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            },
        ) as response:
            if response.status // 100 != 2:
                raise Exception(
                    f"failed to fetch `{url}` with status `{response.status}`"
                )

            html = await response.text()

    content, metadata = await asyncio.gather(
        extract_html_content(html), extract_html_metadata(url, html, llm)
    )

    return content, metadata
