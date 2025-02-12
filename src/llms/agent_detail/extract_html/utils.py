from bs4 import Tag


DEFAULT_REMOVE_TAGS = [
    "script",
    "noscript",
    "style",
    "template",
    "header",
    "nav",
    "footer",
    "aside",
    "form",
    "iframe",
    "video",
    "audio",
    "picture",
    "source",
    "img",
    "svg",
    "canvas",
]


def remove_html_tags(dom: Tag, tags: list[str]):
    for tag in tags:
        for node in dom.find_all(tag):
            node.decompose()
