from bs4 import Tag


def remove_html_tags(dom: Tag, tags: list[str]):
    for tag in tags:
        for node in dom.find_all(tag):
            node.decompose()
