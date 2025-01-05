from typing import TypedDict
import aiohttp
import math

from bs4 import BeautifulSoup, Tag


async def extract_html(url: str) -> str:
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.get(url) as response:
            if response.status // 100 != 2:
                raise Exception(
                    f"failed to fetch `{url}` with status `{response.status}`"
                )

            text = await response.text()
            soup = BeautifulSoup(text, "html.parser")

    body = soup.body

    if body is None:
        return ""

    remove_html_tags(
        body,
        ["script", "style", "template", "header", "nav", "footer", "aside", "form"],
    )

    stats = compute_text_density(body)

    # find maximum density sum tag
    max_density_sum = max(stats.values(), key=lambda x: x["density_sum"])

    # find minimum density in the path from the maximum density sum tag to the body
    densities_in_path = []

    if max_density_sum["tag"].name != "body":
        for node in max_density_sum["tag"].parents:
            densities_in_path.append(stats[id(node)]["density"])

            if node.name == "body":
                break

    min_density = (
        min(densities_in_path) if densities_in_path else stats[id(body)]["density"]
    )

    threshold = max(min_density, 1)
    contents = set()

    def extract_content(stat: TagStat):
        if stat["density"] < threshold:
            return

        maximum = find_max_density_sum_tag(stat)
        contents.add(id(maximum["tag"]))

        if stat["tag"].string is not None:
            return

        for child in stat["tag"].children:
            if child is not maximum["tag"]:
                extract_content(stats[id(child)])

    def find_max_density_sum_tag(stat: TagStat) -> TagStat:
        if stat["tag"].string is not None:
            return stat

        element = max(
            stat["tag"].children,
            key=lambda x: stats[id(x)]["density_sum"],
        )

        if element is None:
            return stat

        return stats[id(element)]

    extract_content(stats[id(body)])

    tags = [stats[tag] for tag in contents]
    tags.sort(key=lambda x: x["order"])
    contents = [" ".join(tag["tag"].stripped_strings) for tag in tags]
    content = " ".join(contents)

    return content


def remove_html_tags(dom: Tag, tags: list[str]):
    for tag in tags:
        for node in dom.find_all(tag):
            node.decompose()


class TagStat(TypedDict):
    tag: Tag
    density: float
    density_sum: float
    tag_count: int
    text_length: int
    link_tag_count: int
    link_text_length: int
    order: int


def compute_text_density(body: Tag) -> dict[int, TagStat]:
    if body.string is not None:
        stats = {
            id(body): TagStat(
                tag=body,
                density=0,
                density_sum=0,
                tag_count=0,
                text_length=len(body.string),
                link_tag_count=0,
                link_text_length=0,
                order=0,
            )
        }
    else:
        stats = {
            id(body): TagStat(
                tag=body,
                density=0,
                density_sum=0,
                tag_count=0,
                text_length=0,
                link_tag_count=0,
                link_text_length=0,
                order=0,
            )
        }
        collect_tag_stats(body, stats)

    body_lc = stats[id(body)]["link_text_length"]
    body_c = stats[id(body)]["text_length"]
    body_lc_c = body_lc / max(1, body_c)

    def td(stat: TagStat):
        t = max(1, stat["tag_count"])
        c = stat["text_length"]
        density = c / max(1, t)

        stat["density"] = density
        stat["density_sum"] = density

    def ctb(stat: TagStat):
        t = max(1, stat["tag_count"])
        c = stat["text_length"]
        lt = stat["link_tag_count"]
        lc = stat["link_text_length"]

        ct = c / max(1, t)
        log_base = math.log(c / max(1, c - lc) * lc + body_lc_c * c + math.e)
        log_arg = c / max(1, lc) * t / max(1, lt)

        density = ct * math.log(max(1, log_arg), max(2, log_base))

        stat["density"] = density
        stat["density_sum"] = density

    for stat in stats.values():
        if stat["tag"].name == "a":
            ctb(stat)
        else:
            td(stat)

    def compute_density_sum(node: Tag):
        stats[id(node.parent)]["density_sum"] += stats[id(node)]["density"]

        for child in node.children:
            if child.string is not None:
                continue

            compute_density_sum(child)

    for child in body.children:
        if child.string is not None:
            continue

        compute_density_sum(child)

    return stats


def collect_tag_stats(node: Tag, stats: dict[int, TagStat]):
    for child in node.children:
        if child.string is not None:
            has_child = False
            text_length = len(child.string.strip())
        else:
            has_child = True
            text_length = sum(
                len(text.string.strip())
                for text in child.children
                if text.string is not None
            )

        if child.name == "a":
            link_tag_count = 1
            link_text_length = text_length
        else:
            link_tag_count = 0
            link_text_length = 0

        stats[id(child)] = TagStat(
            tag=child,
            density=0,
            tag_count=0,
            text_length=text_length,
            link_tag_count=link_tag_count,
            link_text_length=link_text_length,
            order=len(stats),
        )

        for parent in child.parents:
            parent_density = stats[id(parent)]
            parent_density["tag_count"] += 1
            parent_density["text_length"] += text_length
            parent_density["link_tag_count"] += link_tag_count
            parent_density["link_text_length"] += link_text_length

            if parent.name == "body":
                break

        if not has_child:
            continue

        collect_tag_stats(child, stats)
