from typing import Any, Dict

from scholarly import scholarly

NO_AUTHOR_FOUND = dict(name="", citation=0)


def get_authors_citations(
    author: str,
) -> Dict[str, Any]:
    search_query = scholarly.search_author(author)

    # Retrieve the first result from the iterator
    try:
        first_author_result = next(search_query)
    except StopIteration:
        first_author_result = None

    if first_author_result is None:
        return NO_AUTHOR_FOUND
    elif first_author_result.get("container_type") != "Author":
        return NO_AUTHOR_FOUND
    else:
        return dict(
            name=first_author_result.get("name", ""),
            citation=first_author_result.get("citedby", 0),
        )


if __name__ == "__main__":
    author = get_authors_citations("Serge-Olivier Paquette")
    print(author)
