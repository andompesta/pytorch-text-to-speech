import arxiv
from datetime import datetime, timedelta
from pytz import UTC
from collections import defaultdict
from scholarly import scholarly


def floor(datetime_obj: datetime) -> datetime:
    return datetime_obj.replace(second=0, hour=0, minute=0)


if __name__ == "__main__":
    max_returns = 150
    max_podcast_size = 10
    now = datetime.now().astimezone(UTC)
    date = floor(now - timedelta(1))
    search_query = "cat:cs.LG"

    search = arxiv.Search(
        query=search_query,
        max_results=150,
        sort_by=arxiv.SortCriterion.LastUpdatedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = search.results()
    results = filter(
        lambda entity: entity.updated.astimezone(UTC) > date,
        results,
    )
    results = list(results)
    print("results size: {}".format(len(results)))

    # get authors citations count
    author_query_template = "AUTHLAST({last_name}) and AUTHFIRST({first_name})"
    authors_citations = defaultdict(int)
    authors_info = dict()

    for idx, result in enumerate(results):
        for author in result.authors:
            author_search_query = scholarly.search_author(author.name)
            try:
                author_info = scholarly.fill(next(author_search_query))

                author_info[author.name] = author_info
                authors_citations[author.name] = author_info.get("citedby", 0)
            except StopIteration:
                print("WARNING: author {} is not present on gScholar".format(author.name))

        score = sum([authors_citations.get(author.name, 0) for author in result.authors])
        score = score / len(result.authors)
        result.score = score
        results[idx] = result

    results = sorted(
        results,
        key=lambda r: r.score,
        reverse=True,
    )

    import json
    with open("results.json", "w") as writer:
        json.dump(results, writer)

    with open("authors.json", "w") as writer:
        json.dump(authors_info, writer)
    with open("authors_citations.json", "w") as writer:
        json.dump(authors_citations, writer)
