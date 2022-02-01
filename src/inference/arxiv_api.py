import arxiv


def get_arxiv_abstract(
    query: str = "all:\"real-time bidding\" OR all:\"online advertisment\" cat:cs",
):
    search = arxiv.Search(
        query=query,
        max_results=10,
        sort_by=arxiv.SortCriterion.LastUpdatedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    return search.results()


if __name__ == "__main__":

    for r in get_arxiv_abstract():
        print(r)
        print(r.title)
        print(r.summary)
        for a in r.authors:
            print(a.name)
