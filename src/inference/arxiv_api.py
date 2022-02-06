import arxiv
import pandas as pd


def get_arxiv_articles(
    query: str = 'all:"real-time bidding" OR all:"online advertisment" cat:cs',
    time_delta: pd.Timedelta = pd.to_timedelta("7D")
):
    search = arxiv.Search(
        query=query,
        max_results=100,
        sort_by=arxiv.SortCriterion.LastUpdatedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = search.results()
    today = pd.Timestamp.now(tz="UTC").floor("D")

    results = filter(
        lambda r: (today - time_delta) <= pd.to_datetime(r.published).floor("D") <= today,
        results,
    )
    return results

if __name__ == "__main__":
    results = get_arxiv_articles()

    for idx, r in enumerate(results):
        print(idx, r)
        print(r.title)
        print(r.published)
        print(r.summary)

        for a in r.authors:
            print(a.name)

        print()
