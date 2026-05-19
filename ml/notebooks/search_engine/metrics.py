def hits_and_rrs(df, col):
    hits = []
    rrs = []
    for _, row in df.iterrows():
        ids = row['article_id']
        res = row[f"{col}_res"]

        hit = 0
        rr = 0
        for i, idx in enumerate(res, start=1):
            if idx in ids:
                hit = 1
                rr = 1/i
                break
        hits.append(hit)
        rrs.append(rr)
    return hits, rrs

def failure_rate(df, col):
    failures = []
    for _, row in df.iterrows():
        ids = row['article_id']
        res = row[f"{col}_res"]

        fail = 1
        for idx in res:
            if idx in ids:
                fail = 0
                break
        failures.append(fail)
    return failures
    