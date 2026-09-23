# -*- coding: utf-8 -*-
"""Annual PubMed counts for the two survey literatures, 2003 to 2026, on one date.

Behind Supplementary Figure S1 and Table S10.3. Why this replaced the earlier 30 July 2026
series (2004 to 2025):

1. Window. The title-level literature mapping (Supplementary Appendix S10) covers
   2003 to 2026, and the growth series covered 2004 to 2025, so the two totals
   quoted in the paper were cut on different timelines.

2. A double count. The NHANES query was
       NHANES[tiab] OR "National Health and Nutrition Examination Survey"[tiab]
   and PubMed's phrase match finds that full name inside "Korea National Health and
   Nutrition Examination Survey". Every KNHANES paper that spells out its survey was
   therefore counted as an NHANES paper too: 5,529 of the 5,860 KNHANES records in
   2003 to 2026. A random sample of 20 of the 2016 overlap records were all Korean-only
   studies, none mentioning the US survey. The NHANES line of the earlier figure was inflated by
   up to 40% in the mid-2010s (2016: 1,375 published, 985 corrected), and the
   "35,482 NHANES and 5,069 KNHANES, about 40,000" total counted most KNHANES papers
   twice.

   The corrected NHANES query keeps the acronym, which never matches KNHANES, and keeps
   the full name only where the record is not the Korean survey:
       NHANES[tiab] OR ("National Health and Nutrition Examination Survey"[tiab] NOT
         (KNHANES[tiab] OR "Korea National ..."[tiab] OR "Korean National ..."[tiab]))
   A genuinely cross-national paper names the US survey by its acronym and so stays in
   both series; those are the only records the two series now share.

The union of the two queries is unchanged by the correction, because every record the
correction removes from NHANES is already a KNHANES record. That deduplicated union is
the size of the two literatures and is the single number the paper should quote.

2026 is a partial year, counted to the query date, and is marked as such wherever it
is drawn.

Writes publication_counts_2003_2026.csv and publication_counts_2003_2026.queries.json
next to this script.
"""
import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
KOREAN = ('KNHANES[tiab] OR "Korea National Health and Nutrition Examination Survey"[tiab] '
          'OR "Korean National Health and Nutrition Examination Survey"[tiab]')
QUERIES = {
    "NHANES": ('NHANES[tiab] OR ("National Health and Nutrition Examination Survey"[tiab] '
               f'NOT ({KOREAN}))'),
    "KNHANES": KOREAN,
}
QUERIES["union"] = f'({QUERIES["NHANES"]}) OR ({QUERIES["KNHANES"]})'
QUERIES["both"] = f'({QUERIES["NHANES"]}) AND ({QUERIES["KNHANES"]})'
# the uncorrected query, kept so the size of the double count is on record
QUERIES["NHANES_published"] = ('NHANES[tiab] OR '
                               '"National Health and Nutrition Examination Survey"[tiab]')


def count(term):
    url = (E + "?db=pubmed&retmode=json&retmax=0&tool=knhanes-nhanes-platform&term="
           + urllib.parse.quote(term))
    for attempt in range(5):
        try:
            return int(json.load(urllib.request.urlopen(url, timeout=60))
                       ["esearchresult"]["count"])
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(term)


def main():
    rows = []
    for y in range(2003, 2027):
        r = {"year": y}
        for k, q in QUERIES.items():
            r[k] = count(f"({q}) AND {y}[dp]")
            time.sleep(0.35)
        rows.append(r)
        print(r, flush=True)
    d = pd.DataFrame(rows)
    d.attrs["query_date"] = dt.date.today().isoformat()
    d.to_csv(os.path.join(HERE, "publication_counts_2003_2026.csv"), index=False)
    with open(os.path.join(HERE, "publication_counts_2003_2026.queries.json"), "w",
              encoding="utf-8") as f:
        json.dump({"query_date": dt.date.today().isoformat(), "queries": QUERIES,
                   "window": "2003 to 2026 by [dp]; 2026 partial to the query date"},
                  f, indent=2, ensure_ascii=False)
    s = d.drop(columns="year").sum()
    print("\n2003-2026 sums:", s.to_dict())
    print(f"double count removed from NHANES: {s.NHANES_published - s.NHANES:,}")
    assert s.union == s.NHANES + s.KNHANES - s.both, "union identity failed"


if __name__ == "__main__":
    main()
