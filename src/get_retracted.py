import requests
import json
import time
import pandas as pd

def get_paper(doi):
    return requests.get(f'http://api.semanticscholar.org/v1/paper/{doi}?include_unknown_references=true')

def load_retracted(path='data/pubmed_retracted.csv'):
    return pd.read_csv(path)

def get_doi(descr):
    if 'doi' in descr:
        doi_idx = descr.index('doi:') if 'doi:' in descr else descr.index('doi/')
        doi_end = descr.index('. ', doi_idx)
        return descr[doi_idx+4:doi_end].strip()
    
def write_retracted(source='data/pubmed_retracted.csv', path='data/retracted'):
    df = load_retracted(source)
    dois = df['Description'].apply(get_doi)
    papers=''
    for doi in dois:
        if doi:
            papers += get_paper(doi).text
    with open(path, 'w') as f:
        f.write(papers)

def gen_retracted(source, **kwargs):
    with open(source, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if 'retracted' in entry['title'][:10].lower() \
                or any(entry[key] in values for key, values in kwargs.items()):
                yield entry

            