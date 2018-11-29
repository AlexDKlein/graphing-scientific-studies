import pandas as pd
import pymongo

def get_doi(descr):
    if 'doi' in descr:
        doi_idx = descr.index('doi:') if 'doi:' in descr else descr.index('doi/')
        doi_end = descr.index('. ', doi_idx)
        return descr[doi_idx+4:doi_end].strip()

def get_length(pages):
    if not pages: 
        return 0
    page_ranges = ''.join(x for x in pages if x in '0123456789-;')
    pg_cnt = 0
    for pgs in page_ranges.split(';'):
        if '-' in pgs:
            try:
                start, end = [x for x in pgs.split('-') if x]
                if len(start) > len(end):
                    end = start[:-len(end)] + end
                pg_cnt += int(end) - int(start) + 1
            except ValueError:
                pass
    return pg_cnt

def is_retracted(df, pmids=None, dois=None):
    filter_terms = ('withdraw', 'retract')
    filter_terms_2 = ('paper', 'study', 'article', 'publication')

    return (df['paperAbstract'].apply(lambda x: any(t1 in x.lower()[:40] and t2 in x.lower()[:40] 
                                                    for t1 in filter_terms
                                                    for t2 in filter_terms_2)) | 
    df['title'].apply(lambda x: any(t1 in x.lower()[:10] for t1 in filter_terms)) |
    df['doi'].apply(lambda x: x in dois) |
    df['pmid'].apply(lambda x: x.split('v')[0] in pmids))

def get_identifiers(source='data/pubmed_redacted.csv'):
    '''returns tuple of sets containing target pmid and dois'''
    pmr = pd.read_csv(source)
    pmids=set(pmr['Db'].apply(lambda x: str(x)))
    dois = set(pmr['Description'].apply(get_doi))
    return pmids, dois

def load_retracted(source='data/retracted_articles'):
    '''Reads in a file of potential retracted articles and returns a 
    filtered dataframe containing relevent fields and labels.'''
    df_retracted = pd.read_json(source, lines=-1)

    filter_terms = ('withdraw', 'retract')
    filter_terms_2 = ('paper', 'study', 'article', 'publication')

    df_retracted = df_retracted[(df_retracted['paperAbstract'].apply(lambda x: any(t1 in x.lower()[:40] and t2 in x.lower()[:40] 
                                                    for t1 in filter_terms
                                                    for t2 in filter_terms_2)) | 
    df_retracted['title'].apply(lambda x: any(t1 in x.lower()[:10] for t1 in filter_terms)) |
    df_retracted['doi'].apply(lambda x: x in dois) |
    df_retracted['pmid'].apply(lambda x: x.split('v')[0] in pmids))]
    return df_retracted

def format_dataframe(df):
    X = df.copy()
    pmids, dois = get_identifiers()
    X['pageLength'] = X['journalPages'].apply(get_length)
    X['numAuthors'] = X['authors'].apply(len)
    X['numEntities'] = X['entities'].apply(len)
    X['numInCitations'] = X['inCitations'].apply(len)
    X['retracted'] = is_retracted(X, pmids=pmids, dois=dois)
    X = X.loc[:, ['id', 'inCitations', 'outCitations', 'numInCitations',
            'authors', 'numAuthors', 'entities', 'numEntities', 
            'venue', 'sources', 'year', 'pageLength', 'retracted']]
    return X

def load_dataframe(source='data/s2-corpus-00', limit=1000):
    text = ''
    limit = limit
    with open(source) as f:
        for i,line in enumerate(f):
            if i == limit: break
            text += line + '\n'
    return pd.read_json(text, lines=-1)