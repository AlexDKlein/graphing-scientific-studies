import boto3
import time
import pandas as pd
from src.data_reader import DataReader, RetractionFinder
from src.get_redacted import get_paper, load_redacted, get_doi, gen_retracted



if __name__ == '__main__':
    pmr = load_redacted()
    pmids=set(pmr['Db'].apply(lambda x: str(x)))
    dois = set(pmr['Description'].apply(get_doi))

    s3 = boto3.client('s3')
    s3_finder = RetractionFinder(pmids=pmids, dois=dois)
    tot_lines = 0
    for i in range(40):
        t = time.process_time()
        corpus = s3.get_object(
            Bucket='alexklein', 
            Key=f'capstone/data/s2-corpus-{i if i >= 10 else "0" + str(i)}')
        s3_finder.search_stream(corpus['Body'])
        with open('data/retracted_articles', 'w') as f:
            for article in s3_finder.found:
                f.writelines(json.dumps(article))
                f.writelines('\n')
        
        print(f's2-corpus-{i if i >= 10 else "0" + str(i)} completed in {(time.process_time()-t):0.0f} s.')
        print(f'{len(s3_finder.found) - tot_lines} new lines found. {len(s3_finder.found)} lines total.\n')
        tot_lines = len(s3_finder.found)

    