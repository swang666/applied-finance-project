import time
import os
from sec_edgar_downloader import Downloader
from ucla_topic_analysis import get_filings_folder
from ucla_topic_analysis.data.coroutines import print_progress

def download(tickers):
    path = get_filings_folder()
    dl = Downloader(path)
    n = len(tickers)
    for i in range(n):
        print_progress(i, n)    
        if os.path.exists('../Filings/sec_edgar_filings/' + tickers[i]) == False:
            dl.get_10k_filings(tickers[i])