import time
import requests
import asyncio
import pandas as pd
import numpy as np
from ucla_topic_analysis.analysis.download_10k import download
from ucla_topic_analysis.analysis.risk_score import RiskScorePipeline


sp500_constituents = pd.read_csv('rus2k_tic.csv')
tickers = sp500_constituents[~sp500_constituents['x'].isna()]['x'].unique()

def run_downloader():
    try:
        download(tickers)
    except requests.exceptions.RequestException:
        #wait 2 min
        print('encounter 503, wait 2 min to re run')
        time.sleep(120)
        run_downloader()

run_downloader()

'''
if __name__ == "__main__":
    riskscore = RiskScorePipeline()
    asyncio.run(riskscore.calc_risk())
'''  
