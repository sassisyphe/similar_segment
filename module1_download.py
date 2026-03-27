import os
import pandas as pd
import numpy as np
import baostock as bs
import concurrent.futures
from tqdm import tqdm
from datetime import datetime, timedelta
import time
import re

START_DATE = '2010-01-01'
END_DATE = '2025-12-31'
DATA_DIR = 'data/daily'
META_FILE = 'stock_meta.csv'
MIN_DAYS = 200

# Global lock for thread-safe baostock login/logout (baostock is not strictly thread-safe in some environments)
import threading
bs_lock = threading.Lock()

def process_stock_optimized(stock_info):
    code = stock_info['code']
    name = stock_info['code_name']
    
    file_path = os.path.join(DATA_DIR, f"{code}.parquet")
    
    if os.path.exists(file_path):
        try:
            df = pd.read_parquet(file_path, columns=['date'])
            if len(df) >= MIN_DAYS:
                return {
                    'code': code, 'name': name,
                    'start_date': df['date'].min().strftime('%Y-%m-%d'),
                    'end_date': df['date'].max().strftime('%Y-%m-%d'),
                    'valid_days': len(df)
                }
            return None
        except Exception:
            pass

    max_retries = 3
    data_list = []
    
    # 每个独立的子进程进来时，单独为自己建立一条独立的 TCP 连接
    import io, sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    bs.login()
    sys.stdout = old_stdout
    
    for attempt in range(max_retries):
        try:
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
                start_date=START_DATE, end_date=END_DATE,
                frequency="d", adjustflag="2"
            )
            
            error_code = rs.error_code
            if error_code == '0':
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                break
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    # 重连
                    sys.stdout = io.StringIO()
                    bs.login()
                    sys.stdout = old_stdout
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                sys.stdout = io.StringIO()
                bs.login()
                sys.stdout = old_stdout
                continue
            break
            
    # 离开前 logout，释放 socket
    sys.stdout = io.StringIO()
    bs.logout()
    sys.stdout = old_stdout
    
    if not data_list: return None
        
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    df['tradestatus'] = df['tradestatus'].astype(int)
    df['isST'] = df['isST'].astype(int)
    
    df = df[(df['tradestatus'] == 1) & (df['isST'] == 0)].copy()
    if len(df) < MIN_DAYS: return None
        
    df['date'] = pd.to_datetime(df['date'])
    numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.sort_values('date').reset_index(drop=True)
    
    # 将原始数据直接保存到 Parquet 文件
    df.to_parquet(file_path, engine='pyarrow')
    
    return {
        'code': code, 'name': name,
        'start_date': df['date'].min().strftime('%Y-%m-%d'),
        'end_date': df['date'].max().strftime('%Y-%m-%d'),
        'valid_days': len(df)
    }

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    bs.login()
    
    # 只保留是股票的资产
    print("Fetching stock basic info to filter individual A-shares...")
    rs_basic = bs.query_stock_basic()
    stock_list = []
    while (rs_basic.error_code == '0') & rs_basic.next():
        row = rs_basic.get_row_data()
        # row: [code, code_name, ipoDate, outDate, type, status]
        if row[4] == '1':
            stock_list.append({'code': row[0], 'code_name': row[1]})
            
    print(f"Total valid individual A-shares found (type=1): {len(stock_list)}")
    
    print("Fetching stock industry classification...")
    rs_industry = bs.query_stock_industry()
    industry_dict = {}
    while (rs_industry.error_code == '0') & rs_industry.next():
        row = rs_industry.get_row_data()
        # 仅当行业信息不为空且有实际内容时才加入字典
        if row[3] and str(row[3]).strip():
            industry_dict[row[1]] = row[3]
    bs.logout()
    
    print("Filtering out stocks without valid industry info...")
    filtered_stock_list = []
    for stock in stock_list:
        if stock['code'] in industry_dict:
            filtered_stock_list.append(stock)
            
    print(f"Stocks retained after industry filter: {len(filtered_stock_list)} / {len(stock_list)}")
    stock_list = filtered_stock_list
    
    meta_data = []
    
    max_workers = 12
    print(f"Starting download with max_workers={max_workers} (ProcessPool) to massively accelerate process...")
    
    # ProcessPoolExecutor 必须在 __main__ 块内，且每个 worker 内部自己管理 session
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_stock_optimized, stock_list), total=len(stock_list), desc="Downloading K-data"))
        
    for res in results:
        if res: 
            code = res['code']
            raw_industry = industry_dict.get(code, 'Unknown')
            res['industry'] = raw_industry
            
            match = re.match(r'^([A-Z0-9]+)', raw_industry)
            if match:
                res['industry_simple'] = match.group(1)
            else:
                res['industry_simple'] = raw_industry
                
            meta_data.append(res)
            
    meta_df = pd.DataFrame(meta_data)
    meta_df.to_csv(META_FILE, index=False)
    print(f"Data download complete. Valid stocks saved: {len(meta_df)}")

if __name__ == '__main__':
    main()
