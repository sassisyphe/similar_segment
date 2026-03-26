import os
import random
import pandas as pd
from tqdm import tqdm

DATA_DIR = 'data/daily'
META_FILE = 'stock_meta.csv'
OUTPUT_FILE = 'A_list.csv'
LENGTHS = [20, 40, 60]
N_SAMPLES = 100
MAX_K = 10

def main():
    random.seed(42)
    
    if not os.path.exists(META_FILE):
        print(f"{META_FILE} 不存在，请先运行模块一")
        return

    meta_df = pd.read_csv(META_FILE)
    codes = meta_df['code'].tolist()
    
    print("Loading stock dates...")
    stock_dates = {}
    for code in tqdm(codes, desc="Reading Parquet dates"):
        file_path = os.path.join(DATA_DIR, f"{code}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path, columns=['date'])
            dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
            # 预留前60天用于计算技术指标(如60日Z-score, MACD等)，确保这些天不被作为片段起始点
            if len(dates) > 60 + MAX_K + max(LENGTHS):
                stock_dates[code] = dates[60:]

    all_segments = []
    
    for L in LENGTHS:
        print(f"Generating segments for length {L}...")
        valid_pool = []
        for code, dates in stock_dates.items():
            max_start_idx = len(dates) - L - MAX_K
            if max_start_idx >= 0:
                for idx in range(max_start_idx + 1):
                    valid_pool.append((code, idx))
        
        if len(valid_pool) < N_SAMPLES:
            print(f"Warning: Not enough valid segments for length {L}. Required {N_SAMPLES}, got {len(valid_pool)}")
            samples = valid_pool
        else:
            samples = random.sample(valid_pool, N_SAMPLES)
            
        for i, (code, start_idx) in enumerate(samples):
            dates = stock_dates[code]
            start_date = dates[start_idx]
            end_date = dates[start_idx + L - 1]
            all_segments.append({
                'A_id': f"L{L}_{i+1:03d}",
                'code': code,
                'length': L,
                'start_idx': start_idx,
                'start_date': start_date,
                'end_date': end_date
            })
            
    res_df = pd.DataFrame(all_segments)
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully generated {len(res_df)} segments, saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
