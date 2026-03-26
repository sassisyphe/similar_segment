import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global batched tensor cache
GLOBAL_POOL_TENSORS = {}
GLOBAL_POOL_MASKS = {}
GLOBAL_STOCK_CODES = []
GLOBAL_DATES = {}

def build_batched_tensor(pool_dict, feature_cols):
    """
    Build a massive 3D tensor containing ALL stocks.
    Shape: [Batch (num_stocks), Channels (D), Max_Length]
    """
    codes = list(pool_dict.keys())
    max_len = max(len(pool_dict[c]) for c in codes)
    D = len(feature_cols)
    
    batched_tensor = torch.zeros((len(codes), D, max_len), dtype=torch.float32)
    mask_tensor = torch.zeros((len(codes), max_len), dtype=torch.bool)
    
    for i, code in enumerate(codes):
        df = pool_dict[code]
        L = len(df)
        if L > 0:
            feat_array = df[feature_cols].values # (L, D)
            # transpose to (D, L) and copy to tensor
            batched_tensor[i, :, :L] = torch.tensor(feat_array, dtype=torch.float32).t()
            mask_tensor[i, :L] = True
            
    return batched_tensor.to(DEVICE), mask_tensor.to(DEVICE), codes

# ----------------- Feature Engineering -----------------
def calculate_technical_indicators(df):
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
    
    # Shape features
    df['body'] = df['close'] - df['open']
    df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
    
    # Tech
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_mid'] = df['close'].rolling(20).mean()
    df['BB_std'] = df['close'].rolling(20).std()
    df['BB_width'] = (4 * df['BB_std']) / (df['BB_mid'] + 1e-8)
    
    # MFI (Money Flow Index) 14 days
    raw_money_flow = df['typical_price'] * df['volume']
    diff = df['typical_price'].diff()
    positive_flow = np.where(diff > 0, raw_money_flow, 0)
    negative_flow = np.where(diff < 0, raw_money_flow, 0)
    
    pos_mf = pd.Series(positive_flow).rolling(14).sum()
    neg_mf = pd.Series(negative_flow).rolling(14).sum()
    money_ratio = pos_mf / (neg_mf + 1e-8)
    df['MFI'] = 100 - (100 / (1 + money_ratio))
        
    return df.dropna().reset_index(drop=True)

# ----------------- Strategies -----------------
class SimilarityStrategy(ABC):
    @abstractmethod
    def extract_features(self, df): pass
    @abstractmethod
    def get_feature_columns(self): pass
    @abstractmethod
    def calculate_similarities_batched(self, target_features, batched_pool, mask_pool, length):
        pass

class IndependentFeatureStrategy(SimilarityStrategy):
    def calculate_similarities_batched(self, target_features, batched_pool, mask_pool, length):
        # target_features: (L, D)
        # batched_pool: (B, D, M_max)
        B, D, M_max = batched_pool.shape
        if M_max < length: return torch.zeros((B, M_max), device=DEVICE)
        
        target_t = torch.tensor(target_features, dtype=torch.float32, device=DEVICE).transpose(0, 1).unsqueeze(1) # (D, 1, L)
        
        # 1. Target Stats
        target_mean = target_t.mean(dim=2, keepdim=True)
        target_centered = target_t - target_mean
        target_std = target_centered.std(dim=2, unbiased=False)
        target_std = torch.where(target_std == 0, torch.tensor(1e-8, device=DEVICE), target_std)
        # In PyTorch, F.conv1d performs cross-correlation, not convolution.
        # So we DO NOT flip the kernel to calculate Pearson correlation.
        target_kernel = target_centered
        ones_kernel = torch.ones((D, 1, length), dtype=torch.float32, device=DEVICE) / length
        
        BATCH_SIZE = 1000
        all_mean_corrs = []
        
        for i in range(0, B, BATCH_SIZE):
            sub_pool = batched_pool[i:i+BATCH_SIZE]
            
            # 2. Pool Local Stats via Conv1d
            w_mean = F.conv1d(sub_pool, ones_kernel, groups=D) # (sub_B, D, M_max - L + 1)
            
            w_mean_sq = F.conv1d(sub_pool ** 2, ones_kernel, groups=D)
            w_var = torch.clamp(w_mean_sq - w_mean ** 2, min=0.0)
            w_std = torch.sqrt(w_var)
            w_std = torch.where(w_std == 0, torch.tensor(1e-8, device=DEVICE), w_std)
            
            # 3. Covariance
            # Using F.conv1d directly gives the unnormalized cross-correlation sum(x_i * y_i)
            # Since target is already centered, cov = sum((w - w_mean) * (t - t_mean)) / L
            # which simplifies to sum(w * (t - t_mean)) / L because sum(t - t_mean) = 0
            cov = F.conv1d(sub_pool, target_kernel, groups=D) / length
            
            # 4. Pearson
            corrs = cov / (w_std * target_std.unsqueeze(0)) # (sub_B, D, M_max - L + 1)
            
            # Clip to strictly [-1, 1] to prevent floating point inaccuracies from making it > 1
            corrs = torch.clamp(corrs, min=-1.0, max=1.0)
            
            mean_corrs = corrs.mean(dim=1) # (sub_B, M_max - L + 1)
            all_mean_corrs.append(mean_corrs)
            
        mean_corrs_full = torch.cat(all_mean_corrs, dim=0)
        
        # 5. Apply Mask
        valid_mask = mask_pool[:, length-1:] # A window is valid if its end index is valid
        mean_corrs_full = torch.where(valid_mask, mean_corrs_full, torch.tensor(-np.inf, device=DEVICE))
        
        return mean_corrs_full

class OHLCV(IndependentFeatureStrategy):
    def get_feature_columns(self): return ['open', 'high', 'low', 'close', 'MFI']
    def extract_features(self, df): return df[self.get_feature_columns()].values

class Shape(IndependentFeatureStrategy):
    def get_feature_columns(self): return ['body', 'upper_shadow', 'lower_shadow', 'typical_price', 'MFI']
    def extract_features(self, df): return df[self.get_feature_columns()].values

class OHLCV_Tech(IndependentFeatureStrategy):
    def get_feature_columns(self): return ['open', 'high', 'low', 'close', 'MFI', 'MACD_hist', 'RSI', 'BB_width']
    def extract_features(self, df): return df[self.get_feature_columns()].values

class Shape_Tech(IndependentFeatureStrategy):
    def get_feature_columns(self): return ['body', 'upper_shadow', 'lower_shadow', 'typical_price', 'MFI', 'MACD_hist', 'RSI', 'BB_width']
    def extract_features(self, df): return df[self.get_feature_columns()].values

STRATEGIES = {
    'OHLCV': OHLCV(),
    'Shape': Shape(),
    'OHLCV_Tech': OHLCV_Tech(),
    'Shape_Tech': Shape_Tech()
}

# ----------------- Global Preloading -----------------
GLOBAL_DATA = {}
def preload_data(data_dir):
    global GLOBAL_DATA
    # 如果已经加载过，直接返回（防止重复加载）
    if len(GLOBAL_DATA) > 0:
        return
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    print(f"Preloading and Engineering Features for {len(files)} stocks in worker...")
    for f in tqdm(files, desc="Loading Parquet", disable=True): # 禁用子进程的tqdm打印防止刷屏
        code = f.replace('.parquet', '')
        df = pd.read_parquet(os.path.join(data_dir, f))
        df = df.sort_values('date').reset_index(drop=True)
        df = calculate_technical_indicators(df)
        if not df.empty:
            GLOBAL_DATA[code] = df

def init_worker():
    """每个进程启动时，独立加载一次全局数据"""
    preload_data('data/daily')

# ----------------- Worker -----------------
def process_A_task(args):
    a_row, strategy_name = args
    strategy = STRATEGIES[strategy_name]
    
    a_id = a_row['A_id']
    a_code = a_row['code']
    a_start_date = pd.to_datetime(a_row['start_date'])
    L = a_row['length']
    
    out_dir = os.path.join("results", strategy_name)
    details_file = os.path.join(out_dir, f"A_{a_id}_details.csv")
    stats_file = os.path.join(out_dir, f"A_{a_id}_stats.csv")
    
    if os.path.exists(details_file) and os.path.exists(stats_file):
        stats_df = pd.read_csv(stats_file)
        if not stats_df.empty:
            summary_dict = stats_df.iloc[0].to_dict()
            summary_dict['A_id'] = a_id
            summary_dict['length'] = L
            summary_dict['strategy'] = strategy_name
            return summary_dict
        return None

    if a_code not in GLOBAL_DATA: return None
        
    df_A = GLOBAL_DATA[a_code]
    mask_A = (df_A['date'] >= a_start_date)
    if not mask_A.any(): return None
        
    idx_A_start = df_A[mask_A].index[0]
    K_DAYS = [1, 3, 5, 10]
    max_k = max(K_DAYS)
    
    if idx_A_start + L - 1 + max_k >= len(df_A): return None
        
    segment_A = df_A.iloc[idx_A_start : idx_A_start + L]
    if len(segment_A) < L: return None
    
    # 1. Extract Target A features
    feat_A = strategy.extract_features(segment_A)
    
    # 2. Get/Build Batched Tensor for this strategy
    feature_cols = strategy.get_feature_columns()
    cache_key = tuple(feature_cols)
    if cache_key not in GLOBAL_POOL_TENSORS:
        batched_tensor, mask_tensor, codes = build_batched_tensor(GLOBAL_DATA, feature_cols)
        GLOBAL_POOL_TENSORS[cache_key] = (batched_tensor, mask_tensor, codes)
    else:
        batched_tensor, mask_tensor, codes = GLOBAL_POOL_TENSORS[cache_key]
        
    # 3. Mass GPU Similarity Computation
    # Returns (B, M_max - L + 1) similarity matrix
    corrs = strategy.calculate_similarities_batched(feat_A, batched_tensor, mask_tensor, L)
    
    # 4. Find Top 100
    B, M_out = corrs.shape
    corrs_flat = corrs.flatten()
    
    # Pre-fetch enough top candidates to account for overlaps and self-matches
    fetch_k = min(500, corrs_flat.numel())
    top_values, top_indices = torch.topk(corrs_flat, k=fetch_k)
    
    top_values = top_values.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    
    close_A_0 = df_A['close'].iloc[idx_A_start + L - 1]
    ret_A = {}
    for k in K_DAYS:
        close_A_k = df_A['close'].iloc[idx_A_start + L - 1 + k]
        ret_A[f'ret_A_{k}'] = (close_A_k - close_A_0) / close_A_0
    
    all_results = []
    
    for val, flat_idx in zip(top_values, top_indices):
        if val == -np.inf or np.isnan(val): continue
        b_idx = flat_idx // M_out
        time_idx = flat_idx % M_out
        
        b_code = codes[b_idx]
        if b_code == a_code: continue
            
        df_B = GLOBAL_DATA[b_code]
        
        # Rule: enough days to predict K=10
        if time_idx + L - 1 + max_k >= len(df_B): continue
            
        b_start_date = df_B['date'].iloc[time_idx]
        b_end_date = df_B['date'].iloc[time_idx + L - 1]
        
        # Rule: B must happen strictly before A
        if b_end_date >= np.datetime64(a_start_date): continue
            
        close_B_0 = df_B['close'].iloc[time_idx + L - 1]
        ret_dict = {}
        for k in K_DAYS:
            close_B_k = df_B['close'].iloc[time_idx + L - 1 + k]
            ret_dict[f'ret_{k}'] = (close_B_k - close_B_0) / close_B_0
            
        all_results.append({
            'B_code': b_code,
            'B_start_date': b_start_date.strftime('%Y-%m-%d'),
            'B_end_date': b_end_date.strftime('%Y-%m-%d'),
            'similarity': val,
            **ret_dict
        })
        
        if len(all_results) >= 100:
            break
            
    if not all_results: return None
        
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(details_file, index=False)
    
    stats = {
        'A_id': a_id,
        'length': L,
        'strategy': strategy_name,
        'mean_sim': res_df['similarity'].mean()
    }
    
    for k in K_DAYS:
        target_ret = ret_A[f'ret_A_{k}']
        b_returns = res_df[f'ret_{k}'].tolist()
        
        similarities = res_df['similarity'].values
        tau = 0.01
        sim_shifted = (similarities - np.max(similarities)) / tau
        exp_weights = np.exp(sim_shifted)
        weights = exp_weights / np.sum(exp_weights)
        p2_weighted_mean = np.average(b_returns, weights=weights)
        
        percentile = (np.array(b_returns) < target_ret).mean() * 100
        
        stats[f'p1_{k}'] = target_ret
        stats[f'p2_{k}'] = str(b_returns)
        stats[f'mean_p2_{k}'] = np.mean(b_returns)
        stats[f'weighted_p2_{k}'] = p2_weighted_mean
        stats[f'percentile_{k}'] = percentile
        
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(stats_file, index=False)
    
    return stats

def main():
    os.makedirs('results', exist_ok=True)
    for strategy_name in STRATEGIES.keys():
        os.makedirs(os.path.join('results', strategy_name), exist_ok=True)
        
    a_list_path = 'A_list.csv'
    if not os.path.exists(a_list_path):
        print("A_list.csv not found, please run module 2 first.")
        return
        
    a_df = pd.read_csv(a_list_path)
    
    print(f"Starting sequential processing on {DEVICE}...")
    preload_data('data/daily')
    
    summary_list = []
    
    # Process strategy by strategy to avoid caching multiple massive tensors in VRAM
    for strategy_name in STRATEGIES.keys():
        print(f"\n--- Running Strategy: {strategy_name} ---")
        tasks = []
        for _, row in a_df.iterrows():
            tasks.append((row.to_dict(), strategy_name))
            
        print(f"Processing {len(tasks)} tasks for {strategy_name}...")
        
        for task in tqdm(tasks, desc=f"Searching {strategy_name}"):
            res = process_A_task(task)
            if res: summary_list.append(res)
            
        # Clear the global tensor cache to free VRAM for the next strategy
        GLOBAL_POOL_TENSORS.clear()
        torch.cuda.empty_cache()
            
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        summary_df.to_csv('all_summary.csv', index=False)
        print("\nAll done! Summary saved to all_summary.csv")

if __name__ == '__main__':
    main()
