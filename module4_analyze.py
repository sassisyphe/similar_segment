import os
import pandas as pd
import warnings

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import mplfinance as mpf
import itertools
import networkx as nx

# Configure Matplotlib for Chinese fonts
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Force Seaborn to use the same font
sns.set_theme(style="whitegrid", font='WenQuanYi Micro Hei')
sns.set(font='WenQuanYi Micro Hei')

LENGTHS = [20, 40, 60]
K_DAYS = [1, 3, 5, 10]

# Update strategy list
STRATEGIES = [
    'OHLCV_Ind', 'OHLCV_Joint', 
    'Shape_Ind', 'Shape_Joint',
    'OHLCV_Tech_Ind', 'OHLCV_Tech_Joint',
    'Shape_Tech_Ind', 'Shape_Tech_Joint'
]

def load_stock_data(code):
    file_path = f"data/daily/{code}.parquet"
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    return None

def calc_softmax_weights(similarities, tau=0.01):
    sim_shifted = (similarities - np.max(similarities)) / tau
    exp_weights = np.exp(sim_shifted)
    return exp_weights / np.sum(exp_weights)

def plot_micro_analysis(strategy, df_summary):
    """
    微观层面深入分析：
    计算每个片段A预测的准确性，并与该片段匹配出的B集合的统计分布特征（方差、偏度等）进行回归/分布对比。
    以此找出“什么样的B集合分布，能带来更准的预测”。
    """
    print(f"Generating Micro Analysis for {strategy}...")
    out_dir = f'plots/Micro_Analysis/{strategy}'
    os.makedirs(out_dir, exist_ok=True)
    
    strat_df = df_summary[df_summary['strategy'] == strategy]
    if strat_df.empty: return
    
    micro_data = []
    
    for _, row in strat_df.iterrows():
        a_id = row['A_id']
        length = row['length']
        details_path = f"results/{strategy}/A_{a_id}_details.csv"
        
        if not os.path.exists(details_path): continue
        df_details = pd.read_csv(details_path)
        if df_details.empty: continue
            
        weights = calc_softmax_weights(df_details['similarity'].values)
        
        for k in K_DAYS:
            p1 = row[f'p1_{k}']
            p2_list = df_details[f'ret_{k}'].values
            
            # 计算预测值（加权平均）
            pred_p2 = np.average(p2_list, weights=weights)
            
            # 预测是否正确（方向）
            is_correct = 1 if np.sign(p1) == np.sign(pred_p2) else 0
            
            # B集合的统计特征
            b_std = np.std(p2_list) # 分布集中度（越小越集中）
            
            # 将偏度替换为更直观的指标：极端值占比 (胜率极化程度)
            # 例如：计算绝对收益率超过 5% 的极端 B 片段占比
            extreme_threshold = 0.05
            b_extreme_ratio = np.mean(np.abs(p2_list) > extreme_threshold)
            
            b_kurtosis = stats.kurtosis(p2_list) # 峰度（极端值厚尾）
            
            # 多空方向倾向
            pred_direction = "Bullish" if pred_p2 > 0 else "Bearish"
            
            micro_data.append({
                'A_id': a_id,
                'Length': length,
                'K_Days': k,
                'Is_Correct': is_correct,
                'Pred_Direction': pred_direction,
                'B_Std': b_std,
                'B_Extreme_Ratio': b_extreme_ratio,
                'B_Kurtosis': b_kurtosis,
                'Pred_Return': pred_p2,
                'Actual_Return': p1
            })
            
    df_micro = pd.DataFrame(micro_data)
    if df_micro.empty: return
    
    df_micro.to_csv(f"{out_dir}/micro_features.csv", index=False)
    
    # 循环所有 L 和 K 的组合
    for l_val in LENGTHS:
        for k in K_DAYS:
            sub_df = df_micro[(df_micro['K_Days'] == k) & (df_micro['Length'] == l_val)]
            if sub_df.empty: continue
            
            # 1. 集中度(Std)与预测正确率的关系（箱线图）
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Is_Correct', y='B_Std', data=sub_df, palette='Set2')
            plt.title(f'B集合收益率方差 vs 预测准确性 (L={l_val}, K={k})\n(1=预测正确, 0=预测错误) - {strategy}')
            plt.xlabel('方向预测是否正确')
            plt.ylabel('B集合收益率标准差 (分布集中度)')
            plt.savefig(f"{out_dir}/B_Std_vs_Accuracy_L{l_val}_K{k}.png")
            plt.close()
            
            # 2. 极值占比与预测正确率的关系
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Is_Correct', y='B_Extreme_Ratio', data=sub_df, palette='Set3')
            plt.title(f'B集合极端收益占比(>5%) vs 预测准确性 (L={l_val}, K={k})\n- {strategy}')
            plt.xlabel('方向预测是否正确')
            plt.ylabel('B集合极端收益占比')
            plt.savefig(f"{out_dir}/B_ExtremeRatio_vs_Accuracy_L{l_val}_K{k}.png")
            plt.close()
            
            # 3. 多空非对称性：看涨和看跌时的胜率对比
            # 固定横坐标顺序为 Bearish 在左，Bullish 在右
            plt.figure(figsize=(8, 6))
            sns.barplot(x='Pred_Direction', y='Is_Correct', data=sub_df, ci=None, 
                        order=['Bearish', 'Bullish'], palette={'Bearish': 'green', 'Bullish': 'red'})
            plt.axhline(0.5, color='gray', linestyle='--', label='50% 随机基准')
            plt.title(f'多空预测非对称性 (L={l_val}, K={k})\n- {strategy}')
            plt.xlabel('系统预测方向')
            plt.ylabel('方向预测准确率 (Hit Rate)')
            plt.legend()
            plt.savefig(f"{out_dir}/Long_Short_Asymmetry_L{l_val}_K{k}.png")
            plt.close()

def plot_candlesticks(strategy, a_id, length, df_details, a_row):
    out_dir = f"plots/{strategy}/{a_id}"
    os.makedirs(out_dir, exist_ok=True)
    
    a_code = a_row['code']
    a_start = pd.to_datetime(a_row['start_date'])
    
    df_A = load_stock_data(a_code)
    if df_A is None: return
    
    mask_A = (df_A.index >= a_start)
    if not mask_A.any(): return
    idx_start_A = np.where(mask_A)[0][0]
    segment_A = df_A.iloc[idx_start_A : idx_start_A + length].copy()
    segment_A.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    
    top3_B = df_details.head(3)
    
    fig = plt.figure(figsize=(24, 14))
    fig.suptitle(f"Candlestick Comparison (Strategy: {strategy} | Target A: {a_id})", fontsize=20, y=0.95)
    
    mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
    s  = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)
    
    ax_a_main = fig.add_axes([0.05, 0.60, 0.40, 0.28])
    ax_a_vol  = fig.add_axes([0.05, 0.50, 0.40, 0.10], sharex=ax_a_main)
    ax_b1_main = fig.add_axes([0.55, 0.60, 0.40, 0.28])
    ax_b1_vol  = fig.add_axes([0.55, 0.50, 0.40, 0.10], sharex=ax_b1_main)
    ax_b2_main = fig.add_axes([0.05, 0.15, 0.40, 0.28])
    ax_b2_vol  = fig.add_axes([0.05, 0.05, 0.40, 0.10], sharex=ax_b2_main)
    ax_b3_main = fig.add_axes([0.55, 0.15, 0.40, 0.28])
    ax_b3_vol  = fig.add_axes([0.55, 0.05, 0.40, 0.10], sharex=ax_b3_main)
    
    ax_a_main.set_title(f"Target A: {a_code} ({a_start.strftime('%Y-%m-%d')} to {segment_A.index[-1].strftime('%Y-%m-%d')})", fontsize=14)
    mpf.plot(segment_A, type='candle', ax=ax_a_main, volume=ax_a_vol, style=s, show_nontrading=False, datetime_format='%Y-%m-%d')
    
    axes_main = [ax_b1_main, ax_b2_main, ax_b3_main]
    axes_vol = [ax_b1_vol, ax_b2_vol, ax_b3_vol]
    
    for i, (_, b_row) in enumerate(top3_B.iterrows()):
        if i >= 3: break
        b_code = b_row['B_code']
        b_start = pd.to_datetime(b_row['B_start_date'])
        sim = b_row['similarity']
        
        df_B = load_stock_data(b_code)
        if df_B is None: continue
            
        mask_B = (df_B.index >= b_start)
        if not mask_B.any(): continue
        idx_start_B = np.where(mask_B)[0][0]
        segment_B = df_B.iloc[idx_start_B : idx_start_B + length].copy()
        segment_B.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        
        ax_m = axes_main[i]
        ax_v = axes_vol[i]
        ax_m.set_title(f"Top {i+1} Match: {b_code} ({b_start.strftime('%Y-%m-%d')} to {segment_B.index[-1].strftime('%Y-%m-%d')})\nSimilarity: {sim:.4f}", fontsize=14)
        mpf.plot(segment_B, type='candle', ax=ax_m, volume=ax_v, style=s, show_nontrading=False, datetime_format='%Y-%m-%d')

    plt.savefig(f"{out_dir}/Candlesticks_A_vs_Top3B.png", bbox_inches='tight')
    plt.close(fig)

def plot_case_study(strategy, a_id, length, df_details, target_returns, a_industry, industry_map):
    out_dir = f"plots/{strategy}/{a_id}"
    os.makedirs(out_dir, exist_ok=True)
    df_details.to_csv(f"{out_dir}/B_details.csv", index=False)
    
    # 1. 行业相似度分布柱状图
    # 映射B片段的行业
    df_details['B_industry'] = df_details['B_code'].map(lambda c: industry_map.get(c, 'Unknown'))
    df_details['B_industry'] = df_details['B_industry'].replace('', 'Unknown').fillna('Unknown')
    
    # 聚合每个行业的相似度总和
    industry_sim_sum = df_details.groupby('B_industry')['similarity'].sum().reset_index()
    industry_sim_sum = industry_sim_sum.sort_values('similarity', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=industry_sim_sum, x='B_industry', y='similarity', palette='viridis')
    plt.title(f'片段A ({a_id}) 相似行业分布\n(A片段行业: {a_industry} | 策略: {strategy} | 长度: {length})', fontsize=14)
    plt.xlabel('匹配片段B的行业', fontsize=12)
    plt.ylabel('相似度总和', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Industry_Similarity_Sum.png")
    plt.close()
    
    # 2. 收益率分布图
    for k in K_DAYS:
        p1 = target_returns[f'p1_{k}']
        p2_list = df_details[f'ret_{k}'].values
        similarities = df_details['similarity'].values
        
        tau = 0.01
        sim_shifted = (similarities - np.max(similarities)) / tau
        exp_weights = np.exp(sim_shifted)
        weights = exp_weights / np.sum(exp_weights)
        p2_weighted_mean = np.average(p2_list, weights=weights)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(p2_list, bins=20, kde=True, color='skyblue', stat='density', alpha=0.6, label='Similar Segments Return (p2_i)')
        plt.axvline(x=p1, color='red', linestyle='--', linewidth=2, label=f'Target Return (p1) = {p1:.4f}')
        p2_mean = np.mean(p2_list)
        plt.axvline(x=p2_mean, color='green', linestyle='-.', linewidth=2, label=f'Simple Mean Return = {p2_mean:.4f}')
        plt.axvline(x=p2_weighted_mean, color='purple', linestyle=':', linewidth=2, label=f'Softmax Weighted Mean = {p2_weighted_mean:.4f}')
        
        plt.title(f'Return Distribution (Horizon K={k} Days)\nStrategy: {strategy} | Target: {a_id} | Length: {length}')
        plt.xlabel('Return')
        plt.ylabel('Density')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/Return_Distribution_K{k}.png", bbox_inches='tight')
        plt.close()

def perform_overlap_analysis(df_summary, a_info_dict):
    print("Performing Overlap, Concentration, and Industry Network Analysis...")
    os.makedirs('plots/Macro_Analysis', exist_ok=True)
    
    # Load meta info to get industries
    meta_df = pd.DataFrame()
    if os.path.exists('stock_meta.csv'):
        meta_df = pd.read_csv('stock_meta.csv')
    industry_map = dict(zip(meta_df['code'], meta_df.get('industry_simple', meta_df.get('industry', 'Unknown')))) if not meta_df.empty else {}
    
    concentration_data = []
    
    # We will analyze overlap specifically for K=1 (B sets are the same regardless of K, so just read one)
    for strategy in STRATEGIES:
        strat_df = df_summary[df_summary['strategy'] == strategy]
        if strat_df.empty: continue
        
        b_sets = {}
        # Prepare graph for this strategy
        G = nx.DiGraph()
        
        for _, row in strat_df.iterrows():
            a_id = row['A_id']
            details_path = f"results/{strategy}/A_{a_id}_details.csv"
            if os.path.exists(details_path):
                df_details = pd.read_csv(details_path)
                unique_stocks = df_details['B_code'].nunique()
                concentration_data.append({'Strategy': strategy, 'A_id': a_id, 'Unique_B_Stocks': unique_stocks})
                b_sets[a_id] = set(df_details['B_code'].astype(str) + "_" + df_details['B_start_date'].astype(str))
                
                # Network analysis
                if a_id in a_info_dict:
                    a_code = a_info_dict[a_id]['code']
                    a_industry = industry_map.get(a_code, 'Unknown')
                    if pd.isna(a_industry) or a_industry == '':
                        a_industry = 'Unknown'
                        
                    for b_code in df_details['B_code']:
                        b_industry = industry_map.get(b_code, 'Unknown')
                        if pd.isna(b_industry) or b_industry == '':
                            b_industry = 'Unknown'
                            
                        # Add edge from A_industry to B_industry
                        if G.has_edge(a_industry, b_industry):
                            G[a_industry][b_industry]['weight'] += 1
                        else:
                            G.add_edge(a_industry, b_industry, weight=1)
                            
        # Draw Network Graph for this strategy
        if G.number_of_nodes() > 0:
            plt.figure(figsize=(14, 14))
            pos = nx.spring_layout(G, k=1.5, seed=42)
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            normalized_weights = [w / max_weight * 5 for w in weights] # Scale for visual
            
            # Draw nodes
            node_sizes = [sum([G[u][v]['weight'] for v in G.successors(u)]) * 10 + 100 for u in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=normalized_weights, edge_color='gray', alpha=0.5, arrowsize=15)
            
            # Draw labels (use a fallback font if needed, here we just use default matplotlib sans-serif)
            # To avoid Chinese font issues, we'll just plot it. If it's missing font, squares will appear.
            # We recommend setting a font family in your environment if you want perfect Chinese labels.
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='WenQuanYi Micro Hei')
            
            plt.title(f'Industry Flow Network (A -> B)\nStrategy: {strategy}', fontsize=16)
            plt.axis('off')
            plt.savefig(f'plots/Macro_Analysis/industry_network_{strategy}.png', bbox_inches='tight')
            plt.close()
                
        # Inter-A overlap calculation (sample 100 random pairs to avoid huge matrix if too many A's)
        a_ids = list(b_sets.keys())
        if len(a_ids) > 1:
            jaccard_sims = []
            import random
            pairs = list(itertools.combinations(a_ids, 2))
            if len(pairs) > 1000:
                pairs = random.sample(pairs, 1000)
            for a1, a2 in pairs:
                set1, set2 = b_sets[a1], b_sets[a2]
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                if union > 0:
                    jaccard_sims.append(intersection / union)
            
            plt.figure(figsize=(8, 5))
            sns.histplot(jaccard_sims, bins=20, kde=True)
            plt.title(f'Inter-A Jaccard Similarity Distribution (Overlap)\nStrategy: {strategy}')
            plt.xlabel('Jaccard Similarity (0=No Overlap, 1=Identical B Sets)')
            plt.ylabel('Frequency')
            plt.savefig(f'plots/Macro_Analysis/overlap_inter_A_{strategy}.png')
            plt.close()
            
    conc_df = pd.DataFrame(concentration_data)
    if not conc_df.empty:
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=conc_df, x='Strategy', y='Unique_B_Stocks', hue='Strategy', palette='Set2', legend=False)
        plt.axhline(100, color='r', linestyle='--', label='Max Possible (100)')
        plt.title('Intra-A Concentration: Number of Unique Stocks among 100 B Segments')
        plt.ylabel('Unique Stock Count')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/Macro_Analysis/overlap_intra_A_concentration.png')
        plt.close()

def generate_markdown_report(metrics_df, df_summary):
    report_path = 'plots/Macro_Analysis/Macro_Analysis_Report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 宏观分析报告：基于相似性的收益预测\n\n")
        
        f.write("## 1. 执行摘要\n")
        f.write("本报告评估了历史上相似形态对未来股票收益的预测能力。我们在**4种不同的特征工程策略**上进行了对比：\n")
        f.write("- **OHLCV**: 基线独立特征（开盘价、最高价、最低价、收盘价、资金流量指标MFI）。\n")
        f.write("- **Shape**: K线形态特征（实体、影线、典型价格、资金流量指标MFI）。\n")
        f.write("- **OHLCV_Tech**: OHLCV + 技术指标（MACD, RSI, 布林带宽度）。\n")
        f.write("- **Shape_Tech**: 形态特征 + 技术指标。\n")
        f.write("所有策略均采用各维度特征独立计算Pearson相关系数后再取平均的方式，避免了将多维特征展平为单一长序列导致的维度灾难。\n\n")
        
        f.write("## 2. 策略详细表现 (按片段长度 L 和 预测周期 K 分类)\n")
        f.write("*注：方向胜率（Directional Accuracy）> 50% 代表具备一定的预测优势。P_Val 为二项检验（单侧）显著性，若小于 0.05 则认为显著优于随机猜测。MAE 为预测收益与真实收益的平均绝对误差。*\n\n")
        
        for L in sorted(metrics_df['Length'].unique()):
            f.write(f"### 片段长度 L = {L} 天\n\n")
            for k in sorted(metrics_df['K_Days'].unique()):
                f.write(f"#### 预测周期 K = {k} 天\n")
                sub = metrics_df[(metrics_df['Length'] == L) & (metrics_df['K_Days'] == k)]
                if sub.empty:
                    continue
                sub_disp = sub[['Strategy', 'Dir_Acc_Simple', 'Dir_Acc_Weighted', 'P_Val_Simple', 'P_Val_Weighted', 'MAE_Simple', 'MAE_Weighted']].copy()
                f.write(sub_disp.to_markdown(index=False))
                f.write("\n\n")
            
        f.write("## 3. 可视化图表索引\n")
        f.write("请参考 `plots/Macro_Analysis/` 目录中生成的详细图表进行深入分析：\n")
        f.write("- `heatmap_acc_weighted_*.png` & `heatmap_mae_weighted_*.png`: 各策略下片段长度 L 和预测周期 K 的二维热力图（用于寻找最优参数组合）。\n")
        f.write("- `industry_network_*.png`: 目标片段A所属行业与匹配出的相似片段B所属行业之间的网络关系图（发现跨行业形态轮动）。\n")
        f.write("- `overlap_intra_A_concentration.png`: 分析这100个相似片段是否来自多样化的股票（避免过于集中在某几只股票上）。\n")
        f.write("- `overlap_inter_A_*.png`: Jaccard 相似度分布，用于验证不同的A片段不会碰巧找到大量相同的B片段。\n")

def plot_macro_statistics(df_summary):
    print("Generating Macro Statistical Charts...")
    os.makedirs('plots/Macro_Analysis', exist_ok=True)
    
    metrics = []
    for strategy in df_summary['strategy'].unique():
        strat_df = df_summary[df_summary['strategy'] == strategy]
        for L in LENGTHS:
            sub_df = strat_df[strat_df['length'] == L]
            if sub_df.empty: continue
            
            for k in K_DAYS:
                p1 = sub_df[f'p1_{k}']
                mean_p2 = sub_df[f'mean_p2_{k}']
                weighted_p2 = sub_df[f'weighted_p2_{k}']
                
                # Directional Accuracy
                hit_rate_simple = np.mean(np.sign(p1) == np.sign(mean_p2)) * 100
                hit_rate_weighted = np.mean(np.sign(p1) == np.sign(weighted_p2)) * 100
                
                # Hypothesis Testing (Binomial test for hit rate > 0.5)
                # Count successes
                successes_simple = np.sum(np.sign(p1) == np.sign(mean_p2))
                successes_weighted = np.sum(np.sign(p1) == np.sign(weighted_p2))
                n_trials = len(p1)
                
                pval_simple = stats.binomtest(successes_simple, n_trials, p=0.5, alternative='greater').pvalue
                pval_weighted = stats.binomtest(successes_weighted, n_trials, p=0.5, alternative='greater').pvalue
                
                # Error Analysis (Difference between actual and predicted)
                mae_simple = np.mean(np.abs(p1 - mean_p2))
                mae_weighted = np.mean(np.abs(p1 - weighted_p2))
                
                metrics.append({
                    'Strategy': strategy,
                    'Length': L,
                    'K_Days': k,
                    'Dir_Acc_Simple': hit_rate_simple,
                    'Dir_Acc_Weighted': hit_rate_weighted,
                    'P_Val_Simple': pval_simple,
                    'P_Val_Weighted': pval_weighted,
                    'MAE_Simple': mae_simple,
                    'MAE_Weighted': mae_weighted
                })
                
    metrics_df = pd.DataFrame(metrics)
    if metrics_df.empty: return
    metrics_df.to_csv('plots/Macro_Analysis/macro_metrics.csv', index=False)
    
    generate_markdown_report(metrics_df, df_summary)

    # --- New: 八大策略大比武 (Strategy Leaderboard) ---
    print("Generating Strategy Leaderboard...")
    # 我们选一个最具代表性的长线组合 (比如 L=60, K=10) 来进行策略对比
    for l_val in [20, 40, 60]:
        for k_val in [1, 2, 5, 10]:
            sub_metrics = metrics_df[(metrics_df['Length'] == l_val) & (metrics_df['K_Days'] == k_val)]
            if sub_metrics.empty: continue
            
            sub_metrics = sub_metrics.sort_values('Dir_Acc_Weighted', ascending=False)
            
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Strategy', y='Dir_Acc_Weighted', data=sub_metrics, palette='mako')
            plt.axhline(50, color='red', linestyle='--', label='50% 随机基准')
            
            # 标注显著性星号
            for i, row in enumerate(sub_metrics.itertuples()):
                pval = row.P_Val_Weighted
                stars = ""
                if pval < 0.01: stars = "***"
                elif pval < 0.05: stars = "**"
                elif pval < 0.1: stars = "*"
                
                ax.text(i, row.Dir_Acc_Weighted + 0.5, f"{row.Dir_Acc_Weighted:.1f}%\n{stars}", 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
                
            plt.title(f'策略横向大比武：加权方向胜率对比 (L={l_val}, K={k_val})')
            plt.xlabel('特征工程策略')
            plt.ylabel('预测准确率 (%)')
            plt.xticks(rotation=45)
            plt.ylim(0, 100)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'plots/Macro_Analysis/Strategy_Leaderboard_L{l_val}_K{k_val}.png')
            plt.close()

def process_plot_task(args):
    strategy, a_id, length, details_path, target_returns, a_industry, industry_map, a_row = args
    try:
        df_details = pd.read_csv(details_path)
        plot_case_study(strategy, a_id, length, df_details, target_returns, a_industry, industry_map)
        if a_row is not None:
            plot_candlesticks(strategy, a_id, length, df_details, a_row)
    except Exception as e:
        print(f"Error plotting {strategy} {a_id}: {e}")

def main():
    os.makedirs('plots', exist_ok=True)
    if not os.path.exists('all_summary.csv'):
        print("all_summary.csv not found.")
        return
        
    df_summary = pd.read_csv('all_summary.csv')
    
    # 1. Macro Statistics & Markdown Report
    plot_macro_statistics(df_summary)
    
    # 1.5 Micro Statistics (B-distribution vs Prediction Accuracy)
    for strategy in STRATEGIES:
        plot_micro_analysis(strategy, df_summary)
        
    a_df = pd.read_csv('A_list.csv')
    a_info_dict = {row['A_id']: row for _, row in a_df.iterrows()}
    
    # 2. Overlap & Concentration & Industry Analysis (Skipped as requested)
    # perform_overlap_analysis(df_summary, a_info_dict)
    
    # 3. Case Study Plots (Ensuring every strategy & segment is covered)
    print("Generating case study plots for each Target Segment A...")
    
    # Reload meta for case study
    meta_df = pd.DataFrame()
    if os.path.exists('stock_meta.csv'):
        meta_df = pd.read_csv('stock_meta.csv')
    industry_map = dict(zip(meta_df['code'], meta_df.get('industry_simple', meta_df.get('industry', 'Unknown')))) if not meta_df.empty else {}
    
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    
    plot_tasks = []
    for _, row in df_summary.iterrows():
        a_id = row['A_id']
        strategy = row['strategy']
        length = row['length']
        
        details_path = f"results/{strategy}/A_{a_id}_details.csv"
        if not os.path.exists(details_path): continue
            
        target_returns = {f'p1_{k}': row[f'p1_{k}'] for k in K_DAYS}
        a_code = a_info_dict[a_id]['code'] if a_id in a_info_dict else 'Unknown'
        a_industry = industry_map.get(a_code, 'Unknown')
        
        a_row = a_info_dict.get(a_id, None)
        
        plot_tasks.append((strategy, a_id, length, details_path, target_returns, a_industry, industry_map, a_row))
        
    print(f"Total plotting tasks: {len(plot_tasks)}. Accelerating with ProcessPoolExecutor...")
    
    # Since matplotlib is not thread-safe, we use processes
    # Limit workers to prevent memory explosion from too many matplotlib figures
    max_workers = min(os.cpu_count() or 4, 8) 
    
    # Process sequentially first to ensure Matplotlib font cache is built properly by one process
    if len(plot_tasks) > 0:
        process_plot_task(plot_tasks[0])
        
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_plot_task, plot_tasks[1:]), total=len(plot_tasks)-1, desc="Plotting"))
        
    print("All analysis complete.")

if __name__ == '__main__':
    main()
