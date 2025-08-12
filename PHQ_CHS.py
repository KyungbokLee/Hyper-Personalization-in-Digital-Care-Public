#!/usr/bin/env python3
"""
Linear Thompson Sampling을 활용한 Contextual Bandit 문제를 시뮬레이션하는 Python 스크립트입니다.
PHQ2 Score(0-6)를 최적화하며, TS와 Cyclic/Single 방식의 성능을 비교합니다.
결과는 Plotly를 통해 시각화되며, 'results/PHQ/CHS' 디렉토리에 저장됩니다.
"""

# 필요한 라이브러리 임포트
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import product
from scipy.optimize import root
import plotly.graph_objects as go

# 결과 저장 디렉토리 생성
os.makedirs('results/PHQ', exist_ok=True)

# 시뮬레이션 파라미터 설정
T = 1000  # 시뮬레이션 timestep 수 (피험자 수)
M = 100    # 실험 반복 횟수
SEED = 2  # 랜덤 시드
MIN_SCORE = -6  # PHQ 최소 Score (글로벌 변수)
MAX_SCORE = 0  # PHQ 최대 Score (글로벌 변수)

# Step 1: 데이터 로드 및 전처리
def load_and_preprocess_data():
    """Generate a synthetic dataset based on given population proportions for CHS."""
    np.random.seed(SEED)  # 재현성을 위한 시드 설정
    n_samples = 10000     # 데이터 크기

    # 제공된 비율 기반으로 변수 생성
    age = np.random.choice([0, 1], size=n_samples, p=[0.53, 0.47])  # 0: Young (53%), 1: Old (47%)
    sex = np.random.choice([0, 1], size=n_samples, p=[0.4165, 0.5835])  # 0: Male (41.65%), 1: Female (58.35%)
    region = np.random.choice([0, 1], size=n_samples, p=[0.5901, 0.4099])  # 0: Rural (59.01%), 1: Urban (40.99%)
    lonely = np.random.choice([0, 1], size=n_samples, p=[0.936, 0.064])  # 0: Not lonely (93.6%), 1: Lonely (6.4%)

    # 나머지 변수는 50:50 비율로 생성
    family = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])  # 0: Not alone, 1: Alone
    econ = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])    # 0: Not poor, 1: Poor
    chronic = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5]) # 0: <3, 1: >=3
    cognition = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])  # 0: Normal, 1: Impaired
    difficulty = np.random.uniform(0, 1, size=n_samples)  # 0~1 사이 연속형 변수

    # DataFrame 생성
    df = pd.DataFrame({
        'sex': sex,
        'family': family,
        'econ': econ,
        'region': region,
        'age': age,
        'chronic': chronic,
        'difficulty': difficulty,
        'cognition': cognition,
        'lonely': lonely
    })

    return df

# Step 2: 사용자 및 아이템 변수 정의
def define_variables():
    """사용자와 아이템 변수를 정의하고 조합을 생성합니다."""
    user_vars = ['sex', 'family', 'econ', 'region', 'age', 'chronic', 'difficulty', 'cognition', 'lonely']
    user_values = list(product([0, 1], repeat=len(user_vars)))
    X_user = pd.DataFrame(user_values, columns=user_vars)

    content = ['intervention', 'health_info', 'check-in']
    call_frequency = [0, 0.5, 1]
    call_duration = [0, 0.5, 1]
    item_values = list(product(content, call_frequency, call_duration))
    X_item = pd.DataFrame(item_values, columns=['Content', 'Call_frequency', 'Call_duration'])
    X_item = pd.get_dummies(X_item, columns=['Content'])

    return X_user, X_item, user_vars, X_item.columns.tolist()

# Step 3: 상호작용 정의 및 베타 계수 초기화
def initialize_beta(df, X_user_columns, X_item_columns):
    """상호작용을 정의하고 Beta 계수를 초기화합니다."""
    interaction_columns = [
        'age*Content_check-in', 'region*Call_frequency', 'sex*Call_duration',
        'lonely*Content_intervention', 'difficulty*Call_duration',
        'cognition*Content_check-in', 'chronic*Content_health_info'
    ]
    beta_index = ['intercept'] + X_user_columns + X_item_columns + interaction_columns

    np.random.seed(SEED)
    beta = pd.Series({var: np.random.uniform(-2, 2) for var in beta_index})
    beta['intercept'] = 10
    
    beta['Call_frequency'] = 5.0 * 2
    beta['Call_duration'] = -10.0 * 2
    beta['Content_check-in'] = -10.0 * 2
    beta['Content_intervention'] = 5.0 * 2
    beta['Content_health_info'] = -5.0 * 2

    beta['region*Call_frequency'] = -10.0 * 2
    beta['sex*Call_duration'] = 10.0 * 2
    beta['difficulty*Call_duration'] = 10.0 * 2
    beta['age*Content_check-in'] = 10.0 * 2
    beta['cognition*Content_check-in'] = 10.0 * 2
    beta['lonely*Content_intervention'] = -10.0 * 2
    beta['chronic*Content_health_info'] = 10.0 * 2

    d = len(beta)
    
    return beta, interaction_columns, d

# Step 4: 스케일링 및 라운딩 함수 정의
def scale(data, min_reward, max_reward):
    """데이터를 MIN_SCORE ~ MAX_SCORE 범위로 스케일링합니다."""
    scaled_data = MIN_SCORE + (data - min_reward) / (max_reward - min_reward) * (MAX_SCORE - MIN_SCORE)
    return scaled_data

def scale_and_round(data, min_reward, max_reward):
    """스케일링 후 1~5 Score로 라운딩합니다."""
    scaled_data = scale(data, min_reward, max_reward)
    probabilities = scaled_data - np.floor(scaled_data)
    random_values = np.random.rand(*data.shape)
    rounded_data = np.floor(scaled_data) + (random_values < probabilities)
    return np.clip(rounded_data, MIN_SCORE, MAX_SCORE).astype(int)

# Step 5: 시뮬레이션 데이터 준비
def prepare_simulation_data(df, X_user, X_item, beta, interaction_columns):
    """PHQ2 결과를 위한 시뮬레이션 데이터를 준비합니다."""
    X_user = df
    X_user_np = df.values
    X_item_np = X_item.values.astype(float)

    n_users, user_features = X_user_np.shape
    n_items, item_features = X_item_np.shape
    d = 1 + user_features + item_features + len(interaction_columns)

    def interaction(user_row, item_row):
        user_indices = [X_user.columns.get_loc(c.split('*')[0]) for c in interaction_columns]
        item_indices = [X_item.columns.get_loc(c.split('*')[1]) for c in interaction_columns]
        return user_row[user_indices] * item_row[item_indices]

    # 보상 범위 계산
    result_table = np.zeros((len(X_user), len(X_item)))
    for i in range(len(X_user)):
        for j in range(len(X_item)):
            user_row = X_user_np[i]
            item_row = X_item_np[j]
            product_row = interaction(user_row, item_row)
            x = np.concatenate(([1], user_row, item_row, product_row))
            reward = np.dot(x, beta)
            result_table[i, j] = reward

    max_reward = np.max(result_table)
    min_reward = np.min(result_table)
    max_arm = np.argmax(np.mean(result_table, axis=0))

    # 시뮬레이션 데이터 생성
    np.random.seed(SEED)
    X_ind = np.random.choice(n_users, size=(M, T))
    selected_users = X_user_np[X_ind]
    user_rows_exp = selected_users[:, :, np.newaxis, :].repeat(n_items, axis=2)
    item_features_exp = X_item_np[np.newaxis, np.newaxis, :, :].repeat(M, axis=0).repeat(T, axis=1)
    user_indices = np.array([X_user.columns.get_loc(c.split('*')[0]) for c in interaction_columns])
    item_indices = np.array([X_item.columns.get_loc(c.split('*')[1]) for c in interaction_columns])
    user_interaction_sim = selected_users[:, :, user_indices][:, :, np.newaxis, :]
    item_interaction_sim = X_item_np[:, item_indices][np.newaxis, np.newaxis, :, :]
    product_rows_sim = user_interaction_sim * item_interaction_sim
    ones_sim = np.ones((M, T, n_items, 1))
    X_simulation = np.concatenate([ones_sim, user_rows_exp, item_features_exp, product_rows_sim], axis=3)

    # PHQ2 결과 생성 (0~6 Score)
    Y_simulation = scale_and_round(np.dot(X_simulation, np.array(beta)), min_reward, max_reward)

    # Cyclic 시나리오
    X_cycle = np.zeros([M, T, d])
    for m in range(M):
        for t in range(T):
            X_cycle[m, t] = X_simulation[m, t, t % n_items]

    # 평균 보상 출력
    rewards = scale(np.dot(X_simulation, np.array(beta)), min_reward, max_reward)
    avg_reward = np.mean(np.max(rewards, axis=2))
    print(f'Average PHQ2 Score of Optimal Arms={avg_reward:.3f}.')
    print(f'Average PHQ2 Score of Cyclic Arms={np.mean(scale(np.dot(X_cycle, np.array(beta)), min_reward, max_reward)):.3f}.')

    return X_simulation, X_cycle, Y_simulation, max_arm, n_items, min_reward, max_reward

# Step 6: TS 클래스 및 헬퍼 함수
def score(beta, Y, X, lam=0):
    """선형 회귀를 위한 스코어 함수."""
    residual = Y - np.dot(X, beta)
    return np.dot(residual, X) - lam * beta

def score_dot(beta, X, lam=0):
    """스코어 함수의 도함수."""
    return -np.dot(X.T, X) - lam * np.eye(X.shape[1])

def sherman_morrison(u, Vinv):
    """Sherman-Morrison 공식을 사용한 역행렬 업데이트."""
    u = u.reshape(-1, 1)
    Vinv_u = Vinv @ u
    return Vinv - (Vinv_u @ Vinv_u.T) / (1 + u.T @ Vinv_u)

class TS:
    """Linear Thompson Sampling 클래스 (선형 결과용)."""
    def __init__(self, d, v):
        self.t = 0
        self.d = d
        self.beta_hat = np.zeros(d)
        self.Vinv = np.eye(d)
        self.v = v
        self.settings = {'v': v}
        self.X = []
        self.Y = []

    def select_ac(self, contexts):
        """Linear TS를 기반으로 최적 행동 선택."""
        contexts = np.array(contexts)
        self.t += 1
        V = (self.v ** 2) * self.Vinv
        beta_tilde = self.beta_hat + np.linalg.cholesky(V) @ np.random.randn(self.d)
        est = np.array([np.dot(contexts[i], beta_tilde) for i in range(len(contexts))])
        a_t = np.random.choice(np.where(est == est.max())[0])
        self.X.append(contexts[a_t])
        self.Vinv = sherman_morrison(contexts[a_t], self.Vinv)
        return a_t

    def update(self, reward):
        """보상을 통해 모델 업데이트 (선형 회귀)."""
        self.Y.append(reward)
        if self.t < 100 or self.t % 10 == 0:
            Y = np.array(self.Y)
            X = np.array(self.X).reshape(-1, self.d)
            f = lambda beta: score(beta, Y, X)
            fprime = lambda beta: score_dot(beta, X)
            self.beta_hat = root(f, x0=np.zeros(self.d), jac=fprime).x

# Step 7: 시뮬레이션 실행
def run_simulation(X_simulation, X_cycle, Y_simulation, max_arm, beta, d, min_reward, max_reward):
    """PHQ2 결과를 위한 Linear TS 시뮬레이션 실행."""
    v_set = [0.01, 0.1, 1.0]
    beta_np = np.array(beta)
    results_TS = []

    for v in v_set:
        print(f"v={v:.2f} Simulation")
        np.random.seed(SEED)

        regrets = np.zeros((M, T))
        rewards = np.zeros((M, T))
        collected_rewards = np.zeros((M, T))
        cyclic_rewards = np.zeros((M, T))
        single_rewards = np.zeros((M, T))
        cyclic_regrets = np.zeros((M, T))
        single_regrets = np.zeros((M, T))

        for m in tqdm(range(M)):
            model = TS(d=d, v=v)
            optRWD = []
            RWD_TS = []
            COLLECTED_TS = []

            for t in range(T):
                contexts = X_simulation[m, t]
                optRWD.append(np.amax(scale(np.dot(contexts, beta_np), min_reward, max_reward)))
                a_t = model.select_ac(contexts)
                rwd = Y_simulation[m, t, a_t]
                RWD_TS.append(scale(np.dot(contexts[a_t], beta_np), min_reward, max_reward))
                COLLECTED_TS.append(rwd)
                model.update(rwd)

                cyclic_rewards[m, t] = scale(np.dot(X_cycle[m, t], beta_np), min_reward, max_reward)
                single_rewards[m, t] = scale(np.dot(X_simulation[m, t, max_arm], beta_np), min_reward, max_reward)

            regrets[m] = np.cumsum(optRWD) - np.cumsum(RWD_TS)
            cyclic_regrets[m] = np.cumsum(optRWD) - np.cumsum(cyclic_rewards[m])
            single_regrets[m] = np.cumsum(optRWD) - np.cumsum(single_rewards[m])
            rewards[m] = RWD_TS
            collected_rewards[m] = COLLECTED_TS

        results_TS.append({
            'settings': model.settings,
            'regrets': regrets.tolist(),
            'collected_rewards': collected_rewards.tolist(),
            'rewards': rewards.tolist(),
            'cyclic_regrets': cyclic_regrets.tolist(),
            'single_regrets': single_regrets.tolist()
        })

    return results_TS

# Step 8: 결과 시각화
def visualize_results(results_TS, X_simulation, X_cycle, max_arm, beta, min_reward, max_reward):
    """CHS 데이터셋의 PHQ2 결과를 시각화합니다."""
    output_dir = 'results/PHQ/CHS'
    os.makedirs(output_dir, exist_ok=True)

    RT_TS = [np.mean(np.array(r['regrets'])[:, -1]) for r in results_TS]
    best_idx = np.argmin(RT_TS)
    regrets = np.array(results_TS[best_idx]['regrets'])
    avg_regrets = np.mean(regrets, axis=0)
    ste_regrets = np.std(regrets, axis=0) / np.sqrt(M)

    avg_cyclic_regrets = np.mean(np.array(results_TS[best_idx]['cyclic_regrets']), axis=0)
    ste_cyclic_regrets = np.std(np.array(results_TS[best_idx]['cyclic_regrets']), axis=0) / np.sqrt(M)
    avg_single_regrets = np.mean(np.array(results_TS[best_idx]['single_regrets']), axis=0)
    ste_single_regrets = np.std(np.array(results_TS[best_idx]['single_regrets']), axis=0) / np.sqrt(M)

    # 누적 Regret 플롯
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, T+1), y=avg_cyclic_regrets, mode='lines', name='Cyclic', line=dict(color='#6A5ACD')))
    fig.add_trace(go.Scatter(x=np.arange(1, T+1), y=avg_cyclic_regrets + ste_cyclic_regrets, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=np.arange(1, T+1), y=avg_cyclic_regrets - ste_cyclic_regrets, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(106,90,205,0.2)', showlegend=False))

    fig.add_trace(go.Scatter(x=np.arange(1, T+1), y=avg_single_regrets, mode='lines', name='Single', line=dict(color='#FFA07A')))
    fig.add_trace(go.Scatter(x=np.arange(1, T+1), y=avg_single_regrets + ste_single_regrets, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=np.arange(1, T+1), y=avg_single_regrets - ste_single_regrets, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,160,122,0.2)', showlegend=False))

    fig.add_trace(go.Scatter(x=np.arange(1, T+1), y=avg_regrets, mode='lines', name='Proposed', line=dict(color='#FF4500')))
    fig.add_trace(go.Scatter(x=np.arange(1, T+1), y=avg_regrets + ste_regrets, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=np.arange(1, T+1), y=avg_regrets - ste_regrets, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,69,0,0.2)', showlegend=False))

    fig.update_layout(
        title='Cumulative Regrets of PHQ2 Scores for CHS Dataset',
        xaxis_title='Number of Calls',
        yaxis_title='Cumulative Regret',
        yaxis_range=[0, 1.2 * max(avg_regrets)],
        template='plotly_white',
        font=dict(size=12),
        width=600, height=400,
        legend=dict(x=0.05, y=0.95, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=60, r=20, t=60, b=60),
        autosize=False
    )
    fig.write_image(os.path.join(output_dir, 'regret.pdf'))
    fig.write_image(os.path.join(output_dir, 'regret.png'))

    # 분포 플롯
    beta_np = np.array(beta)
    max_arm_dist = scale(np.mean(np.dot(X_simulation[:, :, max_arm, :], beta_np), axis=1), min_reward, max_reward)
    optimal_arm_dist = scale(np.mean(np.max(np.dot(X_simulation, beta_np), axis=2), axis=1), min_reward, max_reward)
    cycle_arm_dist = scale(np.mean(np.dot(X_cycle, beta_np), axis=1), min_reward, max_reward)
    TS_dist = np.mean(np.array(results_TS[best_idx]['rewards']), axis=1)  # 이미 스케일링된 값

    # 히스토그램
    fig_hist = go.Figure()
    bins = np.linspace(min([max_arm_dist.min(), optimal_arm_dist.min(), cycle_arm_dist.min(), TS_dist.min()]),
                       max([max_arm_dist.max(), optimal_arm_dist.max(), cycle_arm_dist.max(), TS_dist.max()]), 50)
    fig_hist.add_trace(go.Histogram(x=-cycle_arm_dist, name='Cyclic', opacity=0.8, marker_color='#6A5ACD', nbinsx=50))
    fig_hist.add_trace(go.Histogram(x=-max_arm_dist, name='Single', opacity=0.8, marker_color='#FFA07A', nbinsx=50))
    fig_hist.add_trace(go.Histogram(x=-optimal_arm_dist, name='Optimal', opacity=0.8, marker_color='#00CED1', nbinsx=50))
    fig_hist.add_trace(go.Histogram(x=-TS_dist, name='Proposed', opacity=0.8, marker_color='#FF4500', nbinsx=50))

    fig_hist.update_layout(
        title='Distribution of PHQ2 Scores for CHS Dataset (Histogram)',
        xaxis_title='PHQ2 Score',
        yaxis_title='Counts',
        barmode='overlay',
        template='plotly_white',
        font=dict(size=12),
        width=600, height=400,
        legend=dict(x=0.05, y=0.95, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=60, r=20, t=60, b=60),
        autosize=False
    )
    fig_hist.write_image(os.path.join(output_dir, 'dist.pdf'))
    fig_hist.write_image(os.path.join(output_dir, 'dist.png'))

    # 박스 플롯
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=-cycle_arm_dist, name='Cyclic', marker_color='#6A5ACD', boxpoints=False))
    fig_box.add_trace(go.Box(y=-max_arm_dist, name='Single', marker_color='#FFA07A', boxpoints=False))
    fig_box.add_trace(go.Box(y=-optimal_arm_dist, name='Optimal', marker_color='#00CED1', boxpoints=False))
    fig_box.add_trace(go.Box(y=-TS_dist, name='Proposed', marker_color='#FF4500', boxpoints=False))

    fig_box.update_layout(
        title='Distribution of PHQ2 Scores for CHS Dataset (Box-Plot)',
        yaxis_title='PHQ2 Score',
        template='plotly_white',
        font=dict(size=12),
        width=600, height=400,
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=60),
        autosize=False
    )
    fig_box.write_image(os.path.join(output_dir, 'boxplot.pdf'))
    fig_box.write_image(os.path.join(output_dir, 'boxplot.png'))

    # 요약 테이블
    result_reward_df = pd.DataFrame(columns=['Cyclic', 'Single', 'Optimal', 'Proposed'])
    result_reward_df.loc['Mean'] = [np.mean(cycle_arm_dist), np.mean(max_arm_dist), np.mean(optimal_arm_dist), np.mean(TS_dist)]
    result_reward_df.loc['Std'] = [np.std(cycle_arm_dist), np.std(max_arm_dist), np.std(optimal_arm_dist), np.std(TS_dist)]

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=['Metric'] + list(result_reward_df.columns), fill_color='lightgrey', align='left'),
        cells=dict(values=[result_reward_df.index] + [result_reward_df[col] for col in result_reward_df.columns],
                   fill_color='white', align='left', format=[''] + ['.2f']*4))
    ])
    fig_table.update_layout(width=600, height=200, title='Summary Statistics', margin=dict(l=20, r=20, t=50, b=20))
    fig_table.write_image(os.path.join(output_dir, 'summary_table.png'))

# Main 실행
if __name__ == "__main__":
    df = load_and_preprocess_data()
    X_user, X_item, X_user_columns, X_item_columns = define_variables()
    beta, interaction_columns, d = initialize_beta(df, X_user_columns, X_item_columns)
    X_simulation, X_cycle, Y_simulation, max_arm, n_items, min_reward, max_reward = prepare_simulation_data(
        df, X_user, X_item, beta, interaction_columns
    )
    results_TS = run_simulation(X_simulation, X_cycle, Y_simulation, max_arm, beta, d, min_reward, max_reward)
    visualize_results(results_TS, X_simulation, X_cycle, max_arm, beta, min_reward, max_reward)