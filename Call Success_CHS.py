#!/usr/bin/env python3
"""
Thompson Sampling을 활용한 Contextual Bandit 문제를 시뮬레이션하는 Python 스크립트입니다.
전화 성공률(Call Success Rate)을 최적화하며, TS와 Cyclic/Single 방식의 성능을 비교합니다.
결과는 Plotly를 통해 시각화되며, 'results/success/CHS' 디렉토리에 저장됩니다.
CHS: Community Health Survey
"""

# Variables Description
"""
sex: Gender
    0 - Male (남성)
    1 - Female (여성)

age: Age
    0 - Younger (적음)
    1 - Older (많음, 75세 이상)

family: Household Type
    0 - Not living alone (독거가 아님 - 부부, 자녀, 기타)
    1 - Living alone (독거)

econ: Economic Status
    0 - Not poor (가난하지 않음 - 경제적 안정)
    1 - Poor (가난함 - 경제적 불안정)

region: Region
    0 - Rural (농촌)
    1 - Urban (도시)

difficulty: Difficulty in Daily Life
    0 - No difficulty (어려움 없음)
    1 - Has difficulty (어려움 있음)

chronic: Number of Chronic Diseases
    0 - Less than 3 (3개 미만)
    1 - 3 or more (3개 이상)

cognition: Cognitive Function
    0 - Normal (정상)
    1 - Impaired (인지 어려움 있음)

lonely: Loneliness
    0 - Not lonely (안 외로움)
    1 - Lonely (외로움을 느낌)
"""

# 필요한 라이브러리 임포트
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import product
from scipy.optimize import root
from scipy.special import expit
import plotly.graph_objects as go

# 결과 저장 디렉토리 생성
os.makedirs('results/success', exist_ok=True)

# 시뮬레이션 파라미터 설정
T = 1000  # Number of timesteps/subjects
M = 100    # Number of experimental repetitions
SEED = 0  # Seed for random number generation (Success Rate용)

# Step 1: 가상 데이터 생성
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

# Step 2: Define user and item variables
def define_variables():
    """Define user and item variables and generate combinations."""
    user_vars = ['sex', 'family', 'econ', 'region', 'age', 'chronic', 'difficulty', 'cognition', 'lonely']
    user_values = list(product([0, 1], repeat=len(user_vars)))
    X_user = pd.DataFrame(user_values, columns=user_vars)

    content = ['intervention', 'health_info', 'check-in']
    call_frequency = [0, 0.5, 1]  # Scaled frequency
    call_duration = [0, 0.5, 1]   # Scaled duration
    item_values = list(product(content, call_frequency, call_duration))
    X_item = pd.DataFrame(item_values, columns=['Content', 'Call_frequency', 'Call_duration'])
    X_item = pd.get_dummies(X_item, columns=['Content'])  # One-hot encoding

    return X_user, X_item, user_vars, X_item.columns.tolist()

# Step 3: Define interactions and initialize beta coefficients
def initialize_beta(df, X_user_columns, X_item_columns):
    """Interaction 정의 및 Beta 계수 초기화."""
    interaction_dict = {
        '설명': [
            '여성들은 길이가 긴 call을 선호할 것이다.',
            '일상생활제한이 있는 사람들은 긴 call을 선호할 것이다',
            '시골에 사는 사람들은 전화 횟수가 많을수록 만족할 것이다.',
            '나이가 많은 사람들은 check-in 콜을 선호할 것이다.',
            '인지기능이 낮은 사람은 check-in call을 선호할 것이다',
            '외로운 사람들은 intervention 콜을 덜 선호할 것이다.',
            '만성질환이 있는 사람은 health_info call을 선호할 것이다',
        ]
    }
    interaction_table = pd.DataFrame(interaction_dict)
    print("\nInteraction Scenarios:")
    print(interaction_table)

    interaction_columns = [
        'age*Content_check-in', 'region*Call_frequency', 'sex*Call_duration',
        'lonely*Content_intervention', 'difficulty*Call_duration',
        'cognition*Content_check-in', 'chronic*Content_health_info'
    ]
    beta_index = ['intercept'] + X_user_columns + X_item_columns + interaction_columns

    np.random.seed(SEED)
    beta = pd.Series({var: np.random.uniform(-2, 2) for var in beta_index})
    beta['intercept'] = 0

    beta['Call_frequency'] = 5.0
    beta['Call_duration'] = -10.0
    beta['Content_check-in'] = -10.0
    beta['Content_intervention'] = 5.0
    beta['Content_health_info'] = -5.0

    beta['region*Call_frequency'] = -10.0
    beta['sex*Call_duration'] = 10.0
    beta['difficulty*Call_duration'] = 10.0
    beta['age*Content_check-in'] = 10.0
    beta['cognition*Content_check-in'] = 10.0
    beta['lonely*Content_intervention'] = -10.0
    beta['chronic*Content_health_info'] = 10.0

    d = len(beta)
    beta = beta / np.sqrt(d) * 2

    return beta, interaction_columns, d

# Step 4: Define cost function
def define_cost_function(beta):
    """Define cost function based on frequency and duration."""
    freq_ind = beta.index.get_loc('Call_frequency')
    duration_ind = beta.index.get_loc('Call_duration')

    def cost_function(context):
        return context[..., freq_ind] * context[..., duration_ind]

    return cost_function

# Step 5: Prepare simulation data
def prepare_simulation_data(df, X_user, X_item, beta, interaction_columns, cost_function):
    """Prepare data for simulation."""
    X_user_np = X_user.values
    X_item_np = X_item.values.astype(float)

    n_users, user_features = X_user_np.shape
    n_items, item_features = X_item_np.shape
    d = 1 + user_features + item_features + len(interaction_columns)

    user_indices = np.array([X_user.columns.get_loc(c.split('*')[0]) for c in interaction_columns])
    item_indices = np.array([X_item.columns.get_loc(c.split('*')[1]) for c in interaction_columns])

    user_rows = X_user_np[:, np.newaxis, :].repeat(n_items, axis=1)
    item_rows = X_item_np[np.newaxis, :, :].repeat(n_users, axis=0)
    user_interaction = X_user_np[:, np.newaxis, user_indices]
    item_interaction = X_item_np[np.newaxis, :, item_indices]
    product_rows = user_interaction * item_interaction
    ones = np.ones((n_users, n_items, 1))
    x = np.concatenate([ones, user_rows, item_rows, product_rows], axis=2)

    rewards = expit(np.dot(x, beta))
    costs = np.apply_along_axis(cost_function, 2, x)
    result_table = np.stack([rewards, costs], axis=2)
    max_arm = np.argmax(np.mean(result_table[:, :, 0], axis=0))

    np.random.seed(SEED)
    X_ind = np.random.choice(n_users, size=(M, T))
    selected_users = X_user_np[X_ind]
    user_rows_exp = selected_users[:, :, np.newaxis, :].repeat(n_items, axis=2)
    item_features_exp = X_item_np[np.newaxis, np.newaxis, :, :].repeat(M, axis=0).repeat(T, axis=1)
    user_interaction_sim = selected_users[:, :, user_indices][:, :, np.newaxis, :]
    item_interaction_sim = X_item_np[:, item_indices][np.newaxis, np.newaxis, :, :]
    product_rows_sim = user_interaction_sim * item_interaction_sim
    ones_sim = np.ones((M, T, n_items, 1))
    X_simulation = np.concatenate([ones_sim, user_rows_exp, item_features_exp, product_rows_sim], axis=3)

    X_cycle = np.zeros([M, T, d])
    for m in range(M):
        for t in range(T):
            X_cycle[m, t] = X_simulation[m, t, t % n_items]

    return X_simulation, X_cycle, max_arm, n_items

# Step 6: TS class and helper functions
def mu(x):
    """Sigmoid function (mean)."""
    return expit(x)

def mu_dot(x):
    """Derivative of sigmoid function."""
    mu_val = expit(x)
    return mu_val * (1 - mu_val)

def score(beta, Y, X, lam=0):
    """Score function for optimization."""
    residual = Y - mu(np.dot(X, beta))
    return np.dot(residual, X) - lam * beta

def score_dot(beta, X, lam=0):
    """Derivative of score function."""
    mm = mu_dot(X @ beta)
    return -np.einsum('b,bi,bj -> ij', mm, X, X) - lam * np.eye(X.shape[-1])

def sherman_morrison(X, V, w=1):
    """Sherman-Morrison formula for inverse update."""
    X_V = X @ V
    denominator = 1 + w * X @ X_V
    return V - (w / denominator) * np.outer(X_V, X_V)

class TS:
    """Thompson Sampling algorithm class."""
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
        """Select optimal action."""
        self.t += 1
        V = (self.v ** 2) * self.Vinv
        beta_tilde = self.beta_hat + np.linalg.cholesky(V) @ np.random.randn(self.d)
        est = expit(np.dot(contexts, beta_tilde))
        a_t = np.random.choice(np.where(est == est.max())[0])
        self.X.append(contexts[a_t])
        self.Vinv = sherman_morrison(contexts[a_t], self.Vinv)
        return a_t

    def update(self, reward):
        """Update model with reward."""
        self.Y.append(reward)
        if self.t < 100 or self.t % 10 == 0:
            Y = np.array(self.Y)
            X = np.array(self.X).reshape(-1, self.d)
            f = lambda beta: score(beta, Y, X)
            fprime = lambda beta: score_dot(beta, X)
            self.beta_hat = root(f, x0=np.zeros(self.d), jac=fprime).x

# Step 7: Run simulation
def run_simulation(X_simulation, X_cycle, max_arm, beta, d):
    """Execute TS simulation."""
    v_set = [0.01, 0.1, 1.0]
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
            optRWD = np.zeros(T)
            RWD_TS = np.zeros(T)
            COLLECTED_TS = np.zeros(T)

            for t in range(T):
                contexts = X_simulation[m, t]
                optRWD[t] = np.amax(mu(np.dot(contexts, beta)))
                a_t = model.select_ac(contexts)
                context_dot_beta = np.dot(contexts[a_t], beta)
                rwd = np.random.binomial(n=1, p=mu(context_dot_beta))
                RWD_TS[t] = mu(context_dot_beta)
                COLLECTED_TS[t] = rwd
                model.update(rwd)

                cyclic_rewards[m, t] = mu(np.dot(X_cycle[m, t], beta))
                single_rewards[m, t] = mu(np.dot(X_simulation[m, t, max_arm], beta))

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

# Step 8: Visualize results
def visualize_results(results_TS, X_simulation, X_cycle, max_arm, beta):
    """Visualize simulation results for CHS dataset."""
    output_dir = 'results/success/CHS'
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

    # Cumulative regret plot
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
        title='Cumulative Regrets of Call Success Rates for CHS Dataset',
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

    # Average reward
    mean_TS = np.mean(np.array(results_TS[best_idx]['rewards']))
    print(f"Average Reward of TS for CHS: {mean_TS:.3f}")

    # Distribution plots
    max_arm_dist = 100 * np.mean(mu(np.dot(X_simulation[:, :, max_arm, :], beta)), axis=1)
    optimal_arm_dist = 100 * np.mean(np.max(mu(np.dot(X_simulation, beta)), axis=2), axis=1)
    cycle_arm_dist = 100 * np.mean(mu(np.dot(X_cycle, beta)), axis=1)
    TS_dist = 100 * np.mean(np.array(results_TS[best_idx]['rewards']), axis=1)

    # Histogram
    fig_hist = go.Figure()
    bins = np.linspace(min([max_arm_dist.min(), optimal_arm_dist.min(), cycle_arm_dist.min(), TS_dist.min()]),
                       max([max_arm_dist.max(), optimal_arm_dist.max(), cycle_arm_dist.max(), TS_dist.max()]), 100)
    fig_hist.add_trace(go.Histogram(x=cycle_arm_dist, name='Cyclic', opacity=0.8, marker_color='#6A5ACD', nbinsx=100))
    fig_hist.add_trace(go.Histogram(x=max_arm_dist, name='Single', opacity=0.8, marker_color='#FFA07A', nbinsx=100))
    fig_hist.add_trace(go.Histogram(x=optimal_arm_dist, name='Optimal', opacity=0.8, marker_color='#00CED1', nbinsx=100))
    fig_hist.add_trace(go.Histogram(x=TS_dist, name='Proposed', opacity=0.8, marker_color='#FF4500', nbinsx=100))

    fig_hist.update_layout(
        title='Distribution of Call Success Rates for CHS Dataset (Histogram)',
        xaxis_title='Call Success Rate (%)',
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

    # Box plot
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=cycle_arm_dist, name='Cyclic', marker_color='#6A5ACD', boxpoints=False))
    fig_box.add_trace(go.Box(y=max_arm_dist, name='Single', marker_color='#FFA07A', boxpoints=False))
    fig_box.add_trace(go.Box(y=optimal_arm_dist, name='Optimal', marker_color='#00CED1', boxpoints=False))
    fig_box.add_trace(go.Box(y=TS_dist, name='Proposed', marker_color='#FF4500', boxpoints=False))

    fig_box.update_layout(
        title='Distribution of Call Success Rates for CHS Dataset (Box-plot)',
        yaxis_title='Call Success Rate (%)',
        template='plotly_white',
        font=dict(size=12),
        width=600, height=400,
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=60),
        autosize=False
    )
    fig_box.write_image(os.path.join(output_dir, 'boxplot.pdf'))
    fig_box.write_image(os.path.join(output_dir, 'boxplot.png'))

    # Summary table
    result_reward_df = pd.DataFrame(columns=['Cyclic', 'Single', 'Optimal', 'Proposed'])
    result_reward_df.loc['Mean'] = [np.mean(cycle_arm_dist), np.mean(max_arm_dist), np.mean(optimal_arm_dist), np.mean(TS_dist)]
    result_reward_df.loc['Std'] = [np.std(cycle_arm_dist), np.std(max_arm_dist), np.std(optimal_arm_dist), np.std(TS_dist)]

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=['Metric'] + list(result_reward_df.columns), fill_color='lightgrey', align='left'),
        cells=dict(values=[result_reward_df.index] + [result_reward_df[col] for col in result_reward_df.columns],
                   fill_color='white', align='left', format=[''] + ['.2f']*4))
    ])
    fig_table.update_layout(width=600, height=200, title='Summary Statistics for CHS Dataset', margin=dict(l=20, r=20, t=50, b=20))
    fig_table.write_image(os.path.join(output_dir, 'summary_table.png'))

# Main execution
if __name__ == "__main__":
    df = load_and_preprocess_data()
    X_user, X_item, X_user_columns, X_item_columns = define_variables()
    beta, interaction_columns, d = initialize_beta(df, X_user_columns, X_item_columns)
    cost_function = define_cost_function(beta)
    X_simulation, X_cycle, max_arm, n_items = prepare_simulation_data(df, X_user, X_item, beta, interaction_columns, cost_function)
    results_TS = run_simulation(X_simulation, X_cycle, max_arm, beta, d)
    visualize_results(results_TS, X_simulation, X_cycle, max_arm, beta)