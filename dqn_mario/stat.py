import matplotlib.pyplot as plt
import pandas as pd
from eps_controller import DecayEps

import numpy as np



def read_log_csv(course: str, mem: int):
    return pd.read_csv(f'./models/smb-dqn-{course}_mem={mem}_log.csv')

def tally_x_pos(df: pd.DataFrame):
    return df.groupby('episode')['x_pos'].max().mean()

def tally_reward(df: pd.DataFrame):
    rewards = df.groupby('episode')['ep_reward'].max().to_list()
    plt.plot(rewards)
    plt.show()

def stat_eps():
    eps_ctrl = DecayEps(1.0, 0.01, 300)
    epsodes = np.arange(1, 1001)
    eps = [eps_ctrl(e) for e in epsodes]
    plt.plot(epsodes, eps)

if __name__ == '__main__':
    # course = '1-4'
    df = pd.read_csv('./models/cartpole-dqn_log.csv')
    tally_reward(df)
    exit()
    course = '2-2'
    for mem in [1000, 10000, 100000]:
        df = read_log_csv(course, mem)
        res = tally_x_pos(df)
        print(f'{course}: {mem=} {res=}')
        res = tally_reward(df)
        print(f'{course}: {mem=} {res=}')