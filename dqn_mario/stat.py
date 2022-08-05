import matplotlib.pyplot as plt
import pandas as pd
from eps_controller import DecayEps
import numpy as np
from glob import glob
from icecream import ic

# https://github.com/garrettj403/SciencePlots
plt.style.use('science')

def plot_eps():
    episodes = np.arange(1, 1001)
    eps = DecayEps(1, 0.01, 50)
    eps_list = [eps(i) for i in episodes]
    # set y to log scale
    plt.figure(dpi=300)
    plt.yscale('log')
    plt.plot(episodes, eps_list)
    plt.xlabel('Episode')
    plt.ylabel(r'$\epsilon$')
    plt.title(r'Decay $\epsilon$-greedy')
    plt.savefig('eps.png')

def plot_ep_reward(path: str, title: str, df: pd.DataFrame):
    x, y = df['episode'], df['ep_reward']
    # last 10 episodes average
    rng = 50
    y_avg = np.array([np.mean(y[max(0, i-rng):i]) for i in range(len(y))])
    y_std = np.array([np.std(y[max(0, i-rng):i]) for i in range(len(y))])

    plt.figure(dpi=300)
    plt.plot(x, y_avg)
    # light blue area is standard deviation

    plt.fill_between(x, y_avg - y_std, y_avg + y_std, color='lightblue', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    # plt.show()
    plt.savefig(f'{path}.png')
    print(f'{path} saved')
    plt.close()


def plot_ep_loss(path: str, title: str, df: pd.DataFrame):
    x, y = df['episode'], df['loss']
    rng = 50
    y_avg = np.array([np.mean(y[max(0, i-rng):i]) for i in range(len(y))])
    y_std = np.array([np.std(y[max(0, i-rng):i]) for i in range(len(y))])
    plt.plot(x, y_avg)

    plt.fill_between(x, y_avg - y_std, y_avg + y_std, color='lightblue', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title(title)
    # plt.show()
    plt.savefig(f'{path}.png')
    print(f'{path} saved')
    plt.close()

if __name__ == '__main__':
    # plot_eps()
    # exit()
    for csv_path in glob('./models/cartpole-v1_mem=*.csv'):
        print(csv_path)
        prm = csv_path.split('_')[1].split('.')[0]
        df = pd.read_csv(csv_path)
        # distinct by episode
        df = df.drop_duplicates(subset='episode')
        plot_ep_reward(f'reward_{prm}', f'Episode Rewards ({prm})', df)
        plot_ep_reward(f'loss_{prm}', f'Episode Loss ({prm})', df)

    for csv_path in glob('./models/cartpole-v1_batch=*.csv'):
        print(csv_path)
        prm = csv_path.split('_')[1].split('.')[0]
        df = pd.read_csv(csv_path)
        # distinct by episode
        df = df.drop_duplicates(subset='episode')
        plot_ep_reward(f'reward_{prm}', f'Episode Rewards ({prm})', df)
        plot_ep_reward(f'loss_{prm}', f'Episode Loss ({prm})', df)