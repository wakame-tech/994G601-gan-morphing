import pandas as pd


def read_log_csv(course: str, mem: int):
    return pd.read_csv(f'./models/smb-dqn-{course}_mem={mem}_log.csv')

def tally_x_pos(df: pd.DataFrame):
    return df.groupby('episode')['x_pos'].max().mean()

def tally_reward(df: pd.DataFrame):
    return df.groupby('episode')['reward'].mean()

if __name__ == '__main__':
    # course = '1-4'
    course = '2-2'
    for mem in [1000, 10000, 100000]:
        df = read_log_csv(course, mem)
        res = tally_x_pos(df)
        print(f'{course}: {mem=} {res=}')
        res = tally_reward(df)
        print(f'{course}: {mem=} {res=}')