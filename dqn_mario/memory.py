import numpy as np

class SMBReplayMemory:
    def __init__(self, capa: int):
        self.capa = capa
        self.cusor = 0
        self.buffer = []

    def push(self, state: np.ndarray, action: int, reward: float, next_state, done):
        """
        push `env.step()` result
        """
        # state, next_state: (84, 84) -> (1, *)
        state_ = np.array(state.reshape(1, -1))
        next_state_ = np.array(next_state.reshape(1, -1))
        done_ = np.array(done)
        batch = (state_, action, reward, next_state_, done_)

        if len(self.buffer) < self.capa:
            self.buffer.append(batch)
        else:
            self.buffer[self.cusor] = batch
        # seek
        self.cusor = (self.cusor + 1) % self.capa

    def sample(self, batch_size: int):
        """
        sample from buffer
        """
        indices = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in indices]
        batch = tuple(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = np.array(batch[4])

        weights = np.array([1.0] * batch_size) / len(self.buffer)

        return states, actions, rewards, next_states, dones, indices, weights


    def __len__(self):
        return len(self.buffer)