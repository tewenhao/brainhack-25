from vmas.simulator.scenario import BaseScenario

class MyScenario(BaseScenario):

    def make_world(self, batch_dim, device, **kwargs):
        raise NotImplementedError()

    def reset_world_at(self, env_index):
        raise NotImplementedError()

    def observation(self, agent):
        raise NotImplementedError()

    def reward(self, agent):
        raise NotImplementedError()