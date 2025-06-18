

import gridworld
import time
from hardcode import Scout
env = gridworld.env(env_wrappers=[],render_mode="human",novice=False)
# print(env.gridworld())
env.reset(seed = 20)
# print(env.legal_actions)
scout_agent = env.scout
scout_agent = int(scout_agent[-1])
for i in range(4):
    if scout_agent != i:
        first = i
        break
# first = -1
print(first,scout_agent)
# k+=1
agents = []
for i in range(4):
    print((scout_agent == i),(first == i))
    agents.append(Scout(is_scout = (scout_agent == i),first = (first == i)))
print(agents[0] is agents[1])
obs = env.last()[0]
# k+=1
print(agents[0].get_action(obs))
for episodes in range(1):

    for steps in range(50):
        
        for agent in range(4):
            obs = env.last()[0]
            # if agent == scout_agent:
            #     env.step(4)
            #     continue
            # .get_action(obs)
            action = agents[agent].get_action(obs)
            
            env.step(action)
        agents[0]._render_frame()
        time.sleep(1)
        # input()
print("end")
agents[scout_agent].setup_seeds()

        # print(reward,done,truncated,end ="\t")
    # input()
    # print()
    # time.sleep(2)
    # print(_[0].size)
    # print(_)
# print(env.observation)
# print(env.legal_actions)
input()