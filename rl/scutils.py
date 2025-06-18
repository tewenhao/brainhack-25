import numpy as np
from collections import OrderedDict
sc_utils_key_name = ["direction","location","scout","step","viewcone"]
def extract_obs(original_obs, indexes:list,new_obs=None):
    n_obs = len(original_obs["direction"])
    if new_obs == None:
        new_obs = OrderedDict([(key,None) for key in sc_utils_key_name])
    indexes.sort()
    # for index in indexes[::-1]:
    for key in sc_utils_key_name:
        if new_obs[key] == None:
            print(key,np.take(original_obs[key],indexes))
            new_obs[key] = np.take(original_obs[key],indexes,axis = 0)
        else:
            new_obs[key]= np.concatenate(new_obs[key],np.take(original_obs[key],indexes,axis = 0),axis=0)
        original_obs[key] = np.delete(original_obs[key],indexes,axis = 0)
    return new_obs
def adapt_obs(viewcone_obs):
    for i in range(len(viewcone_obs)):
        for j in range(len(viewcone_obs[i])):
            old = viewcone_obs[i][j]
            viewcone_obs[i][j] = [old>>4,(old>>3)&1,(old>>2)&1,old&3]
    return viewcone_obs
transformations = []
def trans1(old):
    return old>>4
def trans2(old):
    return (old>>3)&1
def trans3(old):
    return (old>>2)&1
def trans4(old):
    return (old&3)
transformation_vector = []
trans1_vector = np.vectorize(trans1)
trans2_vector = np.vectorize(trans2)
trans3_vector = np.vectorize(trans3)
trans4_vector = np.vectorize(trans4)
transformation_vector = [trans1_vector,trans2_vector,trans3_vector,trans4_vector]
transformations = [trans1,trans2,trans3,trans4]
def split_obs(viewcone_obs:np.ndarray):
    global transformations
    channels = np.ndarray((4,viewcone_obs.shape[0],viewcone_obs.shape[1]),dtype=np.uint8)
    for i in range(len(transformations)):
        channels[i] = np.copy(viewcone_obs)
        channels[i] = transformation_vector[i](channels[i])
    return channels
def env_that_ended(info):
    answer = []
    for i in range(len(info)):
        if info[i].get("terminal_observation"):
            answer.append(i)
    return answer