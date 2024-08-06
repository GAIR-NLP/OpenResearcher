import pickle
with open("example_history/multimodal.pkl", 'rb') as file:
    multimodal = pickle.load(file)

with open("example_history/wizard lm.pkl", 'rb') as file:
    wizard_lm = pickle.load(file)

with open("example_history/what is ppo.pkl", 'rb') as file:
    what_is_ppo = pickle.load(file)