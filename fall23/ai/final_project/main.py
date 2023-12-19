import gymnasium as gym
from gymnasium.utils.save_video import save_video
from models.learningagents import QTDLearningAgent, DoubleQLearningAgent
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from matplotlib.patches import Patch

#episodes are full playthroughs of 1 iteration of the game
n_episodes = 1_000_000

#gamma is .95 so that there's a preference for immediate action. Learning rate is .01 based on tuning
hyperparameters = {
    "gamma": .95,
    "learningrate": .01

}
#epsilon (random selection) decays from 100% to 10% after 50% of all episodes are completed.
hyperparameters["start_epsilon"] =  1.0
hyperparameters["epsilon_decay"] = hyperparameters["start_epsilon"] / (n_episodes/ 2)
hyperparameters["final_epsilon"] = .1


if __name__ == '__main__':
    #I run through the same sequence of training steps for two models DoubleQ Agent (an off-policy model)
    # as well as a baseline single Q Learning with Temporal Difference model for comparison.
    # Double Q learning agent was implemented according to van Hosselt et. all (2016)

    env = gym.make("Taxi-v3") # game will be taxiv3
    hyperparameters["env"] = env #I pass env and all hyperparams to the class constructor
    test_env = gym.make("Taxi-v3", render_mode="human") #used to record demo sequence
    dqagent = DoubleQLearningAgent(hyperparameters) #Double Q (off-policy) learning agent

    # below are used to aggregate session statistics
    meansDQN= [] #average number of steps per 1,0000 episode batch required to complete
    variancesDQN = [] #variance of meansDQN
    dqte = [] #training error for all sequential iterations averaged in 1000 episode batches
    n = 1000
    episode_avg = np.zeros((n,))
    te_avg = np.zeros((n,))
    for episode in tqdm(range(n_episodes)):
        if (episode + 1) % n == 0: #batch averaging step
            meansDQN.append(np.mean(episode_avg))
            variancesDQN.append(np.var(episode_avg))
            dqte.append(np.mean(te_avg))
        if episode == n_episodes -2:
            dqagent.env = test_env
        state, info = dqagent.env.reset() #init env

        terminated = False #sequence completes at step 200 (timeout) or when the objective is complete
        truncated = False #irrelevant but useful when extending to other games
        actionCount = 0 #track number of actions take in episode
        while not terminated and not truncated: #episode begin
            action = dqagent.get_action(state) # model selects next action given current state
            nextstate, reward, terminated, truncated, info = dqagent.env.step(action) #take action in environment
            dqagent.update(state, action, nextstate, reward, terminated) #policy iteration given observation
            state = nextstate #advance
            actionCount +=1
        #episode end and update statistics
        te_avg[episode % n] = np.mean(dqagent.training_error[-1*actionCount:])
        episode_avg[episode % n] = actionCount
        dqagent.decay_epsilon()


    #Process above is repeated for Q learning with TD agent with all same parameters and sequence
    qtdagent = QTDLearningAgent(hyperparameters)
    meansQTD = []
    variancesQTD = []
    qtdte = []
    episode_avg = np.zeros((n,))
    te_avg = np.zeros((n,))
    for episode in tqdm(range(n_episodes)):
        if (episode + 1) % n == 0:
            meansQTD.append(np.mean(episode_avg))
            variancesQTD.append(np.var(episode_avg))
            qtdte.append(np.mean(te_avg))
        if episode == n_episodes -2:
                qtdagent.env = test_env
        state, info = qtdagent.env.reset()

        terminated = False
        truncated = False
        actionCount = 0
        while not terminated and not truncated:

            action = qtdagent.get_action(state)
            nextstate, reward, terminated, truncated, info = qtdagent.env.step(action)
            qtdagent.update(state, action, nextstate, reward, terminated)
            state = nextstate
            actionCount += 1
        te_avg[episode % n] = np.mean(dqagent.training_error[-1 * actionCount:])
        episode_avg[episode % n] = actionCount
        qtdagent.decay_epsilon()
    env.close()
    test_env.close()

    #plotting relevant statistics (mean, variance, and avg training error) for both models
    fig, axs = plt.subplots(3,2)

    axs[0][0].scatter(np.arange(len(meansQTD)), meansQTD)
    axs[0][0].set_title("meansQTD")
    axs[0][0].set_yticks(np.linspace(0, max(meansQTD),5))
    axs[0][0].set_ylabel("Average num states in episode")

    axs[1][0].scatter(np.arange(len(variancesQTD)), variancesQTD)
    axs[1][0].set_title("variancesQTD")
    axs[1][0].set_yticks(np.linspace(0, max(variancesQTD),5))
    axs[1][0].set_ylabel("variance of ep in len")

    axs[0][1].scatter(np.arange(len(meansDQN)), meansDQN )
    axs[0][1].set_title("meansDQN")
    axs[0][1].set_yticks(np.linspace(0, max(meansDQN),5))

    axs[1][1].scatter(np.arange(len(variancesDQN)), variancesDQN)
    axs[1][1].set_title("variancesDQN")
    axs[1][1].set_yticks(np.linspace(0, max(variancesDQN),5))

    axs[2][0].scatter(np.arange(len(qtdte)), qtdte)
    axs[2][0].set_title("QTD training error")
    axs[2][0].set_yticks(np.linspace(min(0,min(qtdte)), max(max(qtdte),0), 5))
    axs[2][0].set_ylabel("Loss")

    axs[2][1].scatter(np.arange(len(dqte)), dqte)
    axs[2][1].set_title("DQN training error")
    axs[2][1].set_yticks(np.linspace(min(0,min(dqte)), max(max(dqte),0), 5))

    fig.tight_layout()
    plt.show()
    plt.legend()
    # value_grid, policy_grid = create_grids(agent, usable_ace=True)
    # fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
    # plt.show()
    env.close()
    test_env.close()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
