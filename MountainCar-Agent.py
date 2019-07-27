import gym
import random
import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

# Action Space - 0 = left, 1 = right, 2 = nothing
# Obv Space - 0 = horiz. pos. (-1.2 - 0.6), 1 = horiz. vel. (-0.07 - 0.07)

env = gym.make('MountainCar-v0')
env.reset()

goal_steps = 200
score_max = -200
initial_games = 10000

def data_prep():
    training_data = []
    accepted_scores = []
    for game in range(initial_games):
        score = 0
        game_mem = []
        prev_obs = []
        for step in range(goal_steps):
            action = random.randrange(0, 3)
            observation, reward, done, info = env.step(action)

            if len(prev_obs) > 0:
                game_mem.append([prev_obs, action])

            prev_obs = observation
            score += reward
            if done:
                break

            if score >= score_max:
                accepted_scores.append(score)

                for game in game_mem: # iterates through each game stored in memory
                    if game[1] == 0:
                        output = [1, 0, 0]
                    elif game[1] == 1:
                        output = [0, 1, 0]
                    elif game[1] == 2:
                        output = [0, 0, 1]

                    training_data.append([game[0], output])

            env.reset()
        print(accepted_scores)

        return training_data

def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(256, input_dim=input_size, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(optimizer=Adam(), loss='mse')

    return model

def train_model(training_data):
    x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_size=len(x[0]), output_size=len(y[0]))

    model.fit(x, y, epochs=6)
    return model


training_data = data_prep()
trained_model = train_model(training_data)

scores = []
choices = []
for game in range(100):
    score = 0
    prev_obs = []
    for step in range(goal_steps):

        if len(prev_obs) == 0:
            action = random.randrange(0, 3)
        else:
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])

        choices.append(action)
        new_obs, reward, done, info = env.step(action)
        env.render()

        prev_obs = new_obs
        score += reward
        if done:
            break

        env.reset()
        scores.append(score)

print(scores)
print('Average Score:', sum(scores) / len(scores))
