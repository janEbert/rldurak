import sys
import threading
if sys.version_info[0] == 2:
    import Queue as queue
    range = xrange
elif sys.version_info[0] == 3:
    import queue
from os.path import isfile, exists
from random import choice, sample
from time import clock
from traceback import print_exc

import keras.backend as K
from keras.utils import plot_model
import numpy as np
import tensorflow as tf

import agent.actor as actor_m
import agent.critic as critic_m
import game.deck as deck
import game.player as player_m
import game.field as field
import game.game as game_m

episodes = 10000
# whether only AIs are in the game or one AI and random bots
only_ais = False
load = False # whether to load the models' weights
verbose = False # whether to print game progress
feature_type = 2 # 1, 2 or (unsupported) 3
# epsilon_start is the starting value for how often a random action is
# taken by AIs
# linearly anneals min_epsilon in the first epsilon_episodes episodes
min_epsilon = 0.1
epsilon_start = 1 # if not load else min_epsilon
epsilon_episodes = 6000
optimizer = 'adam' # 'adam' or 'rmsprop'
# learning rates
alpha_actor = 0.001
alpha_critic = 0.01
# numerical stability epsilon (recommended to change when using Adam!)
epsilon_actor = 1e-8
epsilon_critic = 1e-8
# update factors
tau_actor = 0.01
tau_critic = 0.01
# number of hidden neurons in each layer
n1_actor = 100
n1_critic = 100
n2_actor = 50
n2_critic = 50
gamma = 0.99 # discount factor
max_experience_count = 500 # amount of experiences to store
batch_size = 32 # amount of experiences to replay
win_reward = 12
loss_reward = -12
wait_reward = -0.05
illegal_action_reward = -100 # if >=0, do not reward illegal actions
# weights for difference in mean hand card value without trumps,
# difference in mean trump value and difference in trump amount
weights = (1, 2, 2)
# whether the features always contain 52 cards even though less are
# necessary (so that shape is the same for any amount of cards)
buffer_features = False
# how often random bots wait
# calculated from a normal distribution with the given values
psi_mu = 0.95
psi_sigma = 0.1
# how often bots check
# calculated from a normal distribution with the given values
chi_mu = 0.08
chi_sigma = 0.1
# whether the agent waits until all cards are defended before
# it attacks
wait_until_defended = True
action_shape = 5

# 'Kraudia' is added automatically if only_ais is false
names = ['Alice', 'Kraudia']
deck_size = 12
hand_size = 3
trump_suit = 2 # hearts (better not change this for consistency)


def main():
    """Main function for durak."""
    global psi, chi, epsilon
    durak_ix = -1
    wins = 0
    training_counter = 0
    completed_episodes = episodes
    for n in range(episodes):
        if not only_ais:
            psi = min(0.99, max(0, np.random.normal(psi_mu, psi_sigma)))
            chi = max(0, min(0.99, np.random.normal(chi_mu, chi_sigma)))
        create_game()
        reshuffle(hand_size)
        if durak_ix < 0:
            beginner_ix, beginner_card = game.find_beginner()
            if beginner_card == game.deck.bottom_trump:
                if verbose:
                    print('Beginner was chosen randomly\n')
        else:
            game.defender_ix = durak_ix
        try:
            result, training_counter_add = main_loop()
        except KeyboardInterrupt:
            clear_threads()
            print('Program was stopped by keyboard interrupt\n')
            completed_episodes = n
            break
        except:
            clear_threads()
            print_exc()
            print('')
            completed_episodes = n
            break
        training_counter += training_counter_add
        if not result:
            print('No action was retrieved in time. Program stopped\n')
            completed_episodes = n
            break
        durak_ix = names.index(game.players[0].name)
        if not only_ais:
            if game.kraudia_ix < 0:
                win_stats[n] = 1
                wins += 1
                if verbose:
                    print('Kraudia did not lose!\n')
            elif verbose:
                print('Kraudia is the durak...\n')
        if epsilon > min_epsilon:
            epsilon -= epsilon_step
        n_plus_one = n + 1
        if n_plus_one % 100 == 0:
            if not only_ais:
                print('Episode {0} ended. Total win rate: {1:.2f} %; win rate '
                        'over last 100 games: {2} %'.format(n_plus_one,
                        100 * wins / float(n_plus_one),
                        np.sum(win_stats[n_plus_one - 100:n_plus_one])))
            else:
                print('Episode {0} ended'.format(n_plus_one))
    return wins, completed_episodes, training_counter


def create_game():
    """Create a new game with the global parameters."""
    global game
    game = game_m.Game(names, deck_size, hand_size, trump_suit, feature_type,
            buffer_features, only_ais)


def reshuffle(hand_size):
    """Reshuffle if a player has more than the given hand size minus
    one cards of the same suit (except trump) in their hand.
    """
    hand_size -= 1
    for player in game.players:
        counts = [0] * 4
        for card in player.cards:
            counts[card.num_suit] += 1
        if (max(counts) >= hand_size
                and counts[game.deck.num_trump_suit] < hand_size):
            create_game()
            break
    while (max(counts) >= hand_size
            and counts[game.deck.num_trump_suit] < hand_size):
        for player in game.players:
            counts = [0] * 4
            for card in player.cards:
                counts[card.num_suit] += 1
            if (max(counts) >= hand_size
                    and counts[game.deck.num_trump_suit] < hand_size):
                create_game()
                break


def main_loop():
    """Main loop for receiving and executing actions and
    giving rewards.
    """
    training_counter = 0
    while not game.ended():
        if only_ais:
            hand_means = [game.hand_means(ix)
                    for ix in range(game.player_count)]
        else:
            hand_means = game.hand_means(game.kraudia_ix)
        active_player_indices = spawn_threads()
        first_attacker_ix = active_player_indices[0]
        last_experiences = {ix: None for ix in active_player_indices}
        while not game.attack_ended():
            try:
                player_ix, action = action_queue.get(timeout=10)
            except queue.Empty:
                clear_threads()
                return False, training_counter
            state = game.features.copy()
            if game.players[player_ix].checks or action[0] == 4:
                action_queue.task_done()
                if action[0] == 4:
                    if only_ais:
                        store_experience((state[player_ix], action,
                                wait_reward, state[player_ix]))
                    elif player_ix == game.kraudia_ix:
                        store_experience((state, action, wait_reward, state))
                    threads[active_player_indices.index(player_ix)].event.set()
                continue
            if only_ais or player_ix == game.kraudia_ix:
                if last_experiences[player_ix] is not None:
                    store_experience(last_experiences[player_ix])
                    last_experiences[player_ix] = None
            if verbose:
                print(action_to_string(player_ix, action))
            if action[0] == 0:
                if game.field.is_empty():
                    game.attack(player_ix, [make_card(action)])
                    if verbose:
                        print(game.field, '\n')
                    action_queue.task_done()
                    if game.is_winner(player_ix):
                        reward_winner(player_ix, state, action)
                        clear_threads()
                        if player_ix < first_attacker_ix:
                            first_attacker_ix -= 1
                        elif first_attacker_ix == game.player_count - 1:
                            first_attacker_ix = 0
                        last_experiences = remove_from_last_experiences(
                                last_experiences, player_ix)
                        if only_ais:
                            del hand_means[player_ix]
                        if game.remove_player(player_ix):
                            break
                        active_player_indices = spawn_threads()
                    elif only_ais:
                        last_experiences[player_ix] = (state[player_ix],
                                action, 0, game.features[player_ix])
                    elif player_ix == game.kraudia_ix:
                        last_experiences[player_ix] = (state, action, 0,
                                game.features)
                    for thread in threads:
                        thread.event.set()
                    continue
                else:
                    if game.exceeds_field([None], game.defender_ix):
                        action_queue.task_done()
                        threads[active_player_indices.index(
                                player_ix)].event.set()
                        continue
                    game.attack(player_ix, [make_card(action)])
                    for ix in active_player_indices[0::2]:
                        if game.players[ix].checks:
                            game.uncheck(ix)
                            threads[active_player_indices.index(
                                    ix)].event.set()
            elif action[0] == 1:
                to_defend, card = make_card(action)
                game.defend(to_defend, card)
                for ix in active_player_indices[0::2]:
                    if game.players[ix].checks:
                        game.uncheck(ix)
                        threads[active_player_indices.index(ix)].event.set()
            elif action[0] == 2:
                if game.exceeds_field([None]):
                    action_queue.task_done()
                    threads[active_player_indices.index(
                            player_ix)].event.set()
                    continue
                game.push([make_card(action)])
                action_queue.task_done()
                if game.is_winner(player_ix):
                    reward_winner(player_ix, state, action)
                    clear_threads()
                    if player_ix < first_attacker_ix:
                        first_attacker_ix -= 1
                    elif first_attacker_ix == game.player_count - 1:
                        first_attacker_ix = 0
                    last_experiences = remove_from_last_experiences(
                            last_experiences, player_ix)
                    if only_ais:
                        del hand_means[player_ix]
                    if game.remove_player(player_ix):
                        break
                else:
                    if only_ais:
                        last_experiences[player_ix] = (state[player_ix],
                                action, 0, game.features[player_ix])
                    elif player_ix == game.kraudia_ix:
                        last_experiences[player_ix] = (state, action, 0,
                                game.features)
                    clear_threads()
                active_player_indices = spawn_threads()
                if active_player_indices[2:]:
                    ix = active_player_indices[2]
                    if ix not in last_experiences:
                        last_experiences.update({ix: None})
                for thread in threads:
                    thread.event.set()
                if verbose:
                    print(game.field, '\n')
                continue
            elif action[0] == 3:
                game.check(player_ix)
            if verbose:
                print(game.field, '\n')
            action_queue.task_done()
            if game.is_winner(player_ix):
                reward_winner(player_ix, state, action)
                clear_threads()
                if player_ix < first_attacker_ix:
                    first_attacker_ix -= 1
                elif first_attacker_ix == game.player_count - 1:
                    first_attacker_ix = 0
                last_experiences = remove_from_last_experiences(
                        last_experiences, player_ix)
                if only_ais:
                    del hand_means[player_ix]
                if game.remove_player(player_ix):
                    break
                active_player_indices = spawn_threads()
                for thread in threads:
                    thread.event.set()
            else:
                if only_ais:
                    last_experiences[player_ix] = (state[player_ix], action, 0,
                            game.features[player_ix])
                elif player_ix == game.kraudia_ix:
                    last_experiences[player_ix] = (state, action, 0,
                            game.features)
                threads[active_player_indices.index(player_ix)].event.set()
        # attack ended
        clear_threads()
        if not game.ended():
            if only_ais:
                for ix in last_experiences:
                    if last_experiences[ix] is None:
                        last_experiences[ix] = (state[ix], game.check_action(),
                                None, None)
            elif (game.kraudia_ix in last_experiences
                    and last_experiences[game.kraudia_ix] is None):
                last_experiences[game.kraudia_ix] = (state,
                        game.check_action(), None, None)
            end_turn(first_attacker_ix, last_experiences, hand_means)
        training_counter += 1
        if verbose:
            print('Starting to learn from experiences...')
            train_from_memory()
            print('Finished learning')
        else:
            train_from_memory()
    return True, training_counter


def spawn_threads():
    """Spawn the action receiving threads for each active player and
    return the active players' indices and false.

    False is for a flag showing whether the threads have been cleared.
    """
    global threads
    active_player_indices = game.active_player_indices()
    threads = [spawn_thread(player_ix) for player_ix in active_player_indices]
    if verbose:
        print('')
    return active_player_indices


def spawn_thread(player_ix):
    """Spawn a thread for the given player index."""
    if verbose:
        print(player_ix, deck.cards_to_string(game.players[player_ix].cards))
    thread = ActionReceiver(player_ix)
    thread.start()
    return thread


def clear_threads():
    """Responsibly clear the list of threads."""
    for thread in threads:
        thread.ended = True
        game.check(thread.player_ix)
        thread.event.set()
        thread.join()
        game.uncheck(thread.player_ix)
    del threads[:]
    clear_queue()


def clear_queue():
    """Clear the action queue."""
    while not action_queue.empty():
        try:
            action_queue.get(timeout=1)
        except queue.Empty:
            continue
        action_queue.task_done()


def remove_from_last_experiences(last_experiences, player_ix):
    """Remove a player from the given list of last experiences."""
    update_ix = lambda ix: ix - 1 if player_ix < ix else ix
    del last_experiences[player_ix]
    return {update_ix(ix): last_experiences[ix] for ix in last_experiences}


def update_last_experience(last_experiences, player_ix, reward):
    """Update the last experience stored for the given player index
    with the given reward and current game state and remove it.
    """
    exp = last_experiences.pop(player_ix)
    if only_ais:
        store_experience((exp[0], exp[1], reward,
                game.features[player_ix]))
    else:
        store_experience((exp[0], exp[1], reward, game.features))
    return last_experiences


def reward_winner(player_ix, state, action):
    """Reward the player with the given index as a winner.

    Also reward loser if there is one already.
    """
    if only_ais:
        store_experience((state[player_ix], action, win_reward,
                game.features[player_ix]))
    elif player_ix == game.kraudia_ix:
        store_experience((state, action, win_reward, game.features))
    if game.will_end():
        if only_ais:
            store_experience((state[1 - player_ix], action, loss_reward,
                    game.features[1 - player_ix]))
        elif player_ix != game.kraudia_ix and game.kraudia_ix >= 0:
            store_experience((state, action, loss_reward, game.features))


def reward_winner_from_last_experience(last_experiences, player_ix):
    """Reward the player with the given index as a winner and update
    the last experiences accordingly.

    Also reward loser if there is one already."""
    exp = last_experiences[player_ix]
    if only_ais:
        store_experience((exp[0], exp[1], win_reward,
                game.features[player_ix]))
    elif player_ix == game.kraudia_ix:
        store_experience((exp[0], exp[1], win_reward, game.features))
    if game.will_end():
        exp = last_experiences[1 - player_ix]
        if only_ais:
            store_experience((exp[0], exp[1], loss_reward,
                    game.features[1 - player_ix]))
        elif player_ix != game.kraudia_ix and game.kraudia_ix >= 0:
            store_experience((exp[0], exp[1], loss_reward, game.features))
        return {}
    return remove_from_last_experiences(last_experiences, player_ix)


def train(state, action, reward, new_state):
    """Train the networks with the given states, action and reward."""
    target_q = critic.target_model.predict([new_state,
            actor.target_model.predict(new_state)])
    if reward == win_reward or reward == loss_reward:
        target = reward
    else:
        target = reward + gamma * target_q
    critic.model.train_on_batch([state, action], target)
    predicted_action = actor.model.predict(state)
    gradients = critic.get_gradients(state, predicted_action)
    actor.train(state, gradients)
    actor.train_target()
    critic.train_target()


def train_from_memory():
    """Train the networks with data from memory."""
    if len(experiences) >= batch_size:
        batch = sample(experiences, batch_size)
    else:
        batch = sample(experiences, len(experiences))
    states = np.asarray([experience[0] for experience in batch], dtype=np.int8)
    actions = np.asarray([experience[1] for experience in batch],
            dtype=np.int8)
    rewards = np.asarray([experience[2] for experience in batch])
    new_states = np.asarray([experience[3] for experience in batch],
            dtype=np.int8)
    target_qs = critic.target_model.predict([new_states,
            actor.target_model.predict(new_states, batch_size=len(batch))],
            batch_size=len(batch))
    targets = actions.copy()
    for i in range(len(batch)):
        if rewards[i] != win_reward and rewards[i] != loss_reward:
            targets[i] = rewards[i]
        else:
            targets[i] = rewards[i] + gamma * target_qs[i]
    critic.model.train_on_batch([states, actions], targets)
    predicted_actions = actor.model.predict(states)
    gradients = critic.get_gradients(states, predicted_actions)
    actor.train(states, gradients)
    actor.train_target()
    critic.train_target()


def end_turn(first_attacker_ix, last_experiences, hand_means):
    """End a turn by drawing cards for all attackers and the defender.

    Also give rewards.
    """
    if first_attacker_ix == game.defender_ix:
        first_attacker_ix += 1
    if first_attacker_ix == game.player_count:
        first_attacker_ix = 0
    player_ix = first_attacker_ix
    while player_ix != game.defender_ix:
        # first attacker till last attacker, then defender
        if game.is_winner(player_ix):
            last_experiences = reward_winner_from_last_experience(
                    last_experiences, player_ix)
            if game.remove_player(player_ix):
                return
        else:
            game.draw(player_ix)
            if game.will_end() and game.is_winner(1 - player_ix):
                last_experiences = reward_winner_from_last_experience(
                        last_experiences, 1 - player_ix)
                game.remove_player(1 - player_ix)
                return
            elif only_ais or player_ix == game.kraudia_ix:
                last_experiences = update_last_experience(last_experiences,
                        player_ix, hand_mean_reward(hand_means, player_ix))
            player_ix += 1
        if player_ix == game.player_count:
            player_ix = 0
    if first_attacker_ix != 0 and player_ix == game.player_count - 1:
        if game.is_winner(0):
            last_experiences = reward_winner_from_last_experience(
                    last_experiences, 0)
            if game.remove_player(0):
                return
        else:
            game.draw(0)
            if game.will_end() and game.is_winner(1):
                last_experiences = reward_winner_from_last_experience(
                        last_experiences, 1)
                game.remove_player(1)
                return
            elif only_ais or game.kraudia_ix == 0:
                last_experiences = update_last_experience(last_experiences, 0,
                        hand_mean_reward(hand_means, 0))
    elif (first_attacker_ix != player_ix + 1
            and player_ix != game.player_count - 1):
        if game.is_winner(player_ix + 1):
            last_experiences = reward_winner_from_last_experience(
                    last_experiences, player_ix + 1)
            if game.remove_player(player_ix + 1):
                return
        else:
            game.draw(player_ix + 1)
            if game.will_end() and game.is_winner(0):
                last_experiences = reward_winner_from_last_experience(
                        last_experiences, 0)
                game.remove_player(0)
                return
            elif only_ais or player_ix + 1 == game.kraudia_ix:
                last_experiences = update_last_experience(last_experiences,
                        player_ix + 1,
                        hand_mean_reward(hand_means, player_ix + 1))
    if game.field.attack_cards:
        amount = game.take()
        if only_ais or player_ix == game.kraudia_ix:
            last_experiences = update_last_experience(last_experiences,
                    player_ix, hand_mean_reward(hand_means, player_ix))
    else:
        game.clear_field()
        if game.is_winner(player_ix):
            last_experiences = reward_winner_from_last_experience(
                    last_experiences, player_ix)
            game.remove_player(player_ix)
        else:
            game.draw(player_ix)
            if only_ais or player_ix == game.kraudia_ix:
                last_experiences = update_last_experience(last_experiences,
                        player_ix, hand_mean_reward(hand_means, player_ix))
            game.update_defender()


def hand_mean_reward(hand_means, player_ix):
    """Return the mean reward change in the hand of the given player
    weighting trumps as more important."""
    if only_ais:
        avg_before, trump_avg_before, trump_count_before = \
                hand_means[player_ix]
    else:
        avg_before, trump_avg_before, trump_count_before = hand_means
    avg_after, trump_avg_after, trump_count_after = game.hand_means(player_ix)
    return (avg_after - avg_before) * norm_weights[0] \
            + (trump_avg_after - trump_avg_before) * norm_weights[1] \
            + (trump_count_after - trump_count_before) * norm_weights[2]


class ActionReceiver(threading.Thread):
    """Receive all actions for the given player for one round."""

    def __init__(self, player_ix):
        """Construct an action receiver with the given player index."""
        threading.Thread.__init__(self)
        self.player_ix = player_ix
        self.ended = False
        self.event = threading.Event()

    def run(self):
        """Add all actions for one round."""
        player = game.players[self.player_ix]
        if only_ais or self.player_ix == game.kraudia_ix:
            # first attacker
            if (self.player_ix == game.prev_neighbour(game.defender_ix)
                    and game.field.is_empty()):
                self.possible_actions = game.get_actions(self.player_ix)
                self.add_selected_action()
            self.get_extended_actions()
            # attacker
            if game.defender_ix != self.player_ix:
                if wait_until_defended:
                    defender = game.players[game.defender_ix]
                    while not self.ended:
                        # everything is defended
                        if ((not game.field.attack_cards or defender.checks)
                                and not player.checks):
                            self.add_selected_action()
                else:
                    while not self.ended:
                        # everything is defended
                        if not player.checks:
                            self.add_selected_action()
            # defender
            else:
                while not player.checks:
                    self.add_selected_action()
        else:
            # first attacker
            if (self.player_ix == game.prev_neighbour(game.defender_ix)
                    and game.field.is_empty()):
                self.possible_actions = game.get_actions(self.player_ix)
                self.add_action(choice(self.possible_actions))
            self.get_actions()
            # attacker
            if game.defender_ix != self.player_ix:
                defender = game.players[game.defender_ix]
                while not self.ended:
                    # everything is defended
                    if ((not game.field.attack_cards or defender.checks)
                            and not player.checks
                            and np.random.random() > psi):
                        self.add_random_action()
            # defender
            else:
                while not player.checks and self.possible_actions:
                    if np.random.random() > psi:
                        self.add_random_action()
                if not player.checks:
                    self.add_action(game.check_action())

    def get_extended_actions(self):
        """Wait until the game is updated and return a list of possible
        actions including checking and waiting.

        If the wait time is exceeded, return an empty list.
        """
        if not self.event.wait(1):
            self.possible_actions = []
        self.event.clear()
        self.possible_actions = game.get_actions(self.player_ix) \
                + [game.check_action(), game.wait_action()]

    def get_actions(self):
        """Wait until the game is updated and return a list of possible
        actions.

        If the wait time is exceeded, return an empty list.
        """
        if not self.event.wait(1):
            self.possible_actions = []
        self.event.clear()
        self.possible_actions = game.get_actions(self.player_ix)

    def add_action(self, action):
        """Add an action with the belonging player's index to the
        action queue.
        """
        action_queue.put((self.player_ix, action))

    def add_selected_action(self):
        """Add an action calculated by the model or a random one for
        exploration to the action queue.

        Also update the possible actions and store the experience.
        """
        if np.random.random() > epsilon:
            game.feature_lock.acquire()
            if only_ais:
                state = game.features[self.player_ix].copy()
            else:
                state = game.features.copy()
            game.feature_lock.release()
            action = [int(v) for v in actor.model.predict(
                    state.reshape(1, state.shape[0]))[0]]
            # TODO maybe remove?
            if action[0] in [0, 2, 3, 4]:
                action[3] = -1
                action[4] = -1
                if action[0] in [3, 4]:
                    action[1] = -1
                    action[2] = -1
            action = tuple(action)
            if action in self.possible_actions:
                self.add_action(action)
            elif illegal_action_reward < 0:
                store_experience((state, action, illegal_action_reward,
                        state))
                self.add_action(game.wait_action())
        elif self.possible_actions:
            self.add_action(choice(self.possible_actions))
        else:
            self.add_action(game.wait_action())
        self.get_extended_actions()

    def add_random_action(self):
        """Add a random action to the action queue or check at random.
        
        Also update the possible actions.
        """
        if np.random.random() > chi and self.possible_actions:
            self.add_action(choice(self.possible_actions))
            self.get_actions()
        else:
            self.add_action(game.check_action())
            self.get_actions()


def store_experience(experience):
    """Store an experience and overwrite old ones if necessary.

    An experience is a tuple consisting of (state, action, reward,
    new state). This function is threadsafe.
    """
    global experience_ix
    if len(experiences) == max_experience_count:
        experiences[experience_ix] = experience
        experience_ix += 1
        if experience_ix == max_experience_count:
            experience_ix = 0
    else:
        experiences.append(experience)


def action_to_string(player_ix, action):
    """Convert the given player's action to a string."""
    if action[0] < 3:
        if action[0] == 1:
            to_defend, card = make_card(action)
        else:
            card = make_card(action)
        string = (str(player_ix) + ': '
                + {0: 'Att', 1: 'Def', 2: 'Psh'}[action[0]] + ' '
                + str(card))
        if action[0] == 1:
            string += ' on ' + str(to_defend)
        return string
    elif action[0] == 3:
        return str(player_ix) + ': Chk'
    else:
        return str(player_ix) + ': Wait'


def make_card(action):
    """Create a card from an action.

    Create a tuple of two cards if action is defending.
    """
    if buffer_features:
        cards_per_suit = 13
    else:
        cards_per_suit = game.deck.cards_per_suit
    if action[0] == 1:
        return (deck.Card(action[3], action[4],
                action[3] - 13 + cards_per_suit + action[4] * cards_per_suit),
                deck.Card(action[1], action[2],
                action[1] - 13 + cards_per_suit + action[2] * cards_per_suit))
    return deck.Card(action[1], action[2],
            action[1] - 13 + cards_per_suit + action[2] * cards_per_suit)


if __name__ == '__main__':
    if not only_ais and 'Kraudia' not in names:
        names.append('Kraudia')
    assert len(names) == len(set(names)), 'Names must be unique'
    assert 0 not in weights, 'Weights cannot be zero'
    assert feature_type != 3, 'Feature type currently not supported'
    if episodes == 1:
        plural_s = ''
    else:
        plural_s = 's'
    if feature_type == 1:
        state_shape = deck_size + 3
    elif feature_type == 2:
        state_shape = (len(names) + 2) * deck_size + 4
    else:
        state_shape = 29
    weight_sum = weights[0] + weights[1] + weights[2]
    norm_weights = (weight_sum / float(weights[0] * 12),
            weight_sum / float(weights[1] * 12),
            weight_sum / float(weights[2] * hand_size))
    game = None
    psi = None
    chi = None
    threads = None
    action_queue = queue.Queue(len(names) * 6)
    epsilon = epsilon_start
    epsilon_step = (epsilon_start - min_epsilon) / float(epsilon_episodes)
    experiences = []
    experience_ix = 0
    if not only_ais:
        win_stats = np.zeros(episodes, dtype=np.int8)

    sess = tf.Session(config=tf.ConfigProto())
    K.set_session(sess)
    actor = actor_m.Actor(sess, state_shape, action_shape, load, optimizer,
            alpha_actor, epsilon_actor, tau_actor, n1_actor, n2_actor)
    critic = critic_m.Critic(sess, state_shape, action_shape, load, optimizer,
            alpha_critic, epsilon_critic, tau_critic, n1_critic, n2_critic)

    print('\nStarting to play\n')
    start_time = clock()
    wins, completed_episodes, training_counter = main()
    duration = clock() - start_time
    average_duration = duration
    win_rate = wins * 100
    # plot_model(actor, to_file='actor-' + str(state_shape) + '-features.png',
    #         show_shapes=True)
    # plot_model(critic, to_file='critic-' + str(state_shape)
    #         + '-features.png', show_shapes=True)
    if completed_episodes > 0:
        average_duration /= float(completed_episodes)
        win_rate /= float(completed_episodes)
    print('Finished {0}/{1} episode{2} after {3:.2f} seconds; '
            'average: {4:.2f} seconds per episode'.format(completed_episodes,
            episodes, plural_s, duration, average_duration))
    if not only_ais:
        print('Kraudia won {0}/{1} games which is a win rate of '
                '{2:.2f} %'.format(wins, completed_episodes, win_rate))
    print('The neural network was trained a total of {0} times'.format(
            training_counter))
    print('Saving data...')
    if not only_ais:
        if sys.version_info[0] == 2:
            prefix = '/media/data/jebert/'
            if not exists(prefix):
                prefix = ''
            file_name = prefix + 'win_stats_'
        elif sys.version_info[0] == 3:
            file_name = 'win_stats_'
        if completed_episodes != episodes:
            file_name += 'interrupted_during_' + str(completed_episodes + 1) \
                    + '_'
        file_int = 0
        while isfile(file_name + str(file_int) + '.npy'):
            file_int += 1
        try:
            np.save(file_name + str(file_int) + '.npy', win_stats,
                    allow_pickle=False)
        except IOError:
            print_exc()
            print('')
    if sys.version_info[0] == 2:
        file_name = prefix + 'actor-' + optimizer + '-' + str(state_shape) \
                + '-features.h5'
    elif sys.version_info[0] == 3:
        file_name = 'actor-' + optimizer + '-' + str(state_shape) \
                + '-features.h5'
    try:
        actor.save_weights(file_name)
    except IOError:
        print_exc()
        print('')
    if sys.version_info[0] == 2:
        file_name = prefix + 'critic-' + optimizer + '-' + str(state_shape) \
                + '-features.h5'
    elif sys.version_info[0] == 3:
        file_name = 'critic-' + optimizer + '-' + str(state_shape) \
                + '-features.h5'
    try:
        critic.save_weights(file_name)
    except IOError:
        print_exc()
        print('')
    print('Done')
