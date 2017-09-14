import sys
import threading
if sys.version_info[0] == 2:
    import Queue
    range = xrange
elif sys.version_info[0] == 3:
    import queue
from os.path import isfile
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

episodes = 10
# whether only AIs are in the game or one AI and random bots
only_ais = False
load = False # whether to load the models' weights
verbose = True # whether to print game progress
feature_type = 2 # 1, 2 or (unsupported) 3
# starting value for how often a random action is taken by AIs
# linearly anneals min_epsilon in the first epsilon_count actions
min_epsilon = 0.1
epsilon = 1 # if not load else min_epsilon
epsilon_count = 6000
# learning rates
alpha_actor = 0.001
alpha_critic = 0.001
# update factors
tau_actor = 0.001
tau_critic = 0.001
# number of hidden neurons in each layer
n1_actor = 100
n1_critic = 100
n2_actor = 150
n2_critic = 150
gamma = 0.99 # discount factor
max_experience_count = 500 # amount of experiences to store
batch_size = 32 # amount of experiences to replay
win_reward = 70
loss_reward = -70
wait_reward = -1
illegal_action_reward = -100
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
action_shape = 5

names = ['Alice', 'Bob']
deck_size = 52 # cannot be changed at the moment
hand_size = 6
trump_suit = 2 # hearts


def main():
    """Main function for durak."""
    global durak_ix, game, psi, chi
    wins = 0
    completed_episodes = episodes
    for n in range(episodes):
        if not only_ais:
            psi = min(0.98, np.random.normal(psi_mu, psi_sigma))
            chi = max(0, np.random.normal(chi_mu, chi_sigma))
        game = create_game()
        reshuffle(hand_size)
        if durak_ix < 0:
            beginner_ix, beginner_card = game.find_beginner()
            if beginner_card == game.deck.bottom_trump:
                if verbose:
                    print('Beginner was chosen randomly\n')
        else:
            game.defender_ix = durak_ix
        try:
            main_loop()
        except KeyboardInterrupt:
            clear_threads()
            clear_queue()
            print('Program was stopped by keyboard interrupt\n')
            completed_episodes = n
            break
        except:
            clear_threads()
            clear_queue()
            print_exc()
            print('')
            completed_episodes = n
            break
        if not result:
            print('No action was retrieved in time. Program stopped\n')
            completed_episodes = n
            break
        durak_ix = names.index(game.players[0].name)
        if game.kraudia_ix < 0:
            win_stats[n] = 1
            wins += 1
            if verbose:
                print('Kraudia did not lose!\n')
        else:
            if verbose:
                print('Kraudia is the durak...\n')
        if n != 0 and n % 100 == 0:
            print('Episode {0} ended. Win rate: {1:.2f}'.format(n + 1,
                    win_rate / float(n)))
    return wins, completed_episodes


def create_game():
    """Create a new game with the global parameters."""
    return game_m.Game(names, deck_size, hand_size, trump_suit, feature_type,
            buffer_features, only_ais)


def reshuffle(hand_size):
    """Reshuffle if a player has more than the given hand size minus
    one cards of the same suit (except trump) in their hand.
    """
    global game
    hand_size -= 1
    for player in game.players:
        counts = [0] * 4
        for card in player.cards:
            counts[card.num_suit] += 1
        if (max(counts) >= hand_size
                and counts[game.deck.num_trump_suit] < hand_size):
            game = create_game()
            break
    while (max(counts) >= hand_size
            and counts[game.deck.num_trump_suit] < hand_size):
        for player in game.players:
            counts = [0] * 4
            for card in player.cards:
                counts[card.num_suit] += 1
            if (max(counts) >= hand_size
                    and counts[game.deck.num_trump_suit] < hand_size):
                game = create_game()
                break


def main_loop():
    """Main loop for receiving and executing actions and
    giving rewards.
    """
    global game, threads, action_queue, epsilon
    experience_list = []
    while not game.ended():
        active_player_indices = spawn_threads()
        first_attacker_ix = active_player_indices[0]
        while not game.attack_ended():
            if epsilon >= min_epsilon:
                epsilon -= epsilon_step
            try:
                player_ix, action = action_queue.get(timeout=10)
            except queue.Empty:
                clear_threads()
                clear_queue()
                return False
            game.feature_lock.acquire()
            if only_ais:
                state = game.features[player_ix].copy()
            else:
                state = game.features.copy()
            game.feature_lock.release()
            if game.players[player_ix].checks or action[0] == 4:
                action_queue.task_done()
                if action[0] == 4:
                    if only_ais or player_ix == game.kraudia_ix:
                        reward(active_player_indices, player_ix, wait_reward)
                    threads[active_player_indices.index(player_ix)].event.set()
                continue
            if verbose:
                print(action_to_string(player_ix, action))
            if action[0] == 0:
                if game.field.is_empty():
                    game.attack(player_ix, [make_card(action)])
                    if verbose:
                        print(game.field, '\n')
                    action_queue.task_done()
                    if game.is_winner(player_ix):
                        reward_winner(active_player_indices, player_ix)
                        clear_threads()
                        clear_queue()
                        if player_ix < first_attacker_ix:
                            first_attacker_ix -= 1
                        elif first_attacker_ix == game.player_count - 1:
                            first_attacker_ix = 0
                        update_experience_list_indices(experience_list,
                                player_ix)
                        if game.remove_player(player_ix):
                            break
                        active_player_indices = spawn_threads()
                    elif (not game.attack_ended()
                            and (only_ais or player_ix == game.kraudia_ix)):
                        reward(active_player_indices, player_ix, 0)
                    for thread in threads:
                        thread.event.set()
                    continue
                else:
                    game.attack(player_ix, [make_card(action)])
                    if len(threads) != len(active_player_indices):
                        if game.players[game.defender_ix].checks:
                            clear_threads()
                            game.check(game.defender_ix)
                            spawn_threads()
                        else:
                            clear_threads()
                            spawn_thread()
            elif action[0] == 1:
                to_defend, card = make_card(action)
                game.defend(to_defend, card)
                if len(threads) != len(active_player_indices):
                    clear_threads()
                    spawn_threads()
            elif action[0] == 2:
                game.push([make_card(action)])
                action_queue.task_done()
                if game.is_winner(player_ix):
                    reward_winner(active_player_indices, player_ix)
                    clear_threads()
                    clear_queue()
                    if player_ix < first_attacker_ix:
                        first_attacker_ix -= 1
                    elif first_attacker_ix == game.player_count - 1:
                        first_attacker_ix = 0
                    update_experience_list_indices(experience_list, player_ix)
                    if game.remove_player(player_ix):
                        break
                else:
                    if only_ais or player_ix == game.kraudia_ix:
                        reward(active_player_indices, player_ix, 0)
                    clear_threads()
                    clear_queue()
                active_player_indices = spawn_threads()
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
                reward_winner(active_player_indices, player_ix)
                clear_threads()
                clear_queue()
                if player_ix < first_attacker_ix:
                    first_attacker_ix -= 1
                elif first_attacker_ix == game.player_count - 1:
                    first_attacker_ix = 0
                update_experience_list_indices(experience_list, player_ix)
                if game.remove_player(player_ix):
                    break
                active_player_indices = spawn_threads()
                for thread in threads:
                    thread.event.set()
            else:
                if only_ais or player_ix == game.kraudia_ix:
                    if not (game.attack_ended()
                            or game.players[player_ix].checks):
                        reward(active_player_indices, player_ix, 0)
                    else:
                        experience_list.append([(state, action, 1, None),
                                player_ix])
                threads[active_player_indices.index(player_ix)].event.set()
        # attack ended
        clear_threads()
        clear_queue()
        experience_list = end_turn(first_attacker_ix, experience_list)
        assert not experience_list, 'An experience has not been completed'
        train_from_memory()
    return True


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
    global game, threads
    for thread in threads:
        game.check(thread.player_ix)
        thread.event.set()
        thread.join()
        game.uncheck(thread.player_ix)
    del threads[:]

def clear_queue():
    """Clear the action queue."""
    global action_queue
    while not action_queue.empty():
        try:
            action_queue.get(timeout=1)
        except queue.Empty:
            continue
        action_queue.task_done()


def update_experience_list_indices(experience_list, player_ix):
    """Update all indices in the experience list's tuples to reflect
    that the player with the given index has been removed.
    """
    for i, item in enumerate(experience_list):
        if player_ix < item[1]:
            experience_list[i][1] -= 1


def complete_experience(experience_list, player_ix, reward):
    """Update the reward for the experience for the given player's
    index, insert the game state into it and store it.
    """
    pop_ix = -1
    for i, item in enumerate(experience_list):
        if item[1] == player_ix:
            exp = item[0]
            if only_ais:
                store_experience((exp[0], exp[1], reward,
                        game.features[player_ix]))
            else:
                store_experience((exp[0], exp[1], reward,
                        game.features))
            pop_ix = i
            break
    assert pop_ix >= 0, 'Player index not found in experience list'
    experience_list.pop(pop_ix)
    return experience_list


def reward(active_player_indices, player_ix, reward):
    """Reward the player with the given index by the given reward."""
    global threads
    threads[active_player_indices.index(player_ix)].reward = reward


def reward_winner(active_player_indices, player_ix):
    """Reward the player with the given index as a winner.

    Also reward loser if there is one.
    """
    if only_ais or player_ix == game.kraudia_ix:
        reward(active_player_indices, player_ix, win_reward)
    if game.will_end():
        if only_ais:
            reward(active_player_indices, 1 - player_ix, loss_reward)
        elif player_ix != game.kraudia_ix and game.kraudia_ix >= 0:
            reward(active_player_indices, game.kraudia_ix, loss_reward)


def train(state, action, reward, new_state):
    """Train the networks with the given states, action and reward."""
    global actor, critic
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
    global actor, critic
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


def end_turn(first_attacker_ix, experience_list):
    """End a turn by drawing cards for all attackers and the defender.

    Also give rewards and return the finished experience list.
    """
    global game
    if first_attacker_ix == game.defender_ix:
        first_attacker_ix += 1
    if first_attacker_ix == game.player_count:
        first_attacker_ix = 0
    actual_first_attacker_ix = first_attacker_ix
    while first_attacker_ix != game.defender_ix:
        # first attacker till last attacker, then defender
        if game.is_winner(first_attacker_ix):
            if only_ais or first_attacker_ix == game.kraudia_ix:
                experience_list = complete_experience(experience_list,
                        first_attacker_ix, win_reward)
            update_experience_list_indices(first_attacker_ix)
            if game.remove_player(first_attacker_ix):
                if (only_ais or first_attacker_ix != game.kraudia_ix
                        and game.kraudia_ix >= 0):
                    experience_list = complete_experience(experience_list,
                            1 - first_attacker_ix, loss_reward)
                return [item[0] for item in experience_list] # TODO remove
            elif first_attacker_ix == game.player_count:
                first_attacker_ix = 0
        else:
            game.draw(first_attacker_ix)
            if only_ais or first_attacker_ix == game.kraudia_ix:
                experience_list = complete_experience(experience_list,
                        first_attacker_ix, 0)
            first_attacker_ix += 1
        if first_attacker_ix == game.player_count:
            first_attacker_ix = 0
    if (actual_first_attacker_ix != 0
            and first_attacker_ix == game.player_count - 1):
        game.draw(0)
        if only_ais or game.kraudia_ix == 0:
            experience_list = complete_experience(experience_list, 0, 0)
    elif (actual_first_attacker_ix != first_attacker_ix + 1
            and first_attacker_ix != game.player_count - 1):
        game.draw(first_attacker_ix + 1)
        if only_ais or first_attacker_ix + 1 == game.kraudia_ix:
            experience_list = complete_experience(experience_list,
                    first_attacker_ix + 1, 0)
    assert len(experience_list) <= 1, 'Some experiences have not been updated'
    if game.field.attack_cards:
        amount = game.take()
        if only_ais or first_attacker_ix == game.kraudia_ix:
            experience_list = complete_experience(experience_list,
                    first_attacker_ix, -amount)
    else:
        game.clear_field()
        if game.is_winner(first_attacker_ix):
            if only_ais or first_attacker_ix == game.kraudia_ix:
                experience_list = complete_experience(experience_list,
                        first_attacker_ix, win_reward)
            update_experience_list_indices(first_attacker_ix)
            if game.remove_player(first_attacker_ix):
                if (only_ais or first_attacker_ix != game.kraudia_ix
                        and game.kraudia_ix >= 0):
                    experience_list = complete_experience(experience_list,
                            1 - first_attacker_ix, loss_reward)
        else:
            game.draw(first_attacker_ix)
            if only_ais or first_attacker_ix == game.kraudia_ix:
                experience_list = complete_experience(experience_list,
                        first_attacker_ix, 0)
            game.update_defender()
    return [item[0] for item in experience_list] # TODO remove


class ActionReceiver(threading.Thread):
    """Receive all actions for the given player for one round."""

    def __init__(self, player_ix):
        """Construct an action receiver with the given player index."""
        threading.Thread.__init__(self)
        self.player_ix = player_ix
        self.reward = 1
        self.event = threading.Event()

    def run(self):
        """Add all actions for one round."""
        global game
        player = game.players[self.player_ix]
        if only_ais or self.player_ix == game.kraudia_ix:
            # first attacker
            if (self.player_ix == game.prev_neighbour(game.defender_ix)
                    and game.field.is_empty()):
                game.feature_lock.acquire()
                if only_ais:
                    self.state = game.features[self.player_ix].copy()
                else:
                    self.state = game.features.copy()
                game.feature_lock.release()
                self.possible_actions = game.get_actions(self.player_ix)
                self.add_selected_action()
            self.get_extended_actions()
            # attacker
            if game.defender_ix != self.player_ix:
                defender = game.players[game.defender_ix]
                while not player.checks and self.possible_actions:
                    # everything is defended
                    if not game.field.attack_cards or defender.checks:
                        self.add_selected_action()
            # defender
            else:
                while not player.checks and self.possible_actions:
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
                while not player.checks and self.possible_actions:
                    # everything is defended
                    if ((not game.field.attack_cards or defender.checks)
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
        if not self.event.wait(10):
            self.possible_actions = []
        self.event.clear()
        game.feature_lock.acquire()
        if only_ais:
            self.state = game.features[self.player_ix].copy()
        else:
            self.state = game.features.copy()
        game.feature_lock.release()
        self.possible_actions = game.get_actions(self.player_ix) \
                + [game.check_action(), game.wait_action()]

    def get_actions(self):
        """Wait until the game is updated and return a list of possible
        actions.

        If the wait time is exceeded, return an empty list.
        """
        if not self.event.wait(10):
            self.possible_actions = []
        self.event.clear()
        self.possible_actions = game.get_actions(self.player_ix)

    def add_action(self, action):
        """Add an action with the belonging player's index to the
        action queue.
        """
        global action_queue
        action_queue.put((self.player_ix, action))

    def add_selected_action(self):
        """Add an action calculated by the model or a random one for
        exploration to the action queue.

        Also update the possible actions and store the experience.
        """
        if np.random.random() > epsilon:
            action = [int(v) for v in actor.model.predict(
                    self.state.reshape(1, self.state.shape[0]))[0]]
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
            else:
                self.reward = illegal_action_reward
                self.add_action(game.wait_action())
        else:
            action = choice(self.possible_actions)
            self.add_action(action)
        if reward != 1:
            game.feature_lock.acquire()
            if only_ais:
                store_experience((self.state, action, self.reward,
                        game.features[self.player_ix]))
            else:
                store_experience((self.state, action, self.reward,
                        game.features))
            game.feature_lock.release()
        self.get_extended_actions()
        self.reward = 1

    def add_random_action(self):
        """Add a random action to the action queue or check at random.
        
        Also update the possible actions.
        """
        if np.random.random() > chi:
            self.add_action(choice(self.possible_actions))
            self.get_actions()
        else:
            self.add_action(game.check_action())
            if not self.event.wait(10):
                self.possible_actions = []


def store_experience(experience):
    """Store an experience and overwrite old ones if necessary.

    An experience is a tuple consisting of (state, action, reward,
    new state). This function is threadsafe.
    """
    global experiences, experience_ix, experience_lock
    experience_lock.acquire()
    if len(experiences) == max_experience_count:
        experiences[experience_ix] = experience
        experience_ix += 1
        if experience_ix == max_experience_count:
            experience_ix = 0
    else:
        experiences.append(experience)
    experience_lock.release()


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
                action[3] + action[4] * cards_per_suit),
                deck.Card(action[1], action[2],
                action[1] + action[2] * cards_per_suit))
    return deck.Card(action[1], action[2],
            action[1] + action[2] * cards_per_suit)


if __name__ == '__main__':
    if not only_ais and 'Kraudia' not in names:
        names.append('Kraudia')
    assert len(names) == len(set(names)), 'Names must be unique'
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
    durak_ix = -1
    game = None
    psi = None
    chi = None
    threads = None
    if sys.version_info[0] == 2:
        action_queue = Queue.Queue(len(names) * 6)
    elif sys.version_info[0] == 3:
        action_queue = queue.Queue(len(names) * 6)
    epsilon_step = (epsilon - min_epsilon) / float(epsilon_count)
    min_epsilon += epsilon_step
    experiences = []
    experience_lock = threading.Lock()
    experience_ix = 0
    win_stats = np.zeros(episodes, dtype=np.int8)

    sess = tf.Session(config=tf.ConfigProto())
    K.set_session(sess)
    actor = actor_m.Actor(sess, state_shape, action_shape, load,
            alpha_actor, tau_actor, n1_actor, n2_actor)
    critic = critic_m.Critic(sess, state_shape, action_shape, load,
            alpha_critic, tau_critic, n1_critic, n2_critic)

    print('\nStarting to play\n')
    start_time = clock()
    wins, completed_episodes = main()
    duration = clock() - start_time
    average_duration = duration
    win_rate = wins * 100
    plot_model(actor, to_file='actor-' + str(state_shape) + '-features.png',
            show_shapes=True)
    plot_model(critic, to_file='critic-' + str(state_shape) + '-features.png',
            show_shapes=True)
    if completed_episodes > 0:
        average_duration /= float(completed_episodes)
        win_rate /= float(completed_episodes)
    print('Finished {0}/{1} episode{2} after {3:.2f} seconds; '
            'average: {4:.2f} seconds per episode'.format(completed_episodes,
            episodes, plural_s, duration, average_duration))
    print('Kraudia won {0}/{1} games which is a win rate of {2:.2f} %'.format(
            wins, completed_episodes, win_rate))
    print('Saving data...')
    actor.save_weights()
    critic.save_weights()
    file_name = 'win_stats_'
    if completed_episodes != episodes:
        file_name += 'interrupted_during_' + str(completed_episodes + 1) + '_'
    file_int = 0
    while isfile(file_name + str(file_int) + '.npy'):
        file_int += 1
    np.save(file_name + str(file_int) + '.npy', win_stats, allow_pickle=False)
    print('Done')
