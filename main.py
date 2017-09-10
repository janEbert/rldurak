import threading
import queue
from random import choice
from time import clock

import numpy as np

import agent.learning as learning
import game.deck as deck
import game.player as player_m
import game.field as field
import game.game as game_m

names = ['Alice', 'Bob']
deck_size = 52 # cannot be changed at the moment
hand_size = 6
trump_suit = 2 # hearts
# whether only AIs are in the game or one AI and random bots
only_ais = False

verbose = False # whether to print game progress
episodes = 1
gamma = 0.99
# starting value for how often a random action is taken by AIs
# linearly anneals 0.1 in the first 3000 episodes
epsilon = 1
# how often random bots wait
# calculated from a normal distribution with the given values
psi_mu = 0.95
psi_sigma = 0.1
# how often bots check
# calculated from a normal distribution with the given values
chi_mu = 0.08
chi_sigma = 0.12
max_experience_count = 300


def main():
    """Main function for durak."""
    global durak_ix, game, psi, chi
    for n in range(episodes):
        if not only_ais:
            psi = min(0.98, np.random.normal(psi_mu, psi_sigma))
            chi = max(0, np.random.normal(chi_mu, chi_sigma))
        game = game_m.Game(names, deck_size, hand_size, trump_suit, only_ais)
        reshuffle(hand_size)
        if durak_ix < 0:
            beginner_ix, beginner_card = game.find_beginner()
            if beginner_card == game.deck.bottom_trump:
                if verbose:
                    print('Beginner was chosen randomly\n')
        else:
            game.defender_ix = durak_ix
        if not main_loop():
            print('Program was shut down from outside\n')
            break
        durak_ix = names.index(game.players[0].name)
        if game.kraudia_ix < 0:
            # TODO positive reward
            if verbose:
                print('Kraudia did not lose!\n')
        else:
            # TODO negative reward
            if verbose:
                print('Kraudia is the durak...\n')


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
            game = game_m.Game(names, deck_size, hand_size, trump_suit,
                    only_ais)
            break
    while (max(counts) >= hand_size
            and counts[game.deck.num_trump_suit] < hand_size):
        for player in game.players:
            counts = [0] * 4
            for card in player.cards:
                counts[card.num_suit] += 1
            if (max(counts) >= hand_size
                    and counts[game.deck.num_trump_suit] < hand_size):
                game = game_m.Game(names, deck_size, hand_size, trump_suit,
                        only_ais)
                break


def main_loop():
    """Main loop for receiving and executing actions and
    giving rewards.
    """
    global game, threads, action_queue, epsilon
    player_ix = -1
    state = game.features
    action = ()
    old_state = state.copy()
    while not game.ended():
        active_player_indices, cleared = spawn_threads()
        first_attacker_ix = active_player_indices[0]
        while not game.attack_ended():
            if (only_ais or player_ix == game.kraudia_ix
                    and game.kraudia_ix >= 0) and action:
                experience = (old_state, action, reward, state)
                if epsilon >= 0.1003:
                    epsilon -= 0.0003
                store_experience(experience)
                old_state = state.copy()
            # TODO reward if player_ix == game.kraudia_ix
            # and observe new state (or in thread)
            try:
                player_ix, action = action_queue.get(timeout=2)
            except (queue.Empty, KeyboardInterrupt):
                clear_threads(active_player_indices)
                return False
            if game.players[player_ix].checks:
                action_queue.task_done()
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
                        cleared = clear_threads(active_player_indices)
                        if player_ix < first_attacker_ix:
                            first_attacker_ix -= 1
                        elif first_attacker_ix == game.player_count - 1:
                            first_attacker_ix = 0
                        if game.remove_player(player_ix):
                            break
                        active_player_indices, cleared = spawn_threads()
                    for thread in threads:
                        thread.event.set()
                    continue
                else:
                    game.attack(player_ix, [make_card(action)])
            elif action[0] == 1:
                to_defend, card = make_card(action)
                game.defend(to_defend, card)
            elif action[0] == 2:
                cleared = clear_threads(active_player_indices)
                game.push([make_card(action)])
                action_queue.task_done()
                if game.is_winner(player_ix):
                    if player_ix < first_attacker_ix:
                        first_attacker_ix -= 1
                    elif first_attacker_ix == game.player_count - 1:
                        first_attacker_ix = 0
                    if game.remove_player(player_ix):
                        break
                active_player_indices, cleared = spawn_threads()
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
                cleared = clear_threads(active_player_indices)
                if player_ix < first_attacker_ix:
                    first_attacker_ix -= 1
                elif first_attacker_ix == game.player_count - 1:
                    first_attacker_ix = 0
                if game.remove_player(player_ix):
                    break
                active_player_indices, cleared = spawn_threads()
                for thread in threads:
                    thread.event.set()
            else:
                threads[active_player_indices.index(player_ix)].event.set()
        # attack ended
        if not cleared:
            clear_threads(active_player_indices)
        end_turn(first_attacker_ix)
    return True


def spawn_threads():
    """Spawn the action receiving threads for each active player and
    return the active players' indices and false.

    False is for a flag showing whether the threads have been cleared.
    """
    global game, threads
    active_player_indices = game.active_player_indices()
    threads = [None] * len(active_player_indices)
    for thread_ix, player_ix in enumerate(active_player_indices):
        if verbose:
            print(player_ix, deck.cards_to_string(game.players[player_ix].cards))
        thread = ActionReceiver(player_ix)
        thread.start()
        threads[thread_ix] = thread
    return active_player_indices, False


def clear_threads(active_player_indices):
    """Responsibly clear the list of threads and the action queue.

    Return true for a flag showing whether the threads have
    been cleared.
    """
    global game, threads, action_queue
    for thread_ix, player_ix in enumerate(active_player_indices):
        game.check(player_ix)
        threads[thread_ix].event.set()
        threads[thread_ix].join()
        game.uncheck(player_ix)
    threads.clear()
    while not action_queue.empty():
        try:
            action_queue.get(timeout=1)
        except (queue.Empty, KeyboardInterrupt):
            continue
        action_queue.task_done()
    return True


def store_experience(experience):
    """Store an experience and overwrite old ones if necessary."""
    global experiences, experience_ix
    if len(experiences) == max_experience_count:
        experiences[experience_ix] = experience
        experience_ix += 1
        if experience_ix == max_experience_count:
            experience_ix = 0
    else:
        experiences.append(experience)


def end_turn(first_attacker_ix):
    """End a turn by drawing cards for all attackers and
    the defender.
    """
    global game
    if only_ais:
        rewards = np.zeros(game.player_count)
    elif game.kraudia_ix >= 0:
        rewards = 0
    if first_attacker_ix == game.defender_ix:
        first_attacker_ix += 1
    if first_attacker_ix == game.player_count:
        first_attacker_ix = 0
    while first_attacker_ix != game.defender_ix:
        # first attacker till last attacker, then defender
        if game.is_winner(first_attacker_ix):
            if first_attacker_ix == game.player_count - 1:
                first_attacker_ix = 0
            if game.remove_player(first_attacker_ix):
                return
        else:
            game.draw(first_attacker_ix)
            first_attacker_ix += 1
        if first_attacker_ix == game.player_count:
            first_attacker_ix = 0
    if first_attacker_ix == game.player_count - 1:
        game.draw(0)
    else:
        game.draw(first_attacker_ix + 1)
    if game.field.attack_cards:
        if (only_ais or game.defender_ix == game.kraudia_ix
                and game.kraudia_ix >= 0):
            rewards[game.defender_ix] -= game.take()
    else:
        game.field.clear()
        if game.is_winner(first_attacker_ix):
            game.remove_player(first_attacker_ix)
        else:
            game.draw(first_attacker_ix)
            game.update_defender()


class ActionReceiver(threading.Thread):
    """Receive all actions for the given player for one round."""

    def __init__(self, player_ix):
        """Construct an action receiver with the given player index."""
        threading.Thread.__init__(self)
        self.player_ix = player_ix
        self.reward = 0
        self.event = threading.Event()

    def run(self):
        """Add all actions for one round."""
        global game
        player = game.players[self.player_ix]
        if False and (only_ais or self.player_ix == game.kraudia_ix):
            # first attacker
            if (self.player_ix == game.prev_neighbour(game.defender_ix)
                    and game.field.is_empty()):
                self.possible_actions = game.get_actions(self.player_ix)
                self.add_selected_action()
            else:
                self.possible_actions = self.get_extended_actions()
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
            self.possible_actions = self.get_actions()
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
        if not self.event.wait(2.0):
            return []
        self.event.clear()
        return game.get_actions(self.player_ix) + [game.check_action(),
                game.wait_action()]

    def get_actions(self):
        """Wait until the game is updated and return a list of possible
        actions.

        If the wait time is exceeded, return an empty list.
        """
        if not self.event.wait(2.0):
            return []
        self.event.clear()
        return game.get_actions(self.player_ix)

    def add_action(self, action):
        """Add an action with the belonging player's index to the
        action queue.
        """
        global action_queue
        action_queue.put((self.player_ix, action))

    def add_selected_action(self):
        if np.random.random() > epsilon:
            # TODO receive action from neural net
            pass
            if action not in self.possible_actions:
                # TODO high negative reward
                pass
        else:
            action = choice(self.possible_actions)
            self.add_action(action)
        # TODO store experience
        self.possible_actions = self.get_extended_actions()
        # TODO maybe observe new state here?

    def add_random_action(self):
        """Add a random action to the action queue or check at random.
        
        Also update the possible actions.
        """
        if np.random.random() > chi:
            self.add_action(choice(self.possible_actions))
            self.possible_actions = self.get_actions()
        else:
            self.add_action(game.check_action())
            if not self.event.wait(2.0):
                self.possible_actions = []


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
    else:
        return str(player_ix) + ': Chk'


def make_card(action):
    """Create a card from an action.

    Create a tuple of two cards if action is defending.
    """
    if action[0] == 1:
        return (deck.Card(action[3], action[4], numerical=True),
                deck.Card(action[1], action[2], numerical=True))
    return deck.Card(action[1], action[2], numerical=True)


if __name__ == '__main__':
    if not only_ais and 'Kraudia' not in names:
        names.append('Kraudia')
    assert len(names) == len(set(names)), 'Names must be unique'
    if episodes == 1:
        plural_s = ''
    else:
        plural_s = 's'
    durak_ix = -1
    game = None
    psi = None
    chi = None
    threads = []
    action_queue = queue.Queue(len(names) * 6)
    experiences = []
    experience_ix = 0

    start_time = clock()
    model = learning.create_model()
    duration = clock() - start_time

    print('Starting to play\n')
    start_time = clock()
    main()
    duration = clock() - start_time
    print('Finished {0} episode{1} after {2:.2f} seconds; average: {3:.4f} '
            'seconds per episode'.format(episodes, plural_s, duration,
            duration / episodes))