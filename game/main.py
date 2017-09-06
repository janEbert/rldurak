import threading
import queue
from random import choice
from time import clock

import numpy as np

import deck
import player as player_m
import field
import game as game_m

names = ['Bob', 'Alice']
deck_size = 52
hand_size = 6
trump_suit = 2

iterations = 1000
# how often random bots wait
# calculated from a normal distribution with the given values
beta_mu = 0.92
beta_sigma = 0.15
# how often bots check
# calculated from a normal distribution with the given values
chi_mu = 0.1
chi_sigma = 0.15


def main():
    """Main function for durak."""
    global durak_ix, game, beta, chi
    for n in range(iterations):
        beta = min(0.98, np.random.normal(beta_mu, beta_sigma))
        chi = max(0, np.random.normal(chi_mu, chi_sigma))
        game = game_m.Game(names, deck_size, hand_size, trump_suit)
        reshuffle(hand_size)
        if durak_ix < 0:
            beginner_ix, beginner_card = game.find_beginner()
            if beginner_card == game.deck.bottom_trump:
                print('Beginner was chosen randomly')
        else:
            game.defender_ix = durak_ix
        main_loop()
        durak_ix = names.index(game.players[0].name)
        if game.kraudia_ix < 0:
            # TODO positive reward
            print('Kraudia did not lose!')
        else:
            # TODO negative reward
            print('Kraudia is the durak')


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
            game = game_m.Game(names, deck_size, hand_size)
            break
    while (max(counts) >= hand_size
            and counts[game.deck.num_trump_suit] < hand_size):
        for player in game.players:
            counts = [0] * 4
            for card in player.cards:
                counts[card.num_suit] += 1
            if (max(counts) >= hand_size
                    and counts[game.deck.num_trump_suit] < hand_size):
                game = game_m.Game(names, deck_size, hand_size)
                break


def main_loop():
    """Main loop for receiving and executing actions and
    giving rewards.
    """
    global game, threads, action_queue
    while not game.ended():
        active_player_indices, cleared = spawn_threads()
        first_attacker_ix = active_player_indices[0]
        while not game.attack_ended():
            # TODO reward if player_ix == game.kraudia_ix
            player_ix, action = action_queue.get()
            if game.players[player_ix].checks:
                action_queue.task_done()
                continue
            print(action_to_string(player_ix, action))
            if action[0] == 0:
                if game.field.is_empty():
                    game.attack(player_ix, [make_card(action)])
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
                print(game.field, '\n')
                continue
            elif action[0] == 3:
                game.check(player_ix)
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


def spawn_threads():
    """Spawn the action receiving threads for each active player and
    return the active players' indices and false.

    False is for a flag showing whether the threads have been cleared.
    """
    global game, threads
    active_player_indices = game.active_player_indices()
    threads = [None] * len(active_player_indices)
    for thread_ix, player_ix in enumerate(active_player_indices):
        print(player_ix, deck.cards_to_string(game.players[player_ix].cards))
        thread = ActionReceiver(player_ix)
        thread.start()
        threads[thread_ix] = thread
    return active_player_indices, False


def clear_threads(active_player_indices):
    """Responsibly clear the list of threads and the action queue."""
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
        except queue.Empty:
            continue
        action_queue.task_done()
    return True


def end_turn(first_attacker_ix):
    """End a turn by drawing cards for all attackers and
    the defender.
    """
    global game
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
        game.take()
    else:
        game.field.clear()
        if game.is_winner(first_attacker_ix):
            game.remove_player(first_attacker_ix)
        else:
            game.draw(first_attacker_ix)
            game.update_defender()


def make_card(action):
    """Create a card from an action.

    Create a tuple of two cards if action is defending.
    """
    if action[0] == 1:
        return (deck.Card(action[3], action[4], numerical=True),
                deck.Card(action[1], action[2], numerical=True))
    return deck.Card(action[1], action[2], numerical=True)


class ActionReceiver(threading.Thread):
    """Receive all actions for the given player for one round."""

    def __init__(self, player_ix):
        """Construct an action receiver with the given player index."""
        threading.Thread.__init__(self)
        self.player_ix = player_ix
        self.event = threading.Event()

    def run(self):
        """Add all actions for one round."""
        global game
        player = game.players[self.player_ix]
        if False and self.player_ix == game.kraudia_ix:
            while not player.checks and possible_actions:
                possible_actions = game.get_actions(self.player_ix)
                if (not game.defender_ix == self.player_ix
                        and not (game.field.attack_cards
                        and game.field.defended_pairs)
                        or game.defender_ix == self.player_ix):
                    # TODO receive action from neural net
                    if action[0] != 4:
                        possible_actions.remove(action)
                else:
                    add_action(self.player_ix, game.wait_action())
        else:
            # first attacker
            if (self.player_ix == game.prev_neighbour(game.defender_ix)
                    and game.field.is_empty()):
                self.possible_actions = game.get_actions(self.player_ix)
                if len(self.possible_actions) == 1:
                    action = self.possible_actions[0]
                else:
                    action = choice(self.possible_actions)
                add_action(self.player_ix, action)
            self.possible_actions = self.get_actions()
            # attacker
            if game.defender_ix != self.player_ix:
                defender = game.players[game.defender_ix]
                while not player.checks and len(self.possible_actions) > 1:
                    # everything is defended
                    if ((not game.field.attack_cards or defender.checks)
                            and np.random.random() > beta):
                        self.add_random_action()
            # defender
            else:
                while not player.checks and len(self.possible_actions) > 1:
                    if np.random.random() > beta:
                        self.add_random_action()
            if not player.checks:
                add_action(self.player_ix, game.check_action())

    def get_actions(self):
        self.event.wait()
        self.event.clear()
        return game.get_actions(self.player_ix)

    def add_random_action(self):
        if np.random.random() > chi:
            add_action(self.player_ix, choice(self.possible_actions))
            self.possible_actions = self.get_actions()
        else:
            add_action(self.player_ix, game.check_action())
            self.event.wait()


def add_action(player_ix, action):
    """Add an action with the belonging player to the action queue."""
    global action_queue
    action_queue.put((player_ix, action))


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


if __name__ == '__main__':
    names.append('Kraudia')
    assert len(names) == len(set(names)), 'Names must be unique'
    durak_ix = -1
    game = None
    beta = None
    chi = None
    threads = []
    action_queue = queue.Queue(len(names) * 6)

    start_time = clock()
    main()
    duration = clock() - start_time
    print('\nFinished after {0:.2f} seconds; average: {1:.4f} seconds '
            'per episode'.format(duration, duration / iterations))