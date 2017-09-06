import deck, player as player_m, field, game as game_m
import threading, queue
from random import choice
import numpy as np

# name 'Kraudia' is required to find the agent
names = ['Kraudia', 'Bob', 'Alice']
deck_size = 52
hand_size = 6

iterations = 10
# how often random bots wait
# calculated from a normal distribution with the given values
beta_mu = 0.92
beta_sigma = 0.15


def main():
    """Main function for Durak."""

    global durak_ix, game, beta
    for n in range(iterations):
        beta = min(0.98, np.random.normal(beta_mu, beta_sigma))
        game = game_m.Game(names, deck_size, hand_size)
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
            print('Kraudia is the Durak')


def main_loop():
    """Main loop for receiving and executing actions
    and giving rewards."""

    global game, threads, action_queue
    while not game.ended():
        active_player_indices = spawn_threads()
        print('')
        first_attacker_ix = active_player_indices[0]
        while not game.attack_ended():
            # TODO reward if player_ix == game.kraudia_ix
            print('waiting for action')
            print('active:', active_player_indices)
            player_ix, action = action_queue.get()
            if game.players[player_ix].checks:
                action_queue.task_done()
                continue
            print(action_to_string(player_ix, action))
            if action[0] == 0:
                if game.field.is_empty():
                    game.attack(player_ix, [make_card(action)])
                    print(game.field)
                    print('')
                    action_queue.task_done()
                    if game.is_winner(player_ix):
                        clear_threads(active_player_indices)
                        if game.remove_player(player_ix):
                            break
                        active_player_indices = spawn_threads()
                    for thread in threads:
                        thread.event.set()
                    continue
                else:
                    game.attack(player_ix, [make_card(action)])
            elif action[0] == 1:
                to_defend, card = make_card(action)
                game.defend(to_defend, card)
            elif action[0] == 2:
                clear_threads(active_player_indices)
                game.push([make_card(action)])
                action_queue.task_done()
                if game.is_winner(player_ix):
                    if game.remove_player(player_ix):
                        break
                active_player_indices = spawn_threads()
                for thread in threads:
                    thread.event.set()
                print(game.field)
                print('')
                continue
            elif action[0] == 3:
                game.check(player_ix)
            print(game.field)
            print('')
            action_queue.task_done()
            if game.is_winner(player_ix):
                clear_threads(active_player_indices)
                if game.remove_player(player_ix):
                    break
                active_player_indices = spawn_threads()
                for thread in threads:
                    thread.event.set()
            else:
                threads[player_ix].event.set()
        # attack ended
        print('attack ended\n')
        clear_threads(active_player_indices)
        end_turn(first_attacker_ix)


class ActionReceiver(threading.Thread):
    """Receives all actions for the given player for one round."""

    def __init__(self, player_ix):
        threading.Thread.__init__(self)
        self.player_ix = player_ix
        self.event = threading.Event()

    def run(self):
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
                possible_actions = game.get_actions(self.player_ix)
                if len(possible_actions) == 1:
                    action = possible_actions[0]
                else:
                    action = choice(possible_actions[:-1])
                add_action(self.player_ix, action)
            print('reached past first attacker')
            self.event.wait()
            self.event.clear()
            possible_actions = game.get_actions(self.player_ix)
            # attacker
            if game.defender_ix != self.player_ix:
                defender = game.players[game.defender_ix]
                while not player.checks and len(possible_actions) > 1:
                    # everything is defended
                    if ((not game.field.attack_cards
                            or defender.checks)
                            and np.random.random() > beta):
                        add_action(self.player_ix, choice(possible_actions))
                        self.event.wait()
                        self.event.clear()
                        possible_actions = game.get_actions(self.player_ix)
            # defender
            else:
                while not player.checks and len(possible_actions) > 1:
                    # defender
                    if np.random.random() > beta:
                        add_action(self.player_ix, choice(possible_actions))
                        self.event.wait()
                        self.event.clear()
                        possible_actions = game.get_actions(self.player_ix)
            if not player.checks:
                add_action(self.player_ix, game.check_action())


def add_action(player_ix, action):
    """Add an action with the belonging player to the action queue."""

    global action_queue
    print('*** Queue: ' + action_to_string(player_ix, action) + ' ***')
    action_queue.put((player_ix, action))


def make_card(action):
    """Create a card from an action.
    Creates a tuple of two cards if action is defending."""

    if action[0] == 1:
        return (deck.Card(action[3], action[4], numerical=True),
                deck.Card(action[1], action[2], numerical=True))
    return deck.Card(action[1], action[2], numerical=True)


def reshuffle(hand_size):
    """Reshuffle if a player has more than the given hand size minus
    one cards of the same suit (except trump) in their hand."""

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


def end_turn(first_attacker_ix):
    """Ends a turn by drawing cards for all attackers
    and the defender."""

    global game
    while first_attacker_ix != game.defender_ix:
        # first attacker till last attacker, then defender
        if game.is_winner(first_attacker_ix):
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


def spawn_threads():
    """Spawns the action receiving threads for each active player and
    returns the active players' indices."""

    global game, threads
    active_player_indices = game.active_player_indices()
    threads = [None] * len(active_player_indices)
    for ix in active_player_indices:
        print(ix)
        print(deck.cards_to_string(game.players[ix].cards))
        thread = ActionReceiver(ix)
        thread.start()
        threads[ix] = thread
    return active_player_indices


def clear_threads(active_player_indices):
    """Responsibly clears the list of threads and the action queue."""

    global game, threads, action_queue
    for ix in active_player_indices:
        game.check(ix)
        threads[ix].event.set()
        threads[ix].join()
        game.uncheck(ix)
    threads.clear()
    while not action_queue.empty():
        try:
            action_queue.get(timeout=1)
        except queue.Empty:
            continue
        action_queue.task_done()


def action_to_string(player_ix, action):
    """Converts a player's action to a string."""

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
    assert len(names) == len(set(names)), 'Names must be unique'
    durak_ix = -1
    game = None
    beta = None
    threads = []
    action_queue = queue.Queue(len(names) * 6)

    main()