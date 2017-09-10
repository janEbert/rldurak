from itertools import chain

import numpy as np

import game.deck as deck
import game.player as player_m
import game.field as field


class Game:
    """A skeleton offering the most relevant functions for durak.

    Also keeps track of the features.
    """

    def __init__(
            self, names, deck_size=52, hand_size=6,
            trump_suit=None, only_ais=False, full_features=True):
        """Initialize a game of durak with the given names, a card
        deck of the given size and hands of the given size.
        """
        if deck_size != 52:
            print('Only default deck size of 52 is supported at the moment!')
        self.deck = deck.Deck(deck_size, trump_suit)
        self.hand_size = hand_size
        self.only_ais = only_ais
        self.full_features = full_features
        self.players = []
        self.kraudia_ix = -1
        self.player_count = len(names)
        assert self.player_count > 1 and self.player_count < 8, \
                'Player count does not make sense'
        # feature initialization
        if full_features:
            # 52 features for each card's location
            #   -4 is bottom trump, -3 is unknown, -2 is out of game
            #   and -1 is on field
            #   >= 0 are indices from Kraudia's position
            # 1 feature for which suit next player could not defend
            #   num_suit of card
            # 1 feature for whether the defending player checks
            #   binary
            # 1 feature for deck size
            #   size of deck
            if self.only_ais:
                self.indices_from = []
                self.features = np.full((self.player_count, 55), -3)
                self.features[:, self.deck.bottom_trump.index] = -4
                self.features[:, 52] = -1
                self.features[:, 53] = 0
                self.features[:, 54] = self.deck.size
            else:
                self.features = np.full(55, -3)
                self.features[self.deck.bottom_trump.index] = -4
                self.features[52] = -1
                self.features[53] = 0
                self.features[54] = self.deck.size
        else:
            # 13 features for number of cards in both neighbour's hands
            # 13 features for each trump card's location
            # 1 feature for which suit next player could not defend
            # 1 feature for deck size
            if self.only_ais:
                print('Game is not configured for non-full features and only '
                        'ais yet!')
            print('Non-full features not as well supported as full features!')
            self.features = np.full(29, -1)
            self.features[13:26] = -2
            self.features[26] = -1
            self.features[27] = 0
            self.features[28] = self.deck.size
        for ix, name in enumerate(names):
            new_player = player_m.Player(name, self.deck.take(self.hand_size))
            # update features
            if self.only_ais:
                for card in new_player.cards:
                    self.features[ix, card.index] = 0
                self.indices_from.append(self.calculate_indices_from(ix))
            elif name == 'Kraudia' and self.kraudia_ix < 0:
                self.kraudia_ix = ix
                for card in new_player.cards:
                    if full_features:
                        self.features[card.index] = 0
                    elif card.suit == self.deck.trump_suit:
                        self.features[13 + card.num_value] = 0
                self.indices_from_kraudia = self.calculate_indices_from()
            self.players.append(new_player)
        self.field = field.Field()
        self.defender_ix = -1

    def find_beginner(self):
        """Return the index of the player with the lowest trump and
        the card or a random index and the bottom trump if no player
        has a trump in hand.

        Also update the defender index.
        """
        mini = 13
        beginner_ix = np.random.randint(self.player_count)
        beginner_card = self.deck.bottom_trump
        for ix, player in enumerate(self.players):
            if (mini == 2 or self.deck.bottom_trump.num_value == 2
                    and mini == 3):
                break
            for card in player.cards:
                if card.suit == self.deck.trump_suit and card.num_value < mini:
                    mini = card.num_value
                    beginner_ix = ix
                    beginner_card = card
                    if (mini == 2 or self.deck.bottom_trump.num_value == 2
                            and mini == 3):
                        break
        self.defender_ix = (beginner_ix + 1) % self.player_count
        return beginner_ix, beginner_card

    def attack(self, attacker_ix, cards):
        """Attack the defender with the attacker's given cards."""
        assert attacker_ix < self.player_count, 'Attacker does not exist'
        attacker = self.players[attacker_ix]
        assert not self.exceeds_field(cards), ('Number of attack cards '
                'exceeds allowed number')
        assert len(cards) <= len(self.players[self.defender_ix].cards), \
                'Defender does not have that many cards'
        for card in cards:
            assert card in attacker.cards, ('Attacker does not have one of '
                    'the cards')
        if self.field.is_empty():
            assert deck.same_value(cards), ('Cards must have the same value '
                    'for initial attack.')
        else:
            assert self.field.values_on_field(cards), ("One of the cards' "
                    "values is not on the field")
        attacker.attack(cards)
        self.field.attack(cards)
        # update features
        if self.only_ais:
            for card in cards:
                self.features[:, card.index] = -1
        elif self.kraudia_ix >= 0:
            if self.full_features:
                for card in cards:
                    self.features[card.index] = -1
            else:
                for card in cards:
                    if card.suit == self.deck.trump_suit:
                        self.features[13 + card.num_value] = -1
                if (attacker_ix == self.next_neighbour()
                        or attacker_ix == self.prev_neighbour()):
                    for card in cards:
                        if card.suit != self.deck.trump_suit:
                            self.sub_neighbour_card(card)

    def defend(self, to_defend, card):
        """Defend the card to defend with the given card."""
        defender = self.players[self.defender_ix]
        assert card in defender.cards, 'Defender does not have the card'
        assert self.field.on_field_attack(to_defend), ('Card to defend is not '
                'on the field as an attack')
        is_greater = card.value > to_defend.value
        assert (is_greater and card.suit == to_defend.suit
                or card.suit == self.deck.trump_suit), 'Card is too low'
        if to_defend.suit == self.deck.trump_suit:
            assert is_greater, 'Card is too low'
        defender.defend(to_defend, card)
        self.field.defend(to_defend, card)
        # update features
        if self.only_ais:
            self.features[:, card.index] = -1
        elif self.kraudia_ix >= 0:
            if self.full_features:
                self.features[card.index] = -1
            else:
                # TODO optimizable
                if card.suit == self.deck.trump_suit:
                    self.features[13 + card.num_value] = -1
                if (self.defender_ix == self.next_neighbour()
                        or self.defender_ix == self.prev_neighbour()):
                    if card.suit != self.deck.trump_suit:
                        self.sub_neighbour_card(card)

    def push(self, cards):
        """Push the cards to the next player."""
        assert not self.field.defended_pairs, ('Cannot push after '
                'having defended')
        assert not self.exceeds_field(cards,
                self.next_neighbour(self.defender_ix)), ('Number of attack '
                'cards exceeds allowed number') 
        self.players[self.defender_ix].push(cards)
        self.field.push(cards)
        # update features
        if self.only_ais:
            for card in cards:
                self.features[:, card.index] = -1
        elif self.kraudia_ix >= 0:
            if self.full_features:
                for card in cards:
                    self.features[card.index] = -1
            else:
                # TODO optimizable
                for card in cards:
                    if card.suit == self.deck.trump_suit:
                        self.features[13 + card.num_value] = -1
                        break
                if (self.defender_ix == self.next_neighbour()
                        or self.defender_ix == self.prev_neighbour()):
                    for card in cards:
                        if card.suit != self.deck.trump_suit:
                            self.sub_neighbour_card(card)
        self.update_defender()

    def take(self):
        """Make the defender take all the cards on the field and
        return the amount of cards taken."""
        assert not self.field.is_empty(), 'Field cannot be empty'
        # update undefended suit feature
        # TODO still rudimentary
        if self.only_ais:
            self.features[self.prev_neighbour(self.defender_ix), 52] = \
                    self.field.attack_cards[0].num_suit
            if len(self.field.attack_cards) > 1:
                attack_suits = [card.suit for card in self.field.attack_cards]
                for (attack_card, defense_card) in self.field.defended_pairs:
                    if (attack_card.suit != self.deck.trump_suit
                            and defense_card.suit == self.deck.trump_suit
                            and attack_card.suit in attack_suits):
                        self.features[self.prev_neighbour(self.defender_ix),
                                52] = attack_card.num_suit
                        break            
        elif self.defender_ix == self.next_neighbour() and self.kraudia_ix >= 0:
            if self.full_features:
                self.features[52] = self.field.attack_cards[0].num_suit
            else:
                self.features[26] = self.field.attack_cards[0].num_suit
            if len(self.field.attack_cards) > 1:
                attack_suits = [card.suit for card in self.field.attack_cards]
                for (attack_card, defense_card) in self.field.defended_pairs:
                    if (attack_card.suit != self.deck.trump_suit
                            and defense_card.suit == self.deck.trump_suit
                            and attack_card.suit in attack_suits):
                        if self.full_features:
                            self.features[52] = attack_card.num_suit
                        else:
                            self.features[26] = attack_card.num_suit
                        break
        cards = self.field.take()
        self.players[self.defender_ix].take(cards)
        # update features
        if self.only_ais:
            for ix in range(self.player_count):
                for card in cards:
                    self.features[ix, card.index] = self.indices_from[ix,
                            self.defender_ix]
        elif self.kraudia_ix >= 0:
            if self.full_features:
                for card in cards:
                    self.features[card.index] = self.indices_from_kraudia[
                            self.defender_ix]
            else:
                # TODO optimizable
                for card in cards:
                    if card.suit == self.deck.trump_suit:
                        self.features[13 + card.num_value] = \
                                self.indices_from_kraudia[self.defender_ix]
                if (self.defender_ix == self.prev_neighbour()
                        or self.defender_ix == self.next_neighbour()):
                    for card in cards:
                        if card.suit != self.deck.trump_suit:
                            self.features[card.num_value] += 1
        self.update_defender(2)
        return len(cards)

    def check(self, player_ix):
        """Tell the others that the player does not want to attack or
        defend anymore.
        """
        assert player_ix < self.player_count, 'Player does not exist'
        self.players[player_ix].check()
        if player_ix == self.defender_ix:
            # update features
            if self.only_ais:
                self.features[:, 53] = 1
            elif self.kraudia_ix >= 0:
                if self.full_features:
                    self.features[53] = 1
                else:
                    self.features[27] = 1

    def uncheck(self, player_ix):
        """Reset the flag for checking for the given player.
        
        Should only be executed after an attack has ended, not
        during one.
        """
        assert player_ix < self.player_count, 'Player does not exist'
        self.players[player_ix].uncheck()
        if player_ix == self.defender_ix:
            # update features
            if self.only_ais:
                self.features[:, 53] = 0
            elif self.kraudia_ix >= 0:
                if self.full_features:
                    self.features[53] = 0
                else:
                    self.features[27] = 0

    def draw(self, player_ix):
        """Draw cards for the given player until their hand is filled
        or the deck is empty.
        """
        assert player_ix < self.player_count, 'Player does not exist'
        player = self.players[player_ix]
        amount = self.hand_size - len(player.cards)
        if amount > 0:
            cards = self.deck.take(amount)
            player.take(cards)
            # update features
            if self.only_ais:
                self.features[:, 54] = self.deck.size
                for card in cards:
                    self.features[player_ix, card.index] = 0
                if self.deck.size == 0:
                    for ix in range(self.player_count):
                        self.features[:, self.deck.bottom_trump.index] = \
                                self.indices_from[ix, player_ix]
            elif self.kraudia_ix >= 0:
                if self.full_features:
                    self.features[54] = self.deck.size
                else:
                    self.features[28] = self.deck.size
                if player_ix == self.kraudia_ix:
                    if self.full_features:
                        for card in cards:
                            self.features[card.index] = 0
                    else:
                        for card in cards:
                            if card.suit == self.deck.trump_suit:
                                self.features[13 + card.num_value] = 0
                elif self.deck.size == 0:
                    if self.full_features:
                        self.features[self.deck.bottom_trump.index] = \
                                self.indices_from_kraudia[player_ix]
                    else:
                        self.features[13 + self.deck.bottom_trump.num_value] \
                                = self.indices_from_kraudia[player_ix]


    def attack_ended(self):
        """Test whether an attack is over because the maximum allowed
        number of attack cards has been placed and defended or because
        all attackers and the defender have checked.
        """
        defender = self.players[self.defender_ix]
        return (not defender.cards
                or len(self.field.attack_cards)
                + len(self.field.defended_pairs) == self.hand_size
                or self.players[self.prev_neighbour(self.defender_ix)].checks
                and self.players[self.next_neighbour(self.defender_ix)].checks
                and defender.checks)

    def exceeds_field(self, cards, player_ix=None):
        """Return whether the number of cards on the field would
        exceed the maximum allowed number of attack cards if the given
        cards were played when the player with the given
        index defends.

        If no index is given, return whether the number of cards
        would exceed the maximum allowed number of cards in general.
        """
        count = (len(self.field.attack_cards)
                + len(self.field.defended_pairs) + len(cards))
        if player_ix is None:
            return count > self.hand_size
        else:
            return count > min(len(self.players[player_ix].cards),
                    self.hand_size)

    def get_actions(self, player_ix=None):
        """Return a list of possible actions for the current game
        state for the given player index.

        If no index is given, kraudia_ix is used. The actions for
        checking and waiting are left out.

        An action is a tuple consisting of:
        - action types attack (0), defend (1), push (2), check (3)
          and wait (4)
        - numerical value of the card to play (-1 if redundant)
        - numerical suit of the card to play (-1 if redundant)
        - if defending, numerical value of the card to defend (else -1)
        - if defending, numerical suit of the card to defend
          (else -1)
        """
        if player_ix is None:
            player_ix = self.kraudia_ix
        assert player_ix < self.player_count, 'Player does not exist'
        actions = []
        player = self.players[player_ix]
        pushed = 0
        if player_ix == self.defender_ix:
            # actions as defender
            for to_defend in self.field.attack_cards:
                for card in player.cards:
                    is_greater = card.value > to_defend.value
                    if (is_greater and card.suit == to_defend.suit
                            or card.suit == self.deck.trump_suit):
                        if to_defend.suit == self.deck.trump_suit:
                            if is_greater:
                                actions.append(
                                        self.defend_action(card, to_defend))
                        else:
                            actions.append(self.defend_action(card, to_defend))
                    if (pushed < 2 and card.value == to_defend.value
                            and not self.field.defended_pairs
                            and not self.exceeds_field([None],
                                    self.next_neighbour(player_ix))): 
                        actions.append(self.push_action(card))
                        if pushed == 0:
                            pushed = 1
                if pushed == 1:
                    pushed = 2
        else:
            is_first_attacker = player_ix == self.prev_neighbour(
                    self.defender_ix)
            if (is_first_attacker
                    or player_ix == self.next_neighbour(self.defender_ix)):
                # actions as first attacker
                if self.field.is_empty() and is_first_attacker:
                    for card in player.cards:
                        actions.append(self.attack_action(card))
                    return actions
                # actions as attacker
                for field_card in (self.field.attack_cards
                        + list(chain.from_iterable(
                        self.field.defended_pairs))):
                    for card in player.cards:
                        if card.value == field_card.value:
                            actions.append(self.attack_action(card))
            else:
                return []
        return actions

    def attack_action(self, card):
        """Return an action tuple for attacking with the given card."""
        return (0, card.num_value, card.num_suit, -1, -1)

    def defend_action(self, card, to_defend):
        """Return an action tuple for defending the card to defend
        with the given card.
        """
        return (1, card.num_value, card.num_suit, to_defend.num_value,
                to_defend.num_suit)

    def push_action(self, card):
        """Return an action tuple for pushing with the given card."""
        return (2, card.num_value, card.num_suit, -1, -1)

    def check_action(self):
        """Return an action tuple for checking."""
        return (3, -1, -1, -1, -1)

    def wait_action(self):
        """Return an action tuple for waiting."""
        return (4, -1, -1, -1, -1)

    def active_player_indices(self):
        """Return the indices of both attackers and the defender.

        Return the indices of the attacker and defender if only two
        players are left.
        """
        if len(self.players) > 2:
            return [self.prev_neighbour(self.defender_ix), self.defender_ix,
                    self.next_neighbour(self.defender_ix)]
        else:
            return [self.prev_neighbour(self.defender_ix), self.defender_ix]

    def prev_neighbour(self, player_ix=None):
        """Return the index of the player coming before the player of
        the given index.

        If no index is given, Kraudia's index is used.
        """
        if player_ix is None:
            if self.kraudia_ix == 0:
                return self.indices_from_kraudia[self.player_count - 1]
            else:
                return self.indices_from_kraudia[self.kraudia_ix - 1]
        assert player_ix < self.player_count, 'Player does not exist'
        if player_ix == 0:
            return self.player_count - 1
        else:
            return player_ix - 1

    def next_neighbour(self, player_ix=None):
        """Return the index of the player coming after the player of
        the given index.

        If no index is given, Kraudia's index is used.
        """
        if player_ix is None:
            if self.kraudia_ix == self.player_count - 1:
                return self.indices_from_kraudia[0]
            else:
                return self.indices_from_kraudia[self.kraudia_ix + 1]
        assert player_ix < self.player_count, 'Player does not exist'
        if player_ix == self.player_count - 1:
            return 0
        else:
            return player_ix + 1

    def index_from(self, player_ix, from_ix=None):
        """Return how far the player at the given index' position is
        from the other index in clockwise direction.

        For example, if from_ix is 2 and player_ix is 5, then return 3
        (with appropriate wrapping). If from_ix is not given,
        use Kraudia's index.
        """
        assert player_ix < self.player_count, 'Player does not exist'
        if from_ix is None:
            from_ix = self.kraudia_ix 
        return (player_ix - from_ix) % self.player_count

    def calculate_indices_from(self, player_ix=None):
        """Calculate a list of indices expressing how far away those
        players are from the given player index.

        If no index is given, use Kraudia's.
        """
        if player_ix is None:
            player_ix = self.kraudia_ix
        return [self.index_from(x, player_ix)
                for x in range(self.player_count)]

    def update_defender(self, count=1):
        """Increase defender index by count (with wrapping)."""
        for i in range(count):
            self.defender_ix += 1
            if self.defender_ix == self.player_count:
                self.defender_ix = 0

    def sub_neighbour_card(self, card):
        """Subtract one from the feature belonging to the given card's
        value in the neighbours' hands.

        Clips to 0.
        """
        self.features[card.num_value] = max(0,
                self.features[card.num_value] - 1)

    def is_winner(self, player_ix):
        """Return whether a player has no cards left and the deck
        is empty.
        """
        return not self.players[player_ix].cards and self.deck.size == 0

    def remove_player(self, player_ix):
        """Remove the player from the game and return true if only one
        player is left.
        """
        player = self.players[player_ix]
        self.players.remove(player)
        self.player_count -= 1
        if self.defender_ix == self.player_count:
            self.defender_ix = 0
        if player_ix < self.kraudia_ix:
            self.kraudia_ix -= 1
        elif player_ix == self.kraudia_ix:
            self.kraudia_ix = -1
            return self.player_count == 1
        if self.only_ais:
            self.features = np.delete(self.features, player_ix, 0)
            self.indices_from = []
            for ix in range(self.player_count):
                self.indices_from.append(self.calculate_indices_from(ix))
        elif self.kraudia_ix >= 0:
            self.indices_from_kraudia = self.calculate_indices_from()
        return self.player_count == 1

    def ended(self):
        """Return whether no player aside from one is left."""
        return len(self.players) <= 1