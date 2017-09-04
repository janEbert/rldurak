import deck, player as player_m, field
import numpy as np

from itertools import chain


class Game:

    def __init__(self, names, deck_size=52, hand_size=6, full_features=True):
        """Initializes a game of durak with the given names, a card
        deck of the given size and hands of the given size."""

        self.deck = deck.Deck(deck_size)
        self.hand_size = hand_size
        self.full_features = full_features
        self.players = []
        self.kraudia_ix = -1
        # feature initialization
        if full_features:
            # 52 features for each card's location
            #   -3 is unknown, -2 is out of game, -1 is on field
            #   >= 0 are indices from Kraudia's position
            # 1 feature for which suit next player could not defend
            #   num_suit of card
            # 1 feature for deck size
            #   size of deck
            self.features = np.full(54, -3)
            self.features[52] = -1
            self.features[53] = deck_size
        else:
            # 13 features for number of cards in both neighbour's hands
            # 13 features for each trump card's location
            # 1 feature for which suit next player could not defend
            # 1 feature for deck size
            self.features = np.full(28, -1)
            self.features[13:26] = -2
            self.features[27] = deck_size
        for i, name in enumerate(names):
            new_player = player_m.Player(name, self.deck.take(self.hand_size))
            if name == 'Kraudia' and self.kraudia_ix < 0:
                self.kraudia_ix = i
                for card in new_player.cards:
                    if card.suit == self.deck.trump_suit:
                        if full_features:
                            self.features[card.index] = 0
                        else:
                            self.features[13 + card.num_value] = 0
            self.players.append(new_player)
        self.player_count = len(self.players)
        # TODO make two players possible
        assert self.player_count > 2, 'Two players not currently supported'
        assert self.player_count > 1 and self.player_count < 8, \
                'Player count does not make sense'
        self.field = field.Field()
        self.indices_from_kraudia = [self.index_from_kraudia(x)
                for x in range(self.player_count)]
        self.defender_ix = -1

    def find_beginner(self):
        """Returns the index of the player with the lowest trump and
        the card or a random index and the bottom trump if no player
        has a trump in hand.
        Also updates the defender index."""

        mini = 13
        beginner_ix = np.random.randint(self.player_count)
        beginner_card = self.deck.bottom_trump
        for i, player in enumerate(self.players):
            if (mini == 2 or self.deck.bottom_trump.num_value == 2
                    and mini == 3):
                break
            for card in player.cards:
                if card.suit == self.deck.trump_suit and card.num_value < mini:
                    mini = card.num_value
                    beginner_ix = i
                    beginner_card = card
                    if (mini == 2 or self.deck.bottom_trump.num_value == 2
                            and mini == 3):
                        break
        self.defender_ix = (beginner_ix + 1) % self.player_count
        return beginner_ix, beginner_card

    def attack(self, attacker_ix, cards):
        """Attacks the defender with the attacker's given cards."""

        assert attacker_ix < self.player_count, ('One of the players does '
                'not exist')
        attacker = self.players[attacker_ix]
        defender = self.players[self.defender_ix]
        assert not self.exceeds_field(cards, defender), ('Number of attack '
                'cards exceeds allowed number')
        for card in cards:
            assert card in attacker.cards, ('Attacker does not have one of '
                    'the cards')
        if self.field.is_empty():
            assert deck.same_value(cards), ('Cards must have the same value '
                    'for initial attack.')
        else:
            assert self.field.values_on_field(cards), ('One of the cards\' '
                    'values is not on the field')
        attacker.attack(cards)
        self.field.attack(cards)
        # update features
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
        """Defends the card to defend with the given card."""

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
        """Pushes the cards to the next player."""

        assert self.field.defended_pairs == [], ('Cannot push after '
                'having defended')
        defender = self.players[self.defender_ix]
        assert not exceeds_field(cards,
                self.players[self.next_neighbour(self.defender_ix)]), \
                'Number of attack cards exceeds allowed number' 
        defender.push(cards)
        self.field.push(cards)
        # update features
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
        """Makes the defender take all the cards on the field."""

        assert not self.field.is_empty(), 'Field cannot be empty'
        # update undefended suit feature
        # TODO still rudimentary
        if self.defender_ix == self.next_neighbour():
            if self.full_features:
                self.features[52] = self.field.attack_cards[0].num_suit
            else:
                self.features[26] = self.field.attack_cards[0].num_suit
            if len(self.field.attack_cards) > 1:
                att_suits = [card.suit for card in self.field.attack_cards]
                for (att_card, def_card) in defended_pairs:
                    if (att_card.suit != self.deck.trump_suit
                            and def_card.suit == self.deck.trump_suit
                            and att_card.suit in att_suits):
                        if self.full_features:
                            self.features[52] = att_card.suit
                        else:
                            self.features[26] = att_card.suit
                        break
        cards = self.field.take()
        self.players[self.defender_ix].take(cards)
        # update features
        if self.full_features:
            for card in cards:
                self.features[card.index] = self.indices_from_kraudia[
                        self.defender_ix]
        else:
            # TODO optimizable
            for card in cards:
                if card.suit == self.deck.trump_suit:
                    self.features[13 + card.num_value] = \
                            self.index_from_kraudia(self.defender_ix)
            if (self.defender_ix == self.prev_neighbour()
                    or self.defender_ix == self.next_neighbour()):
                for card in cards:
                    if card.suit != self.deck.trump_suit:
                        self.features[card.num_value] += 1
        self.update_defender(2)

    def check(self, player_ix):
        """Tells the others that the player does not want to attack or
        defend anymore."""

        assert player_ix < self.player_count, 'Player does not exist'
        self.players[player_ix].check()

    def uncheck(self, player_ix):
        """Resets the flag for checking for the given player.
        
        Should only be executed after an attack has ended,
        not during one."""

        assert player_ix < self.player_count, 'Player does not exist'
        self.players[player_ix].uncheck()

    def draw(self, player_ix):
        """Draws cards for the given player until their hand is filled
        or the deck is empty."""

        assert player_ix < self.player_count, 'Player does not exist'
        player = self.players[player_ix]
        amount = self.hand_size - len(player.cards)
        assert amount >= 0, 'Amount is less than one'
        cards = self.deck.take(amount)
        player.take(cards)
        # update features
        if self.full_features:
            self.features[53] = self.deck.size
        else:
            self.features[27] = self.deck.size
        if player_ix == self.kraudia_ix:
            if self.full_features:
                for card in cards:
                    self.features[card.index] = 0
            else:
                for card in cards:
                    if card.suit == self.deck.trump_suit:
                        self.features[13 + card.num_value] = 0

    def attack_ended(self):
        """Tests whether an attack is over because the maximum allowed
        number of attack cards has been placed and defended or because
        all attackers and the defender have checked."""

        defender = self.players[self.defender_ix]
        return (defender.cards == []
            or len(self.field.defended_pairs) == self.hand_size
            or self.players[self.prev_neighbour(self.defender_ix)].checks
            and self.players[self.next_neighbour(self.defender_ix)].checks
            and defender.checks)

    def exceeds_field(self, cards, player):
        """Returns whether the number of cards on the field would
        exceed the maximum allowed number of attack cards when the
        player with the given index defends."""

        return (len(self.field.attack_cards) + len(self.field.defended_pairs)
                + len(cards) > min(len(player.cards), self.hand_size))

    def get_actions(self, player_ix=None):
        """Returns a list of possible actions for the current game
        state for the given player index.
        If no index is given, kraudia_ix is used.

        An action is a vector consisting of:
        - action types attack (0), defend (1), push (2), check (3) and
          wait (4)
        - numerical value of the card to play (-1 if redundant)
        - numerical suit of the card to play (-1 if redundant)
        - if defending, numerical value of the card to defend (else -1)
        - if defending, numerical suit of the card to defend
          (else -1)"""

        if player_ix == None:
            player_ix = self.kraudia_ix
        assert player_ix < self.player_count, 'Player does not exist'
        actions = []
        player = self.players[player_ix]
        attack_cards = self.field.attack_cards
        pushed = 0
        if player_ix == self.defender_ix:
            # actions as defender
            for to_defend in attack_cards:
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
                            and self.field.defended_pairs == []
                            and not exceeds_field([None],
                            self.players[self.next_neighbour(player_ix)])): 
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
                if attack_cards == [] and is_first_attacker:
                    for card in player.cards:
                        actions.append(self.attack_action(card))
                    return actions
                # actions as attacker
                for field_card in (attack_cards
                        + list(chain.from_iterable(
                        self.field.defended_pairs))):
                    for card in player.cards:
                        if card.value == field_card.value:
                            actions.append(self.attack_action(card))
            else:
                return []
        actions += [self.check_action(), self.wait_action()]
        return actions

    def attack_action(self, card):
        """Returns an action vector for attacking with the
        given card."""

        return (0, card.num_value, card.num_suit, -1, -1)

    def defend_action(self, card, to_defend):
        """Returns an action vector for defending the card to
        defend with the given card."""

        return (1, card.num_value, card.num_suit, to_defend.num_value,
                to_defend.num_suit)

    def push_action(self, card):
        """Returns an action vector for pushing with the given card."""

        return (2, card.num_value, card.num_suit, -1, -1)

    def check_action(self):
        """Returns an action vector for checking."""

        return (3, -1, -1, -1, -1)

    def wait_action(self):
        """Returns an action vector for waiting."""

        return (4, -1, -1, -1, -1)

    def active_player_indices(self):
        """Returns the indices of both attackers and the defender."""

        return [self.prev_neighbour(self.defender_ix), self.defender_ix,
                self.next_neighbour(self.defender_ix)]

    def prev_neighbour(self, player_ix=None):
        """Returns the index of the player before the player of the
        given index.
        If no index is given, kraudia_ix is used."""

        if player_ix == None:
            if self.kraudia_ix == 0:
                return self.indices_from_kraudia[self.player_count - 1]
            else:
                return self.indices_from_kraudia[self.kraudia_ix - 1]
        assert player_ix < self.player_count, 'Player does not exist'
        return (player_ix - 1) % self.player_count

    def next_neighbour(self, player_ix=None):
        """Returns the index of the player after the player of the
        given index.
        If no index is given, kraudia_ix is used."""

        if player_ix == None:
            if self.kraudia_ix == self.player_count - 1:
                return self.indices_from_kraudia[0]
            else:
                return self.indices_from_kraudia[self.kraudia_ix + 1]
        assert player_ix < self.player_count, 'Player does not exist'
        return (player_ix + 1) % self.player_count

    def index_from_kraudia(self, player_ix):
        """Returns how far the player sitting at player_ix is from
        Kraudia in clockwise direction.
        For example, if kraudia_ix is 2 and player_ix is 5, then this
        returns 3 (with appropriate wrapping)."""

        assert player_ix < self.player_count, 'Player does not exist'
        return (player_ix - self.kraudia_ix) % self.player_count 

    def update_defender(self, count=1):
        """Increases defender index by count (with wrapping)."""

        for i in range(count):
            self.defender_ix += 1
            if self.defender_ix == self.player_count:
                self.defender_ix = 0

    def sub_neighbour_card(self, card):
        """Subtracts one from the feature belonging to the given
        card's value in the neighbours' hands.
        Clips to 0."""

        self.features[card.num_value] = max(0,
                self.features[card.num_value] - 1)

    def ended(self):
        """Checks if no player aside from one has cards left."""

        count = 0;
        for player in self.players:
            if player.cards != []:
                count += 1
        return count <= 1