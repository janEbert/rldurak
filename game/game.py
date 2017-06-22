import deck, player, field
import numpy as np


class Game:

    def __init__(self, names, deck_size=52, hand_size=6):
        """Initializes a game of durak with the given names, a card
        deck of the given size and hands of the given size."""

        self.deck = deck.Deck(deck_size)
        self.hand_size = hand_size
        self.players = []
        self.kraudia_ix = -1
        # feature initialization
        self.features = np.full(28, -1)
        self.features[13:26] = -2
        self.features[27] = deck_size
        for i, name in enumerate(names):
            new_player = player.Player(name, self.deck.take(self.hand_size))
            if name == "Kraudia" and self.kraudia_ix < 0:
                self.kraudia_ix = i
                for card in new_player.cards:
                    if card.suit == self.deck.trump_suit:
                        self.features[13 + card.num_value] = 0
            self.players.append(new_player)
        self.player_count = len(self.players)
        assert self.player_count > 1 and self.player_count < 8, \
                "Player count does not make sense"
        self.field = field.Field()
        self.defender_ix = -1

    def find_beginner(self):
        """Returns the index of the player with the lowest trump and
        the card or a random index and the bottom trump if no player
        has a trump in hand."""

        mini = 13
        beginner_ix = np.random.randint(self.player_count)
        beginner_card = self.deck.bottom_trump
        for i, player in enumerate(self.players):
            if mini == 2 or self.deck.bottom_trump == 2 and mini == 3:
                break
            for card in player.cards:
                if card.suit == self.deck.trump_suit and card.num_value < mini:
                    mini = card.num_value
                    beginner_ix = i
                    beginner_card = card
                    if mini == 2 or self.deck.bottom_trump == 2 and mini == 3:
                        break
        return beginner_ix, beginner_card

    def attack(self, attacker_ix, cards):
        """Attacks the defender with the attacker's given cards."""

        assert attacker_ix < self.player_count, ("One of the players does "
                "not exist")
        attacker = self.players[attacker_ix]
        defender = self.players[defender_ix]
        assert (len(self.field.attack_cards) + len(self.field.defended_pairs)
                + len(cards) <= min(len(defender.cards), self.hand_size)), \
                "Number of attack cards exceeds allowed number"
        for card in cards:
            assert card in attacker.cards, ("Attacker does not have one of "
                    "the cards")
        if self.field.is_empty():
            assert deck.same_value(cards), ("Cards must have the same value "
                    "for initial attack.")
        else:
            assert self.field.values_on_field(cards), ("One of the cards' "
                    "values is not on the field")
        attacker.attack(cards)
        self.field.attack(cards)
        # update features
        for card in cards:
            if card.suit == self.deck.trump_suit:
                self.features[13 + card.num_value] = -1
        if attacker_ix == next_neighbour() or attacker_ix == prev_neighbour():
            for card in cards:
                if card.suit != self.deck.trump_suit:
                    sub_neighbour_card(card)

    def defend(self, to_defend, card):
        """Defends the card to defend with the given card."""

        defender = self.players[defender_ix]
        assert card in defender.cards, "Defender does not have the card"
        assert self.field.on_field_attack(to_defend), ("Card to defend is not "
                "on the field as an attack")
        assert (card.value > to_defend.value and card.suit == to_defend.suit
                or card.suit == self.deck.trump_suit), "Card is too low"
        if to_defend.suit == self.deck.trump_suit:
            assert card.value > to_defend.value, "Card is too low"
        defender.defend(to_defend, card)
        self.field.defend(to_defend, card)
        # update features
        if card.suit == self.deck.trump_suit:
            self.features[13 + card.num_value] = -1
        if defender_ix == next_neighbour() or defender_ix == prev_neighbour():
            if card.suit != self.deck.trump_suit:
                sub_neighbour_card(card)

    def push(self, cards):
        """Pushes the cards to the next player."""

        assert self.field.defended_pairs == [], ("Can't push after "
                "having defended")
        defender = self.players[defender_ix]
        assert (len(self.field.attack_cards) + len(cards)
                <= min(len(self.player[self.next_neighbour()].cards),
                self.hand_size)), ("Number of attack cards exceeds "
                "allowed number") 
        defender.push(cards)
        self.field.push(cards)
        # update features
        for card in cards:
            if card.suit == self.deck.trump_suit:
                self.features[13 + card.num_value] = -1
                break
        if defender_ix == next_neighbour() or defender_ix == prev_neighbour():
            for card in cards:
                if card.suit != self.deck.trump_suit:
                    sub_neighbour_card(card)
        self.update_defender()

    def take(self):
        """Makes the defender take all the cards on the field."""

        assert not self.field.is_empty(), "Field can't be empty"
        cards = self.field.take()
        self.players[defender_ix].take(cards)
        # update features
        for card in cards:
            if card.suit == self.deck.trump_suit:
                self.features[13 + card.num_value] = \
                        index_from_kraudia(defender_ix)
        if defender_ix == prev_neighbour() or defender_ix == next_neighbour():
            for card in cards:
                if card.suit != self.deck.trump_suit:
                    self.features[card.num_value] += 1
        self.update_defender(2)

    def check(self, player_ix):
        """Tells the others that the player does not want to attack or
        defend anymore."""

        assert player_ix < self.player_count, "Player does not exist"
        self.players[player_ix].check()

    def uncheck(self, player_ix):
        """Resets the flag for checking for the given player."""

        assert player_ix < self.player_count, "Player does not exist"
        self.players[player_ix].uncheck()

    def draw(self, player_ix):
        """Draws cards for the given player until their hand is filled
        or the deck is empty."""

        assert player_ix < self.player_count, "Player does not exist"
        drawer = self.players[player_ix]
        amount = self.hand_size - len(drawer.cards)
        assert amount >= 0, "Amount is less than one"
        cards = self.deck.take(amount)
        drawer.take(cards)
        # update features
        if player_ix == kraudia_ix:
            for card in cards:
                if card.suit == self.deck.trump_suit:
                    self.features[13 + card.num_value] = 0

    def attack_ended(self):
        """Tests whether an attack is over because the maximum allowed
        number of attack cards has been placed and defended or because
        all attackers have checked."""

        return (self.players[defender_ix].cards == []
            or len(self.field.defended_pairs) == self.hand_size
            or self.players[prev_neighbour(defender_ix)].checks
            and self.players[next_neighbour(defender_ix)].checks)

    def prev_neighbour(self, player_ix=None):
        """Returns the index of the player before the player of the
        given index.
        If no index is given, kraudia_ix is used."""

        if player_ix == None:
            player_ix = self.kraudia_ix
        assert player_ix < self.player_count, "Player does not exist"
        return (player_ix - 1) % self.player_count

    def next_neighbour(self, player_ix=None):
        """Returns the index of the player after the player of the
        given index.
        If no index is given, kraudia_ix is used."""

        if player_ix == None:
            player_ix = self.kraudia_ix
        assert player_ix < self.player_count, "Player does not exist"
        return (player_ix + 1) % self.player_count

    def index_from_kraudia(self, player_ix):
        """Returns the index of the player sitting player_ix times from
        kraudia_ix in clockwise direction.
        For example, if kraudia_ix is 2 and player_ix is 5, then this
        returns 3 with appropriate wrapping."""

        assert player_ix < self.player_count, "Player does not exist"
        return (player_ix - kraudia_ix) % self.player_count 

    def update_defender(self, count=1):
        """Increases defender index by count (with wrapping)."""

        self.defender_ix = (self.defender_ix + count) % self.player_count

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