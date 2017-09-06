class Player:
    """A player in the game with his own cards."""

    def __init__(self, name, cards):
        """Construct a player with the given name and cards."""
        self.name = name
        self.cards = cards
        self.checks = False

    def attack(self, cards):
        """Attack with the given cards."""
        for card in cards:
            assert card in self.cards, 'Cannot attack with a card not in hand'
            self.cards.remove(card)

    def defend(self, to_defend, card):
        """Defend the card to defend with the given card."""
        assert card in self.cards, 'Cannot defend with a card not in hand'
        self.cards.remove(card)

    def push(self, cards):
        """Push the cards to the next player."""
        suit = cards[0].suit
        for card in cards:
            assert card in self.cards, 'Cannot push with a card not in hand'
            assert card.suit == suit, 'Cannot push with non-matching suits'
            self.cards.remove(card)

    def take(self, cards):
        """Make the player take the given cards in his hand."""
        self.cards.extend(cards)

    def check(self):
        """Tell the others that the player does not want to attack or
        defend anymore.
        """
        self.checks = True

    def uncheck(self):
        """Reset the flag for checking."""
        self.checks = False