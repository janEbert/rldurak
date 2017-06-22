class Player:
    """A player in the game with his own cards."""

    def __init__(self, name, cards):
        self.name = name
        self.cards = cards
        self.checks = False

    def attack(self, cards):
        """Attacks with the given cards."""

        for card in cards:
            assert card in self.cards, "Can't attack with a card not in hand"
            self.cards.remove(card)

    def defend(self, to_defend, card):
        """Defends the card to defend with the given card."""

        assert card in self.cards, "Can't defend with a card not in hand"
        self.cards.remove(card)

    def push(self, cards):
        """Pushes the cards to the next player."""

        suit = cards[0].suit
        for card in cards:
            assert card in self.cards, "Can't push with a card not in hand"
            assert card.suit == suit, "Can't push with non-matching suits"
            self.cards.remove(card)

    def take(self, cards):
        """Makes the player take the given cards in his hand."""

        self.cards.extend(cards)

    def check(self):
        """Tells the others that the player does not want to attack or
        defend anymore."""

        self.checks = True

    def uncheck(self):
        """Resets the flag for checking."""

        self.checks = False

    def reshuffle(self):
        """Makes the game shuffle and give out cards again.

        Only usable if more five or more cards of the same suit are
        in hand."""

        pass