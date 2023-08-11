from fsrs import *
from datetime import datetime


def test_repeat():
    f = FSRS()
    f.params.w = (
        1.14,
        1.01,
        5.44,
        14.67,
        5.3024,
        1.5662,
        1.2503,
        0.0028,
        1.5489,
        0.1763,
        0.9953,
        2.7473,
        0.0179,
        0.3105,
        0.3976,
        0.0,
        2.0902,
    )
    card = Card()

    ratings = (
        Rating.Good,
        Rating.Good,
        Rating.Good,
        Rating.Good,
        Rating.Good,
        Rating.Good,
        Rating.Again,
        Rating.Again,
        Rating.Good,
        Rating.Good,
        Rating.Good,
        Rating.Good,
        Rating.Good,
    )
    ivl_history = []
    now = datetime.now()
    for rating in ratings:
        f.review(card, rating, now=now)

        ivl_history.append(card.scheduled_days)
        now = card.due

    print(f"Output: {ivl_history}")
    assert ivl_history == [0, 5, 16, 43, 106, 236, 0, 0, 12, 25, 47, 85, 147]


if __name__ == "__main__":
    test_repeat()
