<div align="center">
  <img src="https://avatars.githubusercontent.com/u/96821265?s=200&v=4" height="100" alt="Open Spaced Repetition logo"/>
</div>
<div align="center">

# Py-FSRS

</div>
<div align="center">
  <em>🧠🔄 Build your own Spaced Repetition System in Python 🧠🔄   </em>
</div>
<br />
<div align="center" style="text-decoration: none;">
    <a href="https://pypi.org/project/fsrs/"><img src="https://img.shields.io/pypi/v/fsrs"></a>
    <a href="https://github.com/open-spaced-repetition/py-fsrs/blob/main/LICENSE" style="text-decoration: none;"><img src="https://img.shields.io/badge/License-MIT-brightgreen.svg"></a>
    <a href="https://github.com/psf/black" style="text-decoration: none;"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</div>
<br />


**Py-FSRS is a python package that allows developers to easily create their own spaced repetition system using the <a href="https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler">Free Spaced Repetition Scheduler algorithm</a>.**


---


## Installation
You can install the `fsrs` python package from [PyPI](https://pypi.org/project/fsrs/) using pip:
```
pip install fsrs
```

## Quickstart

Import and initialize the FSRS scheduler

```python
from fsrs import *

f = FSRS()
```

Create a new Card object
```python
# all new cards are 'due' immediately upon creation
card = Card()
```

Choose a rating and review the card
```python
# you can choose one of the four possible ratings
"""
Rating.Again # forget; incorrect response
Rating.Hard # recall; correct response recalled with serious difficulty
Rating.Good # recall; correct response after a hesitation
Rating.Easy # recall; perfect response
"""

rating = Rating.Good

card, review_log = f.review_card(card, rating)
```

See when the card is due next
```python
from datetime import datetime, timezone

due = card.due

# how much time between when the card is due and now
time_delta = due - datetime.now(timezone.utc)

print(f"Card due: at {repr(due)}")
print(f"Card due in {time_delta.seconds} seconds")

"""
> Card due: at datetime.datetime(2024, 7, 12, 18, 16, 4, 429428, tzinfo=datetime.timezone.utc)
> Card due in: 599 seconds
"""
```

## Usage

### Custom scheduler

You can initialize the FSRS scheduler with your own custom weights as well as desired retention rate and maximum interval.

```python
f = FSRS(
    w=(
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
    ),
    request_retention=0.85,
    maximum_interval=3650,
)
```

### Advanced reviewing of cards

Aside from using the convenience method `review_card`, there is also the `repeat` method:

```python
from datetime import datetime, timezone

# custom review time (must be UTC)
review_time = datetime(2024, 7, 13, 20, 7, 56, 150101, tzinfo=timezone.utc)

scheduling_cards = f.repeat(card, review_time)

# can get updated cards for each possible rating
card_Again = scheduling_cards[Rating.Again].card
card_Hard = scheduling_cards[Rating.Hard].card
card_Good = scheduling_cards[Rating.Good].card
card_Easy = scheduling_cards[Rating.Easy].card

# get next review interval for each rating
scheduled_days_Again = card_Again.scheduled_days
scheduled_days_Hard = card_Hard.scheduled_days
scheduled_days_Good = card_Good.scheduled_days
scheduled_days_Easy = card_Easy.scheduled_days

# choose a rating and update the card
rating = Rating.Good
card = scheduling_cards[rating].card

# get the corresponding review log for the review
review_log = scheduling_cards[rating].review_log
```

### Serialization

`Card` and `ReviewLog` objects are JSON-serializable via their `to_dict` and `from_dict` methods for easy database storage:

```python
# serialize before storage
card_dict = card.to_dict()
review_log_dict = review_log.to_dict()

# deserialize from dict
new_card = Card.from_dict(card_dict)
new_review_log = ReviewLog.from_dict(review_log_dict)
```

## Reference

Card objects have one of four possible states
```python
State.New # Never been studied
State.Learning # Been studied for the first time recently
State.Review # Graduate from learning state
State.Relearning # Forgotten in review state
```

There are four possible ratings when reviewing a card object:
```python
Rating.Again # forget; incorrect response
Rating.Hard # recall; correct response recalled with serious difficulty
Rating.Good # recall; correct response after a hesitation
Rating.Easy # recall; perfect response
```

## Contribute

Checkout [CONTRIBUTING](CONTRIBUTING.md) to help improve Py-FSRS!

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
