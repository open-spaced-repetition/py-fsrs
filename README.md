<div align="center">
  <img src="https://avatars.githubusercontent.com/u/96821265?s=200&v=4" height="100" alt="Open Spaced Repetition logo"/>
</div>
<br />
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
You can install the `fsrs` python package from PyPI using pip:
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
card_object = Card()
```

Review the card
```python
scheduling_cards = f.repeat(card_object)
```

Choose a rating and update the card object
```python
# you can choose one of the four possible ratings
"""
Rating.Again # forget; incorrect response
Rating.Hard # recall; correct response recalled with serious difficulty
Rating.Good # recall; correct response after a hesitation
Rating.Easy # recall; perfect response
"""

card_rating = Rating.Good

card_object = scheduling_cards[card_rating].card
```

See when the card is due next
```python
from datetime import datetime, timezone

due = card_object.due

# how much time between when the card is due and now
time_delta = due - datetime.now(timezone.utc)

print(f"Card due: at {repr(due)}")
print(f"Card due in {time_delta.seconds} seconds")

"""
> Card due: at datetime.datetime(2024, 7, 6, 20, 6, 39, 147417, tzinfo=datetime.timezone.utc)
> Card due in: 599 seconds
"""
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

Get the review log for a given rating
```python
review_log = scheduling_cards[card_rating].review_log
```

Get the schdeduled days after rating a card
```python
scheduled_days = card_object.scheduled_days
```

## License

Distributed under the MIT License. See `LICENSE` for more information.
