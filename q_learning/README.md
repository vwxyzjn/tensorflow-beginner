# Notes

## 1/15/2019

While doing experiments with the env `Taxi-v2`, I had the hardest time figuring out why my q-learning algorithm wasn't doing great. It turns out that I mistyped the formula...

What I had was

```python
q_table[old_raw_state, action] = (1 - ALPHA) * oldv + ALPHA * (
    reward + GAMMA * np.argmax(q_table[raw_state])
)
```

while the correct solution should have been

```python
oldv = q_table[(old_raw_state, action)]
q_table[old_raw_state, action] = (1 - ALPHA) * oldv + ALPHA * (
    reward + GAMMA * np.max(q_table[raw_state])
)
```

Haven't made this kind of mistakes in a while.
