from typing import Union, Tuple

from torch import Tensor


# Common data types for actor/critic modules -- for better readability
State = Tensor
Action = Tensor
LogProb = Union[Tensor, None]
Value = Tensor
Done = Tensor
Batch = Tuple[Tensor, ...]
