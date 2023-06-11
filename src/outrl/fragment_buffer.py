"""A replay buffer that can reconstruct episodes from fragments."""
from itertools import chain
from collections import deque
from typing import Dict, Union, List, Optional, Any

import torch


class FragmentBuffer:
    """A replay buffer that can re-assemble episodes from timesteps, and can
    sample fragments of those episodes.

    FragmentBuffer attempts to be as agnostic as possible to the type of data
    being stored in it.
    Data will be stored in torch.Tensors when possible, and in lists otherwise.
    Args:
        n_episodes (int): Number of episodes to store at once.
        max_episode_length (int): Maximum episode length. If storing a trailing
        observation is desired, pass max_episode_length + 1.

    """

    def __init__(self, n_episodes: int, max_episode_length: int):
        self.buffers: Dict[str, Union[torch.Tensor, List[List[Any]]]] = {}
        self.episode_length_so_far: torch.Tensor = torch.zeros(
            n_episodes, dtype=torch.int64
        )
        self.episode_complete: torch.Tensor = torch.zeros(n_episodes, dtype=torch.bool)

        # These fields are kept coordinated by the on_out_of_space method
        self._free_episodes = deque(range(n_episodes))
        self._next_allocation_index: int = 0

        # Changing these fields does not work, make them private
        self._n_episodes = n_episodes
        self._max_episode_length = max_episode_length

    @property
    def n_episodes(self):
        return self._n_episodes

    @property
    def max_episode_length(self):
        return self._max_episode_length

    def on_out_of_space(self, space_needed: int):
        for _ in range(space_needed):
            self.episode_length_so_far[self._next_allocation_index] = 0
            self._free_episodes.append(self._next_allocation_index)
            self._next_allocation_index += 1
            self._next_allocation_index %= self._n_episodes

    def clear(self):
        self._free_episodes = deque(range(self._n_episodes))
        self.episode_length_so_far[:] = 0
        self.episode_complete[:] = False
        for v in self.buffers.values():
            if isinstance(v, torch.Tensor):
                v.zero_()
            else:
                for i in range(self._n_episodes):
                    v[i] = [None] * self._max_episode_length

    def get_full_episodes(self, key: str) -> Union[torch.Tensor, List[Any]]:
        buf = self.buffers[key]
        if isinstance(buf, torch.Tensor):
            return buf[self.episode_complete]
        else:
            return [ep for (i, ep) in enumerate(buf) if self.episode_complete[i]]

    def start_episode(self) -> Optional[int]:
        eps = self.start_episodes(1)
        if eps:
            return eps[0]
        else:
            return None

    def start_episodes(self, count: int) -> List[int]:
        episode_indices = []
        for _ in range(count):
            if not self._free_episodes:
                continue
            episode_index = self._free_episodes.popleft()
            self.episode_length_so_far[episode_index] = 0
            self.episode_complete[episode_index] = False
            episode_indices.append(episode_index)
            for v in self.buffers.values():
                if isinstance(v, torch.Tensor):
                    v[episode_index] = torch.zeros_like(v[episode_index])
                else:
                    v[episode_index] = [None] * self._max_episode_length
        return episode_indices

    def end_episode(self, episode_index: int):
        self.episode_complete[episode_index] = True

    def store_timestep(
        self, episode_index: int, timestep: Dict[str, Union[torch.Tensor, List[Any]]]
    ):
        timesteps = {}
        for k, v in timestep.items():
            if isinstance(v, torch.Tensor):
                timesteps[k] = v.unsqueeze(0)
            else:
                timesteps[k] = [v]
        self.store_timesteps(episode_index, timesteps)

    def store_timesteps(
        self,
        episode_index: int,
        timesteps: Dict[str, Union[torch.Tensor, List[List[Any]]]],
    ):
        assert timesteps
        assert not self.episode_complete[episode_index]
        start_step = self.episode_length_so_far[episode_index]
        n_steps = None
        for k, v in timesteps.items():
            if n_steps is None:
                n_steps = len(v)
            else:
                assert n_steps == len(v)

            if isinstance(v, list):
                buffer_m: Optional[List[List[Any]]] = self.buffers.get(k)
                if buffer_m is None:
                    buffer: List[List[Any]] = [
                        [None] * self._max_episode_length
                        for _ in range(self._n_episodes)
                    ]
                    self.buffers[k] = buffer
                else:
                    buffer = buffer_m
                    if not isinstance(buffer, list):
                        raise TypeError(
                            f"Expected a timestep sequence of type {type(buffer)} "
                            f"for {k} but got one of type {type(v)}"
                        )
                buffer[episode_index][start_step : start_step + n_steps] = v
            elif isinstance(v, torch.Tensor):
                buffer_tensor: torch.Tensor = self.buffers.get(k)
                if buffer_tensor is None:
                    buffer_tensor: torch.Tensor = torch.zeros(
                        (
                            self._n_episodes,
                            self._max_episode_length,
                        )
                        + v.shape[1:],
                        dtype=v.dtype,
                    )
                    self.buffers[k] = buffer_tensor
                else:
                    if not isinstance(buffer_tensor, torch.Tensor):
                        raise TypeError(
                            f"Expected a timestep sequence of type {type(buffer_tensor)} "
                            f"for {k} but got one of type {type(v)}"
                        )
                buffer_tensor[episode_index][start_step : start_step + n_steps] = v
            else:
                raise TypeError("Unsupported timestep sequence type {}", type(v))
        self.episode_length_so_far[episode_index] += n_steps

    def valid_indices(self, fragment_length=1):
        assert fragment_length >= 1
        all_allocations = []
        for episode_index in range(self._n_episodes):
            # Invalid episodes are filtered by being zero length
            length = self.episode_length_so_far[episode_index]
            last_valid_start = length - fragment_length
            if last_valid_start < 0:
                continue
            valid_indices = torch.stack(
                [
                    episode_index * torch.ones(last_valid_start + 1, dtype=torch.int64),
                    torch.arange(last_valid_start + 1),
                ],
                dim=1,
            )
            all_allocations.append(valid_indices)
        return torch.cat(all_allocations)

    def index_timesteps(
        self,
        start_indices: torch.tensor,
        extra_buffers: Dict[str, Union[torch.Tensor, List[List[Any]]]] = {},
    ) -> Dict[str, Union[torch.Tensor, List[List[Any]]]]:
        """Look up a set of length 1 fragments, using the provided start_indices.
        Output tensors will have shape (fragment_index, time_index, *)
        start_indices should be valid indices, with for a length passed here at most equal to the
        than the fragment_length passed to valid_indices.

        extra_buffers will be sampled from using the same indexing logic.
        """
        fragments = {}
        # This is an optimization to avoid the nested stack loop.
        # We still want to produce a sequence of fragments (not a
        # sequence of steps), so add an indexing dimension
        indices = start_indices.unsqueeze(2)
        max_lengths = self.episode_length_so_far[start_indices[:, 0]]
        end_indices, _ = torch.min(
            torch.stack([start_indices[:, 1] + 1, max_lengths]), dim=0
        )
        for k, v in chain(self.buffers.items(), extra_buffers.items()):
            if isinstance(v, list):
                fragments[k] = [
                    v[start_indices[i, 0]][start_indices[i, 1] : end_indices[i]]
                    for i in range(len(start_indices))
                ]
            else:
                # Ideally this would be: v[start_indices[:, 0], start_indices[:, 1]:end_indices]
                # But pytorch doesn't support range indexing from a tensor.
                fragments[k] = v[indices[:, 0], indices[:, 1]]
        return fragments

    def index(
        self,
        start_indices: torch.tensor,
        lengths: torch.tensor or int = 1,
        extra_buffers: Dict[str, Union[torch.Tensor, List[List[Any]]]] = {},
    ) -> Dict[str, Union[torch.Tensor, List[List[Any]]]]:
        """Look up a set of fragments, using the provided start_indices.
        Output tensors will have shape (fragment_index, time_index, *)
        start_indices should be valid indices, with for a length passed here at most equal to the
        than the fragment_length passed to valid_indices.

        extra_buffers will be sampled from using the same indexing logic.
        """
        if isinstance(lengths, int) and lengths == 1:
            return self.index_timesteps(
                start_indices=start_indices, extra_buffers=extra_buffers
            )
        fragments = {}
        if isinstance(lengths, int):
            lengths_tensor: torch.Tensor = lengths * torch.ones(
                len(start_indices), dtype=torch.int64
            )
        else:
            lengths_tensor: torch.Tensor = lengths
        max_lengths = self.episode_length_so_far[start_indices[:, 0]]
        end_indices, _ = torch.min(
            torch.stack([start_indices[:, 1] + lengths_tensor, max_lengths]), dim=0
        )
        clipped_lengths = end_indices - start_indices[:, 1]
        # If fragments are not all of equal length, then they cannot be stacked into a Tensor
        assert (clipped_lengths == lengths_tensor).all()
        # There's does not appear to be a good way to do range indexing without a for loop.
        # https://stackoverflow.com/questions/61034839/pytorch-indexing-a-range-of-multiple-indices
        indices = torch.stack(
            [
                torch.stack(
                    [
                        start_indices[i, 0]
                        * torch.ones((int(lengths_tensor[i]),), dtype=torch.int64),
                        torch.arange(start_indices[i, 1], end_indices[i]),
                    ]
                )
                for i in range(len(lengths_tensor))
            ]
        )
        # Dimensions: (expanded) index dim, episode, index in episode
        assert len(indices.shape) == 3
        assert len(indices) <= torch.sum(lengths_tensor)
        for k, v in chain(self.buffers.items(), extra_buffers.items()):
            if isinstance(v, list):
                fragments[k] = [
                    v[start_indices[i, 0]][start_indices[i, 1] : end_indices[i]]
                    for i in range(len(start_indices))
                ]
            else:
                # Ideally this would be: v[start_indices[:, 0], start_indices[:, 1]:end_indices]
                # But pytorch doesn't support range indexing from a tensor.
                fragments[k] = v[indices[:, 0], indices[:, 1]]
        return fragments


class FragmentDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        buffer: FragmentBuffer,
        *,
        extra_buffers: Dict[str, Union[torch.Tensor, List[List[Any]]]] = {},
        fragment_length: int = 1,
    ):
        self.buffer = buffer
        self.fragment_length = fragment_length
        self.extra_buffers = extra_buffers
        self.indices = self.buffer.valid_indices(fragment_length=self.fragment_length)

    def __len__(self):
        return len(self.buffer.valid_indices(fragment_length=self.fragment_length))

    def __getitem__(self, index: int):
        # Shuffling is handled by the DataLoader
        res = self.buffer.index(
            self.indices[index].unsqueeze(0),
            lengths=self.fragment_length,
            extra_buffers=self.extra_buffers,
        )
        assert len(res["rewards"].shape) == 2
        assert len(res["discounted_returns"].shape) == 2
        return res

    def __iter__(self):
        for index in range(len(self.indices)):
            yield self[index]


class FragmentDataloader:
    def __init__(
        self,
        buffer: FragmentBuffer,
        *,
        batch_size: int,
        extra_buffers: Dict[str, Union[torch.Tensor, List[List[Any]]]] = {},
        fragment_length: int = 1,
    ):
        self.buffer = buffer
        self.fragment_length = fragment_length
        self.batch_size = batch_size
        self.extra_buffers = extra_buffers
        self.indices = self.buffer.valid_indices(fragment_length=self.fragment_length)
        self.shuffle_indices = torch.randperm(len(self.indices))
        self.next_start_index = 0

    def reset(self):
        self.shuffle_indices = torch.randperm(len(self.indices))
        self.indices = self.buffer.valid_indices(fragment_length=self.fragment_length)
        self.next_start_index = 0

    def __iter__(self):
        while self.next_start_index + self.batch_size <= len(self.shuffle_indices):
            indices = self.indices[
                self.shuffle_indices[
                    self.next_start_index : self.next_start_index + self.batch_size
                ]
            ]
            self.next_start_index += self.batch_size
            yield self.buffer.index(
                indices,
                lengths=self.fragment_length,
                extra_buffers=self.extra_buffers,
            )


def test_indexing():
    buffer = FragmentBuffer(n_episodes=7, max_episode_length=5)
    for x in range(2):
        ep = buffer.start_episode()
        for i in range(x + 3):
            buffer.store_timestep(ep, {"t": torch.tensor(i)})
    indices = buffer.valid_indices(fragment_length=4)
    assert (indices == torch.tensor([[1, 0]])).all()
    indices = buffer.valid_indices(fragment_length=3)
    assert (indices == torch.tensor([[0, 0], [1, 0], [1, 1]])).all()
    indices = buffer.valid_indices(fragment_length=2)
    assert (indices == torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])).all()
    indices = buffer.valid_indices()
    assert (
        indices
        == torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3]])
    ).all()
    assert len(buffer.index(indices)["t"].shape) == 2
    assert (
        buffer.index(indices)["t"] == torch.tensor([[0], [1], [2], [0], [1], [2], [3]])
    ).all()
    indices = buffer.valid_indices(fragment_length=2)
    assert (
        buffer.index(indices, lengths=2)["t"]
        == torch.tensor([[0, 1], [1, 2], [0, 1], [1, 2], [2, 3]])
    ).all()


def test_list():
    buffer = FragmentBuffer(n_episodes=7, max_episode_length=5)
    for x in range(2):
        ep = buffer.start_episode()
        for i in range(x + 3):
            buffer.store_timestep(ep, {"t": "abcd"[i]})
    indices = buffer.valid_indices(fragment_length=4)
    assert (indices == torch.tensor([[1, 0]])).all()
    indices = buffer.valid_indices(fragment_length=3)
    assert (indices == torch.tensor([[0, 0], [1, 0], [1, 1]])).all()
    indices = buffer.valid_indices(fragment_length=2)
    assert (indices == torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])).all()
    indices = buffer.valid_indices()
    assert (
        indices
        == torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3]])
    ).all()
    assert buffer.index(indices)["t"] == [
        ["a"],
        ["b"],
        ["c"],
        ["a"],
        ["b"],
        ["c"],
        ["d"],
    ]
    indices = buffer.valid_indices(fragment_length=2)
    assert buffer.index(indices, lengths=2)["t"] == [
        ["a", "b"],
        ["b", "c"],
        ["a", "b"],
        ["b", "c"],
        ["c", "d"],
    ]
