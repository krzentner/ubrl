"""A replay buffer that can reconstruct episodes from fragments."""
from itertools import chain
from collections import deque
from typing import Callable, Dict, Union, List, Optional, Any, Tuple

import torch


class FragmentBuffer:
    """A replay buffer that can re-assemble episodes from timesteps, and can
    sample fragments of those episodes.

    FragmentBuffer attempts to be as agnostic as possible to the type of data
    being stored in it.
    Data will be stored in torch.Tensors when possible, and in lists otherwise.

    The number of episodes, and the max length of each episode, must be fixed
    on creation.

    Episodes are indexed by an "episode index", which is between 0 and the
    number of episodes.

    Allocate an episode index by calling start_episode:

        episode_index = buffer.start_episode()

    Then, insert timesteps into it using store_timesteps:

        buffer.store_timesteps(episode_index, {"observations": [...], ...})

    Finally, end the episode with end_episode, possibly passing in episode metadata:

        buffer.end_episode(episode_index, {"last_observation": ..., "timed_out": True, ...})

    Fields:
        buffers: Contains per-timestep data, indexed by key (e.g. "rewards",
            "observations"), then episode index, then timestep.
        episode_data: Contains per-episode data, indexed by episode index, then
            by key. Does not need to have matching keys across episodes.
        episode_length_so_far: Indexed by episode index. 0 before any timesteps
            have been inserted.
        episode_complete: Indexed by episode index. Is only set if end_episode
            is called.

    Args:
        n_episodes (int): Number of episodes to store at once.
        max_episode_length (int): Maximum episode length.
        eviction_policy (str): How to make space when the buffer is full.
            Should be one of "fifo", "uniform", "evict_least", or "evict_most". If given a value
            besides "fifo" (the default), eviction_policy_key should be passed.
        eviction_policy_key (str): Key to look up in episode_data when using an
            eviction_policy of "uniform", "evict_least", or "evict_most".

    """

    def __init__(self, n_episodes: int, max_episode_length: int,
                 eviction_policy: str = "fifo", eviction_policy_key: Optional[str] = None):
        # User controlled per timestep data
        self.buffers: Dict[str, Union[torch.Tensor, List[List[Any]]]] = {}
        # User controlled per episode data
        self.episode_data: List[Dict[str, Any]] = [{} for _ in range(n_episodes)]

        # FragmentBuffer internal (but useful to users) per-episode data
        self.episode_length_so_far: torch.Tensor = torch.zeros(
            n_episodes, dtype=torch.int64
        )
        self.episode_complete: torch.Tensor = torch.zeros(n_episodes, dtype=torch.bool)

        assert eviction_policy in ["fifo", "uniform", "evict_least", "evict_most"]
        self.eviction_policy = eviction_policy
        if eviction_policy != "fifo":
            assert eviction_policy_key is not None
        # This key needs to be present in episode_data
        self.eviction_policy_key = eviction_policy_key

        # Only used for fifo eviction policy
        self._next_allocation_index: int = 0

        # Only used in non-fifo eviction policy
        self._sorted_episode_data: List[Tuple[int, Dict[str, Any]]] = []

        # This field is kept coordinated by above fields the on_out_of_space method
        self._free_episodes = deque(range(n_episodes))

        # Changing these fields does not cause expected results, make them read-only
        self._n_episodes = n_episodes
        self._max_episode_length = max_episode_length

    @property
    def n_episodes(self):
        return self._n_episodes

    @property
    def max_episode_length(self):
        return self._max_episode_length

    def clear_episode(self, episode_index: int):
        self.episode_length_so_far[episode_index] = 0
        self.episode_data[episode_index] = {}
        self.episode_complete[episode_index] = False
        for v in self.buffers.values():
            if isinstance(v, torch.Tensor):
                v[episode_index] = torch.zeros_like(v[episode_index])
            else:
                v[episode_index] = [None] * self._max_episode_length
        self._free_episodes.append(episode_index)

    def on_out_of_space(self, space_needed: int):
        if self.eviction_policy == "fifo":
            for _ in range(space_needed):
                self.clear_episode(self._next_allocation_index)
                self._next_allocation_index += 1
                self._next_allocation_index %= self._n_episodes
        else:
            eviction_key = self.eviction_policy_key
            assert eviction_key is not None
            # Re-sort the list.
            # This could probably be avoided using a binary tree, but profile
            # before/after changing it.
            self._sorted_episode_data = sorted(enumerate(self.episode_data), key=lambda d: d[1][eviction_key])
            if self.eviction_policy == "uniform":
                total_dists = []
                for i, (original_i, data) in enumerate(self._sorted_episode_data):
                    # Uniform will never remove either end-point
                    if i == 0 or i + 1 == len(self._sorted_episode_data):
                        continue
                    dist_to_prev = data[eviction_key] - self._sorted_episode_data[i - 1][1][eviction_key]
                    dist_to_next = self._sorted_episode_data[i + 1][1][eviction_key] - data[eviction_key]
                    # abs should not be necessary with this subtraction order, but may as well
                    total_dist = abs(dist_to_prev) + abs(dist_to_next)
                    total_dists.append((total_dist, original_i))
                for _, original_i in sorted(total_dists)[:space_needed]:
                    self.clear_episode(original_i)
            elif self.eviction_policy == "evict_least":
                for original_i, _ in self._sorted_episode_data[:space_needed]:
                    self.clear_episode(original_i)
            elif self.eviction_policy == "evict_most":
                for original_i, _ in self._sorted_episode_data[-space_needed:]:
                    self.clear_episode(original_i)
            else:
                raise ValueError(f"Invalid eviction_policy {self.eviction_policy}")


    def clear_all(self):
        self._free_episodes = deque(range(self._n_episodes))
        self.episode_length_so_far[:] = 0
        self.episode_complete[:] = False
        for v in self.buffers.values():
            if isinstance(v, torch.Tensor):
                v.zero_()
            else:
                for i in range(self._n_episodes):
                    v[i] = [None] * self._max_episode_length
        self.episode_data = [{} for _ in range(self._n_episodes)]

    def valid_mask(self):
        mask = torch.zeros((self.n_episodes, self.max_episode_length), dtype=torch.bool)
        for i, episode_length in enumerate(self.episode_length_so_far):
            mask[i, :episode_length] = True
        return mask

    def get_full_episodes(self, key: str) -> Union[torch.Tensor, List[Any]]:
        buf = self.buffers[key]
        if isinstance(buf, torch.Tensor):
            return buf[self.episode_complete]
        else:
            return [ep for (i, ep) in enumerate(buf) if self.episode_complete[i]]

    def start_episode(self) -> int:
        return self.start_episodes(1)[0]

    def start_episodes(self, count: int) -> List[int]:
        episode_indices = []
        if len(self._free_episodes) < count:
            self.on_out_of_space(count - len(self._free_episodes))
        for _ in range(count):
            episode_index = self._free_episodes.popleft()
            episode_indices.append(episode_index)
        return episode_indices

    def end_episode(self, episode_index: int, episode_data_update: Optional[Dict[str, Any]] = None):
        if episode_data_update:
            self.episode_data[episode_index].update(episode_data_update)
        self.episode_complete[episode_index] = True

    def store_timestep(
        self, episode_index: int, timestep: Dict[str, Union[torch.Tensor, Any]]
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
        assert len(start_indices.shape) == 2
        assert (start_indices[:, 0] < self._n_episodes).all()
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
                # Hotpath here

                # TODO: Evaluate performance cost of changing the type of self.buffers[k]
                # This is the only place where having self.buffers[k] be a tensor saves time
                # In particular, we could change the type of each buffer to
                # List[Union[List[Any], Tensor]] (instead of
                # Union[List[List[Any]], Tensor]), and this is the only logic
                # that would get slower.
                # However, this line is the hot-path for single-timestep sampling
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
        assert len(start_indices.shape) == 2
        assert (start_indices[:, 0] < self._n_episodes).all()
        if isinstance(lengths, int) and lengths == 1:
            # Opt-in to the hotpath if we can
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
    """Implements the dataset API around a FragmentBuffer.

    You should probably use the FragmentDataloader API instead, but if you need
    to use the standard dataloader then this adapter exists.
    """
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
        return res

    def __iter__(self):
        for index in range(len(self.indices)):
            yield self[index]


class FragmentDataloader:
    """Implements the same API as torch.utils.data.DataLoader.

    This is mostly equivalent to just passing a FragmentDataset to a standard
    dataloader.
    However, this dataloader will call a callback at the start of every epoch
    (usually to re-fill the buffer).
    It's also faster, since it bypasses the dataset api, and can thus make
    batch lookups into the FragmentBuffer.
    """
    def __init__(
        self,
        buffer: FragmentBuffer,
        *,
        batch_size: int,
        extra_buffers: Dict[str, Union[torch.Tensor, List[List[Any]]]] = {},
        fragment_length: int = 1,
        callback: Optional[Callable] = None,
        cycles: int = 1,
        drop_last: bool = False,
    ):
        self.buffer = buffer
        self.fragment_length = fragment_length
        self.batch_size = batch_size
        self.extra_buffers = extra_buffers
        self.indices = torch.tensor(0)
        self.shuffle_indices = torch.tensor(0)
        self.next_start_index = 0
        self.callback = callback
        self.cycles = cycles
        self.drop_last = drop_last

    def reset(self):
        if self.callback is not None:
            self.callback()
        self.indices = self.buffer.valid_indices(fragment_length=self.fragment_length)
        self.shuffle_indices = torch.randperm(len(self.indices))
        self.next_start_index = 0

    def __iter__(self):
        self.reset()
        for _ in range(self.cycles):
            self.next_start_index = 0
            self.shuffle_indices = torch.randperm(len(self.indices))
            while True:
                batch_end = self.next_start_index + self.batch_size
                # If we want to include the last batch, but we've run out of room, move batch_end back.
                if not self.drop_last and batch_end > len(self.shuffle_indices):
                    batch_end = len(self.shuffle_indices)
                if batch_end > len(self.shuffle_indices) or batch_end < self.next_start_index:
                    break
                indices = self.indices[
                    self.shuffle_indices[
                        self.next_start_index : batch_end
                    ]
                ]
                self.next_start_index += self.batch_size
                yield self.buffer.index(
                    indices,
                    lengths=self.fragment_length,
                    extra_buffers=self.extra_buffers,
                )


def test_dataloader_smoke():
    buffer = FragmentBuffer(n_episodes=7, max_episode_length=5)
    # Insert 3 + 4 + 5 = 12 total timesteps
    for x in range(3):
        ep = buffer.start_episode()
        for i in range(x + 3):
            buffer.store_timestep(ep, {"t": torch.tensor(i)})
        buffer.end_episode(ep)
    dataloader = FragmentDataloader(buffer, batch_size=9, fragment_length=1)
    # Should get exactly two batches of size 9 and 3
    batches = list(dataloader)
    assert len(batches) == 2
    assert batches[0]["t"].shape == (9, 1)
    assert batches[1]["t"].shape == (3, 1)

    # Should get only one batch of size 9
    dataloader.drop_last = True
    batches = list(dataloader)
    assert len(batches) == 1
    assert batches[0]["t"].shape == (9, 1)



def test_indexing():
    buffer = FragmentBuffer(n_episodes=7, max_episode_length=5)
    for x in range(2):
        ep = buffer.start_episode()
        for i in range(x + 3):
            buffer.store_timestep(ep, {"t": torch.tensor(i)})
        buffer.end_episode(ep)
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
        buffer.end_episode(ep)
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


def test_no_end_episode_call():
    buffer = FragmentBuffer(n_episodes=5, max_episode_length=3)
    for x in range(7):
        ep = buffer.start_episode()
        for i in range(3):
            buffer.store_timestep(ep, {"x_sq": x * x})
        if x < 2:
            buffer.end_episode(ep, {"x_ep": x})
    # Check first timestep of each episode
    assert buffer.buffers["x_sq"][0][0] == 5 * 5
    assert buffer.buffers["x_sq"][1][0] == 6 * 6
    assert buffer.buffers["x_sq"][2][0] == 2 * 2
    assert buffer.buffers["x_sq"][3][0] == 3 * 3
    assert buffer.buffers["x_sq"][4][0] == 4 * 4
    assert all(len(data) == 0 for data in buffer.episode_data)


def test_eviction_fifo():
    buffer = FragmentBuffer(n_episodes=5, max_episode_length=3)
    for x in range(7):
        ep = buffer.start_episode()
        for i in range(3):
            buffer.store_timestep(ep, {"x_sq": x * x})
        buffer.end_episode(ep, {"x_ep": x})
    # Check first timestep of each episode
    assert buffer.buffers["x_sq"][0][0] == 5 * 5
    assert buffer.buffers["x_sq"][1][0] == 6 * 6
    assert buffer.buffers["x_sq"][2][0] == 2 * 2
    assert buffer.buffers["x_sq"][3][0] == 3 * 3
    assert buffer.buffers["x_sq"][4][0] == 4 * 4
    # Check episode_data
    assert buffer.episode_data[0]["x_ep"] == 5
    assert buffer.episode_data[1]["x_ep"] == 6
    assert buffer.episode_data[2]["x_ep"] == 2
    assert buffer.episode_data[3]["x_ep"] == 3
    assert buffer.episode_data[4]["x_ep"] == 4


def test_eviction_uniform():
    buffer = FragmentBuffer(n_episodes=5, max_episode_length=3, eviction_policy="uniform",
                            eviction_policy_key="x_ep")
    for x in range(7):
        ep = buffer.start_episode()
        for i in range(3):
            buffer.store_timestep(ep, {"x_sq": x * x})
        buffer.end_episode(ep, {"x_ep": x})
    # Check first timestep of each episode
    assert buffer.buffers["x_sq"][0][0] == 0 * 0
    assert buffer.buffers["x_sq"][1][0] == 5 * 5
    assert buffer.buffers["x_sq"][2][0] == 2 * 2
    assert buffer.buffers["x_sq"][3][0] == 6 * 6
    assert buffer.buffers["x_sq"][4][0] == 4 * 4
    # Check episode_data
    assert buffer.episode_data[0]["x_ep"] == 0
    assert buffer.episode_data[1]["x_ep"] == 5
    assert buffer.episode_data[2]["x_ep"] == 2
    assert buffer.episode_data[3]["x_ep"] == 6
    assert buffer.episode_data[4]["x_ep"] == 4


def test_eviction_least():
    buffer = FragmentBuffer(n_episodes=5, max_episode_length=3, eviction_policy="evict_least",
                            eviction_policy_key="x_ep")
    for x in range(7):
        ep = buffer.start_episode()
        for i in range(3):
            buffer.store_timestep(ep, {"x_sq": x * x})
        buffer.end_episode(ep, {"x_ep": -x})
    # Check first timestep of each episode
    assert buffer.buffers["x_sq"][0][0] == 0 * 0
    assert buffer.buffers["x_sq"][1][0] == 1 * 1
    assert buffer.buffers["x_sq"][2][0] == 2 * 2
    assert buffer.buffers["x_sq"][3][0] == 3 * 3
    assert buffer.buffers["x_sq"][4][0] == 6 * 6
    # Check episode_data
    assert buffer.episode_data[0]["x_ep"] == -0
    assert buffer.episode_data[1]["x_ep"] == -1
    assert buffer.episode_data[2]["x_ep"] == -2
    assert buffer.episode_data[3]["x_ep"] == -3
    assert buffer.episode_data[4]["x_ep"] == -6


def test_eviction_most():
    buffer = FragmentBuffer(n_episodes=5, max_episode_length=3, eviction_policy="evict_most",
                            eviction_policy_key="x_ep")
    for x in range(7):
        ep = buffer.start_episode()
        for i in range(3):
            buffer.store_timestep(ep, {"x_sq": x * x})
        buffer.end_episode(ep, {"x_ep": x})
    # Check first timestep of each episode
    assert buffer.buffers["x_sq"][0][0] == 0 * 0
    assert buffer.buffers["x_sq"][1][0] == 1 * 1
    assert buffer.buffers["x_sq"][2][0] == 2 * 2
    assert buffer.buffers["x_sq"][3][0] == 3 * 3
    assert buffer.buffers["x_sq"][4][0] == 6 * 6
    # Check episode_data
    assert buffer.episode_data[0]["x_ep"] == 0
    assert buffer.episode_data[1]["x_ep"] == 1
    assert buffer.episode_data[2]["x_ep"] == 2
    assert buffer.episode_data[3]["x_ep"] == 3
    assert buffer.episode_data[4]["x_ep"] == 6
