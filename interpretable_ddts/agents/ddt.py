# Created by Andrew Silva on 2/21/19
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence, cast
from typing_extensions import Self

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from interpretable_ddts.agents.ddt_agent import LeafInfo

logger = logging.getLogger(__name__)


class DDT(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        weights,
        comparators,
        leaves: int | list[LeafInfo],
        output_dim: int | None = None,
        alpha=1.0,
        is_value=False,
        use_gpu=False,
    ):
        """
        Initialize the DDT, taking in premade weights for inputs to comparators and sigmoids
        initialized tree.
        :param weights: input weights (for intelligent init or for loading)
        :param comparators: input comparators (for intelligent init or for loading)
        :param input_dim: int. always required for input dimensionality
        :param leaves: int, must be 2**N
        :param output_dim: None or int, must be an int if weights and comparators are None
        :param alpha: int. Strictness of the tree, default 1
        :param is_value: if False, outputs are passed through a Softmax final layer. Default: False
        :param use_gpu: is this a GPU-enabled network? Default: False
        """
        nn.Module.__init__(self)
        self.use_gpu = use_gpu
        self._depth: int
        if isinstance(leaves, int):
            if hasattr(int, "bit_count"):
                _power_of_two = leaves.bit_count() == 1
            else:
                _power_of_two = leaves & (leaves - 1) == 0 and leaves != 0
            if not _power_of_two:
                raise ValueError("`leaves` must be a power of 2 or a list of lists[int]")
            self._depth = int(np.floor(np.log2(leaves)))
        else:
            # With rule_list or duplication
            # logger.warning(
            #    "Unexpected leaf information type %s; using depth=4. NOTE: the value should not be passed",
            #    type(leaves),
            # )
            # self._depth = None  # should not be used
            assert weights is not None
            assert comparators is not None
            assert isinstance(leaves, list)
        self._unprocessed_leaf_info: int | list[LeafInfo] = leaves

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.selector = None

        self.comparators: nn.Parameter
        self.init_comparators(comparators)

        self.layers: nn.Parameter
        self.init_weights(weights)
        self.init_alpha(alpha)
        self.init_paths()

        self.leaf_init_information: list[LeafInfo]
        """
        List of left and right paths and their initial probabilities/logits.
        These relate to the weights stored in self.action_probs.
        """
        self.init_leaves()
        self.added_levels = nn.Sequential()

        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.is_value = is_value

    def init_comparators(self, comparators: Optional[Sequence[Sequence[float]]]):
        if comparators is None:
            # Add value for each node in the tree; for each level 2**level nodes
            comparators = np.full([2 ** (self._depth) - 1, 1], 1.0 / self.input_dim)  # pyright: ignore[reportAssignmentType]
        new_comps = torch.Tensor(comparators)
        new_comps.requires_grad = True
        if self.use_gpu:
            new_comps = new_comps.cuda()
        self.comparators = nn.Parameter(new_comps)

    def init_weights(self, weights):
        if weights is None:
            new_weights = torch.rand(2 ** (self._depth) - 1, self.input_dim)
        else:
            new_weights = torch.Tensor(weights)
        new_weights.requires_grad = True
        if self.use_gpu:
            new_weights = new_weights.cuda()
        self.layers = nn.Parameter(new_weights)

    def init_alpha(self, alpha):
        self.alpha = torch.Tensor([alpha])
        if self.use_gpu:
            self.alpha = self.alpha.cuda()
        self.alpha.requires_grad = True
        self.alpha = nn.Parameter(self.alpha)  # NOTE: Alpha is learned!

    def init_paths(self):
        """Create 0|1 boolean tensors for left and right paths."""
        if isinstance(self._unprocessed_leaf_info, list):
            left_branches = torch.zeros((len(self.layers), len(self._unprocessed_leaf_info)))
            right_branches = torch.zeros((len(self.layers), len(self._unprocessed_leaf_info)))
            for n in range(len(self._unprocessed_leaf_info)):
                for i in self._unprocessed_leaf_info[n][0]:
                    left_branches[i][n] = 1.0
                for j in self._unprocessed_leaf_info[n][1]:
                    right_branches[j][n] = 1.0
        else:
            left_branches = torch.zeros((2**self._depth - 1, 2**self._depth))
            for n in range(self._depth):
                row = 2**n - 1
                for i in range(2**self._depth):
                    col = 2 ** (self._depth - n) * i
                    end_col = col + 2 ** (self._depth - 1 - n)
                    if row + i >= len(left_branches) or end_col >= len(left_branches[row]):
                        break
                    left_branches[row + i, col:end_col] = 1.0
            right_branches = torch.zeros((2**self._depth - 1, 2**self._depth))
            left_turns = np.where(left_branches == 1)
            for row in np.unique(left_turns[0]):
                cols = left_turns[1][left_turns[0] == row]
                start_pos = cols[-1] + 1
                end_pos = start_pos + len(cols)
                right_branches[row, start_pos:end_pos] = 1.0
        left_branches.requires_grad = False
        right_branches.requires_grad = False
        if self.use_gpu:
            left_branches = left_branches.cuda()
            right_branches = right_branches.cuda()
        self.left_path_sigs: torch.BoolTensor = left_branches  # type: ignore[assignment]
        self.right_path_sigs: torch.BoolTensor = right_branches  # type: ignore[assignment]

    def init_leaves(self):
        if isinstance(self._unprocessed_leaf_info, list):
            # Probabilities for each leaf
            new_leaves = [leaf[-1] for leaf in self._unprocessed_leaf_info]
            self.leaf_init_information = self._unprocessed_leaf_info
        else:
            new_leaves: list[float | list[float]] = []

            last_level = cast(list[int], np.arange(2 ** (self._depth - 1) - 1, 2**self._depth - 1))
            going_left = True
            leaf_index = 0
            self.leaf_init_information = []
            for _level in range(2**self._depth):
                curr_node = last_level[leaf_index]
                turn_left = going_left
                left_path: list[int] = []
                right_path: list[int] = []
                while curr_node >= 0:
                    if turn_left:
                        left_path.append(int(curr_node))
                    else:
                        right_path.append(int(curr_node))
                    prev_node = int(np.ceil(curr_node / 2) - 1)
                    if curr_node // 2 > prev_node:
                        turn_left = False
                    else:
                        turn_left = True
                    curr_node = prev_node
                if going_left:
                    going_left = False
                else:
                    going_left = True
                    leaf_index += 1
                new_probs: list[float] | float
                if self.output_dim is None:
                    new_probs = np.random.uniform(
                        0,
                        1,
                        self.output_dim,
                    )  # *(1.0/self.output_dim)
                else:
                    new_probs = np.random.uniform(
                        0,
                        1,
                        self.output_dim,
                    ).tolist()  # *(1.0/self.output_dim)
                self.leaf_init_information.append((sorted(left_path), sorted(right_path), new_probs))
                new_leaves.append(new_probs)
        new_leaves = np.array(new_leaves)  # single array for tensor creation  # type: ignore
        labels = torch.Tensor(new_leaves)
        if self.use_gpu:
            labels = labels.cuda()
        labels.requires_grad = True
        self.action_probs = nn.Parameter(labels)

    def forward(self, inputs: torch.Tensor | dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        # Using _forward follows rllib interface
        return self._forward(inputs, **kwargs)

    def _forward(self, inputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, dict):
            inputs = inputs["obs"]  # rllib input

        inputs = inputs.t().expand(self.layers.size(0), *inputs.t().size())

        inputs = inputs.permute(2, 0, 1)
        comp = self.layers.mul(inputs)
        comp = comp.sum(dim=2).unsqueeze(-1)
        comp = comp.sub(self.comparators.expand(inputs.size(0), *self.comparators.size()))
        comp = comp.mul(self.alpha)
        sig_vals: torch.Tensor = self.sig(comp)

        sig_vals = sig_vals.view(inputs.size(0), -1)

        one_minus_sig = torch.ones(sig_vals.size())
        if self.use_gpu:
            one_minus_sig = one_minus_sig.to("cuda")

        one_minus_sig = torch.sub(one_minus_sig, sig_vals)

        left_path_mask: torch.BoolTensor = self.left_path_sigs.t()  # type: ignore[assignment]
        right_path_mask: torch.BoolTensor = self.right_path_sigs.t()  # type: ignore[assignment]
        left_path_probs = left_path_mask.expand(
            inputs.size(0),
            *left_path_mask.size(),
        ) * sig_vals.unsqueeze(1)
        right_path_probs = right_path_mask.expand(
            inputs.size(0),
            *right_path_mask.size(),
        ) * one_minus_sig.unsqueeze(1)
        left_path_probs = left_path_probs.permute(0, 2, 1)
        right_path_probs = right_path_probs.permute(0, 2, 1)

        # We don't want 0s to ruin leaf probabilities, so replace them with 1s so they don't affect the product
        left_filler = torch.zeros(self.left_path_sigs.size())
        left_filler[self.left_path_sigs == 0] = 1
        right_filler = torch.zeros(self.right_path_sigs.size())
        if self.use_gpu:
            left_filler = left_filler.cuda()
            right_filler = right_filler.cuda()
        right_filler[self.right_path_sigs == 0] = 1

        left_path_probs = left_path_probs.add(left_filler)
        right_path_probs = right_path_probs.add(right_filler)

        probs = torch.cat((left_path_probs, right_path_probs), dim=1)
        probs = probs.prod(dim=1)
        actions = probs.mm(self.action_probs)

        if self.is_value:
            return actions  # Return logits
        # Else return probabilities
        return self.softmax(actions)

    def create_discrete_copy(self, *, preserve_actions: bool = True) -> Self:
        fuzzy_model = self
        new_weights = []
        new_comps = []

        weights = np.abs(fuzzy_model.layers.detach().numpy())
        most_used = np.argmax(weights, axis=1)
        for comp_ind, comparator_tensor in enumerate(fuzzy_model.comparators):
            comparator = comparator_tensor.item()
            divisor = abs(fuzzy_model.layers[comp_ind][most_used[comp_ind]].item())
            if divisor == 0:
                divisor = 1
            comparator /= divisor
            new_comps.append([comparator])
            max_ind = most_used[comp_ind]
            new_weight = np.zeros(len(fuzzy_model.layers[comp_ind].data))
            new_weight[max_ind] = fuzzy_model.layers[comp_ind][most_used[comp_ind]].item() / divisor
            new_weights.append(new_weight)

        new_input_dim = fuzzy_model.input_dim
        new_weights = np.array(new_weights)
        new_comps = np.array(new_comps)
        crispy_model = fuzzy_model.__class__(
            input_dim=new_input_dim,
            output_dim=fuzzy_model.output_dim,
            weights=new_weights,
            comparators=new_comps,
            leaves=fuzzy_model.leaf_init_information,
            alpha=99999.0,
            is_value=fuzzy_model.is_value,
            use_gpu=fuzzy_model.use_gpu,
        )

        # XXX: Are both needed should this be a hyperparameter?
        # For a ddt that preserves actions using softmaxes, use old action probs
        if preserve_actions:
            crispy_model.action_probs.data = fuzzy_model.action_probs.data
        else:
            # For a set of discrete ddt parameters (0, 1) leaves, use the one-hot method:
            max_inds = fuzzy_model.action_probs.data.argmax(dim=1)
            new_action_probs = torch.zeros_like(fuzzy_model.action_probs.data)
            new_action_probs[np.arange(len(new_action_probs)), max_inds] = 10
            crispy_model.action_probs.data = new_action_probs

        if fuzzy_model.use_gpu:
            crispy_model = crispy_model.cuda()

        return crispy_model
