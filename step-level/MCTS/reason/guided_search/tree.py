import json
import math
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
from distributed.utils import print_rank_0
from envs.base_env import CoTEnv
from loguru import logger
from openai import OpenAI
from reason.inference.skywork_score import skywork_score
from transformers import AutoTokenizer


class Node(object):
    """
    Overview:
        The node base class for tree_search.
    """

    def __init__(
        self, parent: "Node" = None, prior_p: float = 1.0, initial_value: float = 0.0
    ) -> None:
        self._parent = parent
        self._children = {}
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = prior_p
        self.prior_p_ori = prior_p

        self._initial_value = initial_value
        self._terminated = False

    def __lt__(self, other):
        return self._initial_value < other._initial_value

    @property
    def terminated(self):
        return self._terminated

    def set_as_terminate_node(self):
        self._terminated = True

    @property
    def value(self) -> float:
        """
        Overview:
            The value of the current node.
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
        """
        if self._visit_count == 0:
            # if not visited, return the initial value
            return self._initial_value
        return self._value_sum / self._visit_count

    def update(self, value: float) -> None:
        """
        Overview:
            Updata the current node information, such as visit_count and value_sum.
        Arguments:
            - value (:obj:`Int`): The value of the node.
        """
        self._visit_count += 1
        self._value_sum += value

    def update_recursive(self, leaf_value: float, mcts_mode: str) -> None:
        """
        Overview:
            Update node information recursively.
        Arguments:
            - leaf_value (:obj:`Int`): The value of the node.
        """
        if mcts_mode == "self_play_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(-leaf_value, mcts_mode)
        if mcts_mode == "play_with_bot_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(leaf_value, mcts_mode)

    def is_leaf(self) -> Dict:
        """
        Overview:
            Check if the current node is a leaf node or not.
        Returns:
            - output (:obj:`Dict`): Dict type children node.
        """
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): Whether it is the parent node.
        """
        return self._parent is None

    @property
    def parent(self) -> None:
        return self._parent

    @property
    def children(self) -> None:
        return self._children

    @property
    def visit_count(self) -> None:
        return self._visit_count

    def get_info(self):
        return {
            "visit_cnt": self.visit_count,
            "value": self.value,
            "prior_p": float(self.prior_p_ori),
            "initial_value": self._initial_value,
            "terminated": self.terminated,
        }

    def clear(self):
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = self.prior_p_ori

    def to_json(self):
        childrens = {}
        for name, child_node in self.children.items():
            childrens[name] = child_node.to_json()

        rets = {"children": childrens, "info": self.get_info()}
        return rets

    def __str__(self) -> str:
        if self.is_root():
            return "root"
        else:
            return "child: {} value: {:.3f}, prior: {:.3f}".format(
                self.last_action, self.value, self.prior_p
            )


class LanguageNode(Node):
    text_state: Optional[str] = None
    last_action: Optional[str] = None
    num_generated_token: Optional[int] = None

    def __init__(
        self,
        parent: Node = None,
        prior_p: float = 1.0,
        prm_value: Optional[float] = None,
        text_state: Optional[str] = None,
        last_action: Optional[str] = None,
        initial_value: float = 0.0,
        num_generated_token: Optional[int] = None,
    ) -> None:
        super().__init__(parent, prior_p, initial_value)
        self.text_state = text_state
        self.last_action = last_action
        self.prm_value = prm_value

        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False

    def get_path(self):
        ans = []
        node = self
        while not node.is_root():
            ans.append(node.last_action)
            node = node.parent
        return "\n".join(reversed(ans))

    def get_info(self):
        info_dict = super().get_info()
        if not self.is_root():
            info_dict["last_action"] = self.last_action
            info_dict["prm_value"] = self.prm_value
        else:
            info_dict["text_state"] = self.text_state
        return info_dict

    def __str__(self):
        if self.is_root():
            return "root: {}".format(self.text_state)
        else:
            return "action: {}, value: {:.3f}, prior: {:.3f}".format(
                self.last_action, self.value, self.prior_p
            )


def get_root(node: Node):
    while not node.is_root():
        node = node.parent
    return node


class SearchTree:
    """
    Overview:
        MCTS search process.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg

        self._num_simulations = self._cfg.get("num_simulations", 20)

        # UCB formula
        self._pb_c_base = self._cfg.get("pb_c_base", 19652)
        self._pb_c_init = self._cfg.get("pb_c_init", 1.25)

        # Root prior exploration noise.
        self._root_dirichlet_alpha = self._cfg.get(
            "root_dirichlet_alpha", 0.3
        )  # 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self._root_noise_weight = self._cfg.get("root_noise_weight", 0.25)

        self.root = None

        self.answers = set()
        self.wrong_answers = set()
        self.visited_paths = None

        self.no_terminal_reward = self._cfg.get("no_terminal_reward", True)
        self.mask_non_terminal_node_value = self._cfg.get(
            "mask_non_terminal_node_value", False
        )

        self._init_critic_value = self._cfg.get("init_critic_value", True)

        self._completion_tokens = 0

    @property
    def num_generated_token(self):
        return self._completion_tokens

    def clear_node(self, node):
        assert node is not None
        node.clear()
        for child in node.children.values():
            self.clear_node(child)

    def vanila_mcts(
        self,
        simulate_env: Type[CoTEnv],
        num_path: int,
        prm_tokenizer: AutoTokenizer,
        select_by_prior: bool = False,
    ) -> List[Dict]:
        api_call_completion_tokens = 0
        _, info = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += info["api_completion_token"]
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, prm_tokenizer)
            self.root = root

        traj_list = []

        #!  for here is split the for loop with select and rollout
        #!  so that arbitrary rollout function can be used here.

        for i_path in range(num_path):
            node = self.root
            env_copy = simulate_env.copy()
            done = False
            while not done:
                if node.visit_count > 0:
                    # if node is visited, select the child with the highest UCB score
                    action, node = self._select_child(node, env_copy)
                else:
                    # choose rollout policy
                    if select_by_prior:
                        # select with prior probability
                        action, node = self._select_by_prior(node, env_copy)
                    else:
                        # select with highest value, since visit_count = 0 in self.ucb
                        # will select node with highest value
                        action, node = self._select_child(node, env_copy)

                # sync terminated flag here
                env_copy._next_state_terminated = {}
                assert node.last_action == action
                env_copy._next_state_terminated[action] = node.terminated

                _, _, terminated, truncated, info = env_copy.step(
                    action, update_legal_action=node.is_leaf()
                )

                done = terminated or truncated

                if not done and node.is_leaf():
                    self._expand_leaf_node(node, env_copy, prm_tokenizer)

                # record api_tokens, if not expand, info["api_completion_token"] is 0
                api_call_completion_tokens += info["api_completion_token"]
            else:
                if node.visit_count > 0:
                    leaf_value = node.value
                else:
                    if self._init_critic_value:
                        leaf_value = node._initial_value
                    else:
                        # --------------- call vllm(prm) server -------------
                        openai_api_key = "EMPTY"
                        openai_api_base = "http://localhost:8081/v1"
                        client = OpenAI(
                            api_key=openai_api_key,
                            base_url=openai_api_base,
                        )
                        models = client.models.list()
                        skywork_prm = models.data[0].id
                        # ---------------------------------------------------
                        #! need to check further
                        leaf_value = skywork_score(
                            questions=[env_copy.question],
                            outputs=[env_copy.answer],
                            model=skywork_prm,
                            client=client,
                            prm_tokenizer=prm_tokenizer,
                        )[0][0]
                        # * use average score
                        leaf_value = np.mean(leaf_value)

            node.update_recursive(leaf_value, env_copy.mcts_mode)

            traj_data = {
                "path_idx": i_path,
                "text": env_copy.answer,
                "value": leaf_value,
                "api_completion_tokens": api_call_completion_tokens,
                "tree_completion_tokens": self._completion_tokens,
            }

            traj_list.append(traj_data)

            # reset api_call_completion_tokens
            api_call_completion_tokens = 0

        return traj_list

    def _select_child(
        self, node: LanguageNode, simulate_env: Type[CoTEnv]
    ) -> Tuple[Union[int, float], Node]:
        """
        Overview:
            Select the child with the highest UCB score.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        Returns:
            - action (:obj:`Int`): choose the action with the highest ucb score.
            - child (:obj:`Node`): the child node reached by executing the action with the highest ucb score.
        """

        action = None
        child = None
        best_score = -9999999

        for action_tmp, child_tmp in node.children.items():
            ucb_score = self._ucb_score(node, child_tmp)
            score = ucb_score
            if score > best_score:
                best_score = score
                action = action_tmp
                child = child_tmp

        if child is None:
            child = node  # child==None, node is leaf node in play_with_bot_mode.

        return action, child

    def _select_by_prior(self, node: Node, simulate_env):
        data_tmp = [
            (x_action, x_node.prior_p) for x_action, x_node in node.children.items()
        ]
        action_list, prior_list = list(zip(*data_tmp))
        chosen_action = np.random.choice(action_list, p=np.array(prior_list))
        chosen_node = node.children[chosen_action]

        return chosen_action, chosen_node

    def _expand_leaf_node(
        self,
        node: Node,
        simulate_env: Type[CoTEnv],
        prm_tokenizer: AutoTokenizer,
    ) -> float:
        # --------------- call vllm(prm) server -------------
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8081/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        models = client.models.list()
        skywork_prm = models.data[0].id
        # ---------------------------------------------------

        text_state = simulate_env.get_state()

        if not self._init_critic_value:
            #! need to check further
            leaf_value = skywork_score(
                questions=[simulate_env.question],
                outputs=[simulate_env.answer],
                model=skywork_prm,
                client=client,
                prm_tokenizer=prm_tokenizer,
            )[0][0]
            # * use average score
            leaf_value = np.mean(leaf_value)

        else:
            leaf_value = node._initial_value
            assert len(simulate_env.legal_actions) > 0
            #! need to check further
            prms = skywork_score(
                questions=[simulate_env.question],
                outputs=[
                    [
                        simulate_env.answer + x["action"]
                        for x in simulate_env.legal_actions
                    ]
                ],
                model=skywork_prm,
                client=client,
                prm_tokenizer=prm_tokenizer,
            )[0]

            child_values = []
            # PRM get last r as single reward
            for act, rs in zip(simulate_env.legal_actions, prms):
                if len(simulate_env.action_history) + 1 != len(rs):
                    logger.warning(
                        "PRM value length not match with action history. \
                            len(prm)={}, len(act_hist)={}, child_value={}".format(
                            len(prms),
                            len(simulate_env.action_history),
                            np.mean(rs),
                        )
                    )
                    # raise RuntimeError("Tokenizer problems")
                    # child_values.append(0.0)
                    child_values.append(np.mean(rs))

                elif len(rs) == 0:
                    logger.warning(
                        "Empty PRM value for: \nState: \n{} \naction: \n{}, will be set to 0.0".format(
                            text_state, act
                        )
                    )
                    child_values.append(0.0)
                else:
                    # * prm-last
                    # child_values.append(rs[-1])
                    # * prm-min
                    # child_values.append(min(rs))
                    # * prob-prm
                    # child_values.append(act['prob'])
                    # * prm-mean
                    child_values.append(np.mean(rs))

        assert len(node.children) == 0
        for i, action_dict in enumerate(simulate_env.legal_actions):
            action, prob = action_dict["action"], action_dict["prob"]

            if self._init_critic_value:
                child_value = child_values[i]
            else:
                # consider turn off this branch, i.e. always assume `self._init_critic=True`, since with LLM
                child_value = 0.0

            node.children[action] = LanguageNode(
                parent=node,
                prior_p=prob,
                # prm_value=prm_value,
                text_state=text_state,
                last_action=action,
                initial_value=child_value,
                num_generated_token=action_dict["num_token"],
            )
            # set terminal node here
            if simulate_env._next_state_terminated[action]:
                node.children[action].set_as_terminate_node()
        if len(node.children) == 0:
            print_rank_0(
                "Prune all current children at node {}".format(node.last_action)
            )

        # collect num tokens
        if not node.has_collected_token_num:
            self._completion_tokens += sum(
                c.num_generated_token for c in node.children.values()
            )
            node.has_collected_token_num = True
        else:
            raise RuntimeError("Token number has been collected again.")

        return leaf_value

    def _ucb_score(self, parent: Node, child: Node) -> float:
        """
        Overview:
            Compute UCB score. The score for a node is based on its value, plus an exploration bonus based on the prior.
        Arguments:
            - parent (:obj:`Class Node`): Current node.
            - child (:obj:`Class Node`): Current node's child.
        Returns:
            - score (:obj:`Bool`): The UCB score.
        """
        pb_c = (
            math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base)
            + self._pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior_p
        value_score = child.value
        return prior_score + value_score

    def reset_prior(self, node: Node) -> None:
        """
        Overview:
            Reset prior probability
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        for a in node.children.keys():
            node.children[a].prior_p = node.children[a].prior_p_ori

    def _add_exploration_noise(self, node: Node) -> None:
        """
        Overview:
            Add exploration noise.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        # Get a list of actions corresponding to the child nodes.
        actions = list(node.children.keys())
        # Create a list of alpha values for Dirichlet noise.
        alpha = [self._root_dirichlet_alpha] * len(actions)
        # Generate Dirichlet noise using the alpha values.
        noise = np.random.dirichlet(alpha)
        # Compute the weight of the exploration noise.
        frac = self._root_noise_weight
        # Update the prior probability of each child node with the exploration noise.
        for a, n in zip(actions, noise):
            node.children[a].prior_p = node.children[a].prior_p * (1 - frac) + n * frac

    @classmethod
    def from_json(cls, cfg: dict, json_path: str, reset_visit_info: bool):
        tree_json = json.load(open(json_path, "r"))

        def build_tree(tree_dict: dict) -> Node:
            node_info = tree_dict["info"]
            current_node = LanguageNode(
                text_state=node_info.get("text_state", None),
                last_action=node_info.get("last_action", None),
                prior_p=node_info["prior_p"],
                prm_value=node_info.get("prm_value", None),
                initial_value=node_info.get("initial_value", 0.0),
            )

            if not reset_visit_info:
                current_node._visit_count = node_info["visit_cnt"]
                current_node._value_sum = node_info["value"] * current_node.visit_count
            if node_info.get("terminated", False):
                current_node.set_as_terminate_node()

            for name, child_dict in tree_dict["children"].items():
                child_node = build_tree(child_dict)
                current_node._children[name] = child_node
                child_node._parent = current_node

            return current_node

        root_node = build_tree(tree_dict=tree_json)

        obj = cls(cfg)
        obj.root = root_node
        return obj

    def draw_tree(self):
        # Not tested yet
        root = self.root
        assert root, "Root node is None"

        def draw_node(node, depth):
            print("|" + "-" * depth + str(node))
            for child in node.children.values():
                draw_node(child, depth + 1)

        print("\n---------Expanded Tree---------")
        draw_node(self.root)
