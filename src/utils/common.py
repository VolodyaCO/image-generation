# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import time
from typing import Optional

import networkx as nx
import numpy as np
from dimod import Sampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite, ParallelEmbeddingComposite


def greedy_get_subgraph(
    n_nodes: int,
    random_seed: Optional[int],
    graph: Optional[nx.Graph] = None,
    qpu: Optional[str] = None,
) -> tuple[nx.Graph, dict]:
    """TODO

    Args:
        n_nodes: TODO
        random_seed: TODO
        graph: TODO
        qpu: The selected QPU.

    Returns:
        nx.Graph: TODO
        dict: TODO
    """
    generator = random.Random(random_seed)
    if graph is None:
        qpu = DWaveSampler(solver=qpu)
        graph = qpu.to_networkx_graph()

    assert isinstance(graph, nx.Graph)

    selected_nodes = [generator.choice(list(graph.nodes()))]
    MAXIMUM_CONNECTIVITY = max([len(list(graph.neighbors(n))) for n in graph.nodes()])

    while len(selected_nodes) < n_nodes:
        maximum_connectivity = 0
        target_maximum_connectivity = min(MAXIMUM_CONNECTIVITY, len(selected_nodes))
        target_node = None
        found_optimal_target_node = False
        generator.shuffle(selected_nodes)

        for node in selected_nodes:
            neighbours = list(graph.neighbors(node))
            generator.shuffle(neighbours)

            for neighbour in neighbours:
                if neighbour not in selected_nodes:
                    # If we were to add neighbour to the selected nodes, how many of the
                    # selected nodes would it be connected to?
                    connectivity = len(set(graph.neighbors(neighbour)).intersection(selected_nodes))
                    if connectivity >= target_maximum_connectivity:
                        found_optimal_target_node = True
                        target_node = neighbour
                        selected_nodes.append(target_node)
                        break
                    elif connectivity > maximum_connectivity:
                        maximum_connectivity = connectivity
                        target_node = neighbour

            if found_optimal_target_node:
                break

        if found_optimal_target_node:
            continue

        selected_nodes.append(target_node)

    subgraph = graph.subgraph(selected_nodes)
    mapping = {
        physical: logical for (physical, logical) in zip(subgraph.nodes(), range(len(subgraph)))
    }

    return nx.relabel_nodes(subgraph, mapping), mapping


class InsistentSampler:
    def __init__(
        self,
        sampler: Sampler,
        max_attempts: int = 10,
        progressive_sleep: bool = True,
        initial_sleep: float = 1.0,
    ):
        self.sampler = sampler
        self.max_attempts = max_attempts
        self.progressive_sleep = progressive_sleep
        self.initial_sleep = initial_sleep

    def sample(self, *args, **kwargs):
        import traceback

        attempt = 0
        sleep_time = self.initial_sleep
        while attempt < self.max_attempts:
            try:
                return self.sampler.sample(*args, **kwargs)
            except Exception as e:
                print(f"Sampler attempt {attempt + 1} failed with exception: {e}")
                traceback.print_exc()
                attempt += 1
                if self.progressive_sleep:
                    time.sleep(sleep_time)
                    sleep_time *= 2  # Exponential backoff
                else:
                    time.sleep(self.initial_sleep)

    def sample_ising(self, *args, **kwargs):
        return self.sampler.sample_ising(*args, **kwargs)


def get_sampler_and_sampler_kwargs(
    num_reads: int,
    annealing_time: int | float,
    num_sweeps: int,
    n_latents: int,
    random_seed: int | None,
    use_qpu: bool,
    parallel_embedding: bool = True,
    solver_name: str = "Advantage2_prototype2.6",
) -> tuple[InsistentSampler, nx.Graph, list | None, list | None, dict]:
    qpu = DWaveSampler(solver=solver_name)
    h_range, j_range = qpu.properties["h_range"], qpu.properties["j_range"]
    qpu_graph = qpu.to_networkx_graph()
    graph, mapping = greedy_get_subgraph(
        n_nodes=n_latents, random_seed=random_seed, graph=qpu_graph
    )
    if use_qpu:
        if parallel_embedding:
            from math import ceil

            from minorminer.utils.parallel_embeddings import find_multiple_embeddings

            embeddings = find_multiple_embeddings(
                graph,
                qpu_graph,
                max_num_emb=qpu_graph.number_of_nodes() // graph.number_of_nodes(),
                one_to_iterable=False,
                timeout=60 * 5,  # 5 minutes
            )
            print(f"Found {len(embeddings)} embeddings for the graph.")
            taken_nodes = {}
            for i, emb in enumerate(embeddings):
                for v in emb.values():
                    if v in taken_nodes:
                        raise ValueError(
                            f"QPU node {v} is already taken by embedding {taken_nodes[v]}:\n"
                            f"{embeddings[taken_nodes[v]]}\nCurrent embedding:\n{emb}"
                        )
                    taken_nodes[v] = i
            sampler = ParallelEmbeddingComposite(
                qpu, embeddings=[{l_: [p] for l_, p in emb.items()} for emb in embeddings]
            )
            n_embeddings = len(embeddings)
            num_reads = ceil(num_reads / n_embeddings)
        else:
            sampler = FixedEmbeddingComposite(qpu, {l_: [p] for p, l_ in mapping.items()})
        sampler_kwargs = dict(
            num_reads=num_reads,
            # Set `answer_mode` to "raw" so no samples are aggregated
            answer_mode="raw",
            # Set `auto_scale`` to `False` so the sampler sample from the intended
            # distribution
            auto_scale=False,
            annealing_time=annealing_time,
        )
    else:
        from dwave.samplers import SimulatedAnnealingSampler

        sampler = SimulatedAnnealingSampler()
        sampler_kwargs = dict(
            num_reads=num_reads,
            beta_range=[1, 1],
            proposal_acceptance_criterion="Gibbs",
            randomize_order=True,
            num_sweeps=num_sweeps,
        )
    return InsistentSampler(sampler), graph, h_range, j_range, sampler_kwargs


def get_coeff(step, n_rampup, min_coeff, max_coeff, delay=0, cutoff=np.inf):
    if step < delay:
        return 0.0
    elif step < n_rampup + delay:
        return min_coeff + (max_coeff - min_coeff) * ((step - delay) / n_rampup)
    elif step > n_rampup + delay and step < cutoff:
        return max_coeff
    else:
        return 0.0


def get_exp_coeff(step, n_rampup, min_coeff, max_coeff, delay=0, cutoff=np.inf):
    if step < delay:
        return 0.0
    elif step < n_rampup + delay:
        return min_coeff * (max_coeff / min_coeff) ** ((step - delay) / n_rampup)
    elif step > n_rampup + delay and step < cutoff:
        return max_coeff
    else:
        return 0.0
