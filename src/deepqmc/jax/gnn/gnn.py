from functools import partial

import haiku as hk
import jax.numpy as jnp

from .graph import Graph, GraphUpdate, MolecularGraphEdgeBuilder


class MessagePassingLayer(hk.Module):
    r"""
    Base class for all message passing layers.

    Args:
        ilayer (int): the index of the current layer in the list of all layers
        shared (dict): attribute names and values which are shared between the
            layers and the :class:`GraphNeuralNetwork` instance.
    """

    def __init__(self, ilayer, shared):
        super().__init__()
        self.ilayer = ilayer
        for k, v in shared.items():
            setattr(self, k, v)
        self.update_graph = GraphUpdate(
            update_nodes_fn=self.get_update_nodes_fn(),
            update_edges_fn=self.get_update_edges_fn(),
            aggregate_edges_for_nodes_fn=self.get_aggregate_edges_for_nodes_fn(),
        )

    def __call__(self, graph):
        r"""
        Args:
            graph (:class:`Graph`)

        Returns:
            :class:`Graph`: updated graph
        """
        return self.update_graph(graph)

    def get_update_edges_fn(self):
        r"""
        Creates a function that updates the graph edges.

        Returns:
            :data:`Callable[GraphNodes,GraphEdges]`: a function
            that outputs the updated edges as a :class:`GraphEdges` instance.
        """
        raise NotImplementedError

    def get_update_nodes_fn(self):
        r"""
        Creates a function that updates the graph nodes.

        Returns:
            :data:`Callable[GraphNodes,*]`: a function
            that outputs the updated nodes as a :class:`GraphNodes` instance.
            The second argument will be the aggregated graph edges.
        """
        raise NotImplementedError

    def get_aggregate_edges_for_nodes_fn(self):
        r"""
        Creates a function that aggregates the graph edges.

        Returns:
            :data:`Callable[GraphNodes,GraphEdges]`: a function
            that outputs the aggregated edges.
        """
        raise NotImplementedError


class GraphNeuralNetwork(hk.Module):
    r"""
    Base class for all graph neural networks on molecules.
    """

    def __init__(
        self,
        mol,
        embedding_dim,
        cutoff,
        n_interactions,
        layer_kwargs=None,
        ghost_coords=None,
        share_with_layers=None,
    ):
        r"""
        Args:
            mol (:class:`deepqmc.jax.Molecule`): the molecule on which the GNN
                is defined
            embedding_dim (int): the size of the electron embeddings to be returned.
            cutoff (float): cutoff distance above which graph edges are discarded.
            n_interactions (int): the number of interaction layers in the GNN.
            layer_kwargs (dict): optional, kwargs to be passed to the layers.
            ghost_coords (float, [N, 3]): optional, coordinates of ghost atoms.
                These will be included as nuclear nodes in the graph. Useful for
                breaking undesired spatial symmetries.
            share_with_layers (dict): optional, attribute names and values to share
                with the interaction layers.
        """
        super().__init__()
        n_nuc, n_up, n_down = mol.n_particles
        self.coords = mol.coords
        self.cutoff = cutoff
        if ghost_coords is not None:
            self.coords = jnp.concatenate([self.coords, jnp.asarray(ghost_coords)])
            n_nuc = len(self.coords)
        share_with_layers = share_with_layers or {}
        share_with_layers.setdefault('embedding_dim', embedding_dim)
        share_with_layers.setdefault('n_nuc', n_nuc)
        share_with_layers.setdefault('n_up', n_up)
        share_with_layers.setdefault('n_down', n_down)
        for k, v in share_with_layers.items():
            setattr(self, k, v)
        share_with_layers.setdefault('edge_types', self.edge_types)
        self.layers = [
            self.layer_factory(
                i,
                share_with_layers,
                **(layer_kwargs or {}),
            )
            for i in range(n_interactions)
        ]
        self.n_up, self.n_down = n_up, n_down

    def init_state(self, shape, dtype):
        r"""
        Initializes the haiku state that communicates the sizes of edge lists.
        """
        raise NotImplementedError

    def initial_embeddings(self):
        r"""
        Returns the initial embeddings as a :class:`GraphNodes` instance.
        """
        raise NotImplementedError

    def edge_feature_callback(
        self, edge_type, pos_sender, pos_receiver, sender_idx, receiver_idx
    ):
        r"""
        Defines the :func:`feature_callback` to be called on the edges
        of different types.

        Args:
            edge_typ (str): name of the edge_type for which features are calculated.
            pos_sender (float, (:math:`N_\text{nodes}`, 3)): coordinates of the
                sender nodes.
            pos_receiver (float, (:math:`M_\text{nodes}`, 3]): coordinates of the
                receiver nodes.
            sender_idx (int, (:data:`occupancy_limit`)): indeces of the sender nodes.
            receiver_idx (int, (:data:`occupancy_limit`)): indeces of the receiver
                nodes.

        Returns:
            the features for the given edges
        """
        raise NotImplementedError

    @classmethod
    @property
    def edge_types(cls):
        r"""
        A tuple containing the names of the edge types used in the GNN.
        See :class:`~deepqmc.jax.gnn.graph.MolecularGraphEdgeBuilder` for possible
        edge types.
        """
        raise NotImplementedError

    def edge_factory(self, r, occupancies):
        edge_factory = MolecularGraphEdgeBuilder(
            self.n_nuc,
            self.n_up,
            self.n_down,
            self.coords,
            self.edge_types,
            kwargs_by_edge_type={
                typ: {
                    'cutoff': self.cutoff[typ],
                    'feature_callback': partial(self.edge_feature_callback, typ),
                }
                for typ in self.edge_types
            },
        )
        return edge_factory(r, occupancies)

    @classmethod
    @property
    def layer_factory(cls):
        r"""
        The class of interaction layer to be used.
        """
        return MessagePassingLayer

    def __call__(self, r):
        r"""
        Args:
            r (float, (:math:`N_\text{elec}`, 3)): electron coordinates.

        Returns:
            float, (:math:`N_\text{elec}`, :data:`embedding_dim`):
            the final embeddings of the electrons.
        """
        occupancies = hk.get_state(
            'occupancies',
            shape=1,
            dtype=jnp.int32,
            init=self.init_state,
        )
        graph_edges, occupancies = self.edge_factory(r, occupancies)
        hk.set_state('occupancies', occupancies)
        graph_nodes = self.initial_embeddings()
        graph = Graph(
            graph_nodes,
            graph_edges,
        )

        for layer in self.layers:
            graph = layer(graph)

        return graph.nodes.electrons
