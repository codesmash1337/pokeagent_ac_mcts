"""
Neural Policy Server
Loads Metamon model and serves policy inference requests via HTTP
"""

import os
import logging
import time
from typing import Dict, List, Optional
from flask import Flask, request, jsonify
import numpy as np
import torch

from .state_translator import StateTranslator

logger = logging.getLogger(__name__)


class NeuralPolicyServer:
    """
    HTTP server that provides neural policy inference.

    Loads a pretrained Metamon model and serves policy queries from MCTS.
    Includes caching and batching for performance.
    """

    def __init__(
        self,
        model_name: str = "Minikazam",
        device: str = "cpu",
        cache_size: int = 1000,
        port: int = 5000
    ):
        """
        Initialize the neural policy server.

        Args:
            model_name: Name of Metamon model to load (e.g., "Minikazam", "Abra")
            device: Device to run inference on ("cpu" or "cuda")
            cache_size: Maximum number of states to cache
            port: HTTP server port
        """
        self.model_name = model_name
        self.device = device
        self.cache_size = cache_size
        self.port = port

        # Initialize state translator
        self.translator = StateTranslator()

        # Policy cache (state_str -> policy distribution)
        self.policy_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Statistics
        self.total_queries = 0
        self.total_inference_time = 0.0

        # Load Metamon model
        self._load_model()

        # Create Flask app
        self.app = Flask(__name__)
        self._setup_routes()

        logger.info(f"Initialized NeuralPolicyServer with model={model_name}, device={device}")

    def _load_model(self):
        """Load the pretrained Metamon model."""
        try:
            # Import Metamon dependencies
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../metamon"))

            from metamon.rl.pretrained import get_pretrained_model

            logger.info(f"Loading Metamon model: {self.model_name}")

            # Get the pretrained model class
            pretrained_model = get_pretrained_model(self.model_name)

            # Initialize agent (this loads the model weights)
            self.agent = pretrained_model.initialize_agent(checkpoint=None, log=False)

            # Set to evaluation mode
            self.agent.eval()

            # Store observation/action spaces for reference
            self.observation_space = pretrained_model.observation_space
            self.action_space = pretrained_model.action_space

            logger.info(f"Successfully loaded {self.model_name} model")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Create a dummy model for testing
            logger.warning("Using dummy model for testing")
            self.agent = None
            self.observation_space = None
            self.action_space = None

    def _setup_routes(self):
        """Setup Flask HTTP routes."""

        @self.app.route("/health", methods=["GET"])
        def health():
            """Health check endpoint."""
            return jsonify({
                "status": "healthy",
                "model": self.model_name,
                "device": self.device,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_size": len(self.policy_cache),
                "total_queries": self.total_queries
            })

        @self.app.route("/policy", methods=["POST"])
        def get_policy():
            """
            Get policy for a single state.

            Request body:
            {
                "state": "<state_string>",
                "legal_actions": [0, 1, 2, ...]  # optional
            }

            Response:
            {
                "policy": [0.1, 0.2, ...],  # 13-dim probability distribution
                "cache_hit": true/false,
                "inference_time_ms": 15.3
            }
            """
            start_time = time.time()

            data = request.json
            state_str = data.get("state")
            legal_actions = data.get("legal_actions")

            if state_str is None:
                return jsonify({"error": "Missing 'state' in request"}), 400

            # Check cache
            cache_key = state_str
            if cache_key in self.policy_cache:
                self.cache_hits += 1
                policy = self.policy_cache[cache_key]
                cache_hit = True
            else:
                self.cache_misses += 1
                # Perform inference
                policy = self._infer_policy(state_str, legal_actions)

                # Update cache
                if len(self.policy_cache) < self.cache_size:
                    self.policy_cache[cache_key] = policy
                cache_hit = False

            self.total_queries += 1
            inference_time = (time.time() - start_time) * 1000

            return jsonify({
                "policy": policy.tolist(),
                "cache_hit": cache_hit,
                "inference_time_ms": inference_time
            })

        @self.app.route("/policy/batch", methods=["POST"])
        def get_policy_batch():
            """
            Get policies for multiple states (batched inference).

            Request body:
            {
                "states": ["<state1>", "<state2>", ...],
                "legal_actions": [[...], [...], ...]  # optional
            }

            Response:
            {
                "policies": [[...], [...], ...],
                "cache_hits": [true, false, ...],
                "total_inference_time_ms": 42.1
            }
            """
            start_time = time.time()

            data = request.json
            states = data.get("states", [])
            legal_actions_list = data.get("legal_actions", [None] * len(states))

            if not states:
                return jsonify({"error": "Missing 'states' in request"}), 400

            policies = []
            cache_hits_list = []

            # Check cache for each state
            uncached_indices = []
            uncached_states = []
            uncached_legal_actions = []

            for i, (state_str, legal_actions) in enumerate(zip(states, legal_actions_list)):
                if state_str in self.policy_cache:
                    self.cache_hits += 1
                    policies.append(self.policy_cache[state_str])
                    cache_hits_list.append(True)
                else:
                    self.cache_misses += 1
                    policies.append(None)  # Placeholder
                    cache_hits_list.append(False)
                    uncached_indices.append(i)
                    uncached_states.append(state_str)
                    uncached_legal_actions.append(legal_actions)

            # Batch inference for uncached states
            if uncached_states:
                batch_policies = self._infer_policy_batch(uncached_states, uncached_legal_actions)

                # Fill in results and update cache
                for idx, policy in zip(uncached_indices, batch_policies):
                    policies[idx] = policy
                    if len(self.policy_cache) < self.cache_size:
                        self.policy_cache[states[idx]] = policy

            self.total_queries += len(states)
            total_time = (time.time() - start_time) * 1000

            return jsonify({
                "policies": [p.tolist() for p in policies],
                "cache_hits": cache_hits_list,
                "total_inference_time_ms": total_time
            })

        @self.app.route("/clear_cache", methods=["POST"])
        def clear_cache():
            """Clear the policy cache."""
            self.policy_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            return jsonify({"status": "cache cleared"})

    def _infer_policy(self, state_str: str, legal_actions: Optional[List[int]] = None) -> np.ndarray:
        """
        Perform neural policy inference for a single state.

        Args:
            state_str: Serialized state string
            legal_actions: Legal action indices

        Returns:
            Policy distribution (13-dim probability vector)
        """
        if self.agent is None:
            # Dummy policy for testing (uniform random over legal actions)
            policy = np.ones(13, dtype=np.float32) / 13.0
            if legal_actions is not None:
                mask = np.zeros(13, dtype=np.float32)
                mask[legal_actions] = 1.0
                policy = policy * mask
                policy = policy / policy.sum()
            return policy

        # Translate state to observation
        obs = self.translator.translate(state_str, legal_actions)

        # Convert to tensors
        # Note: Actual implementation would match Metamon's input format
        # This is a simplified version

        with torch.no_grad():
            # Get policy from agent
            # Note: Actual agent interface would be different
            # This is placeholder logic
            policy_logits = torch.randn(13)  # Placeholder

            # Apply softmax
            policy = torch.softmax(policy_logits, dim=0).cpu().numpy()

            # Apply legal action mask
            if legal_actions is not None:
                mask = np.zeros(13, dtype=np.float32)
                mask[legal_actions] = 1.0
                policy = policy * mask
                policy = policy / (policy.sum() + 1e-10)

        return policy

    def _infer_policy_batch(
        self,
        states: List[str],
        legal_actions_list: List[Optional[List[int]]]
    ) -> List[np.ndarray]:
        """
        Perform batched neural policy inference.

        Args:
            states: List of serialized state strings
            legal_actions_list: List of legal action lists

        Returns:
            List of policy distributions
        """
        if self.agent is None:
            # Dummy policies for testing
            return [self._infer_policy(s, la) for s, la in zip(states, legal_actions_list)]

        # Batch translate states
        batch_obs = self.translator.batch_translate(states, legal_actions_list)

        # Batch inference
        with torch.no_grad():
            # Note: Actual implementation would use agent's batched inference
            # This is placeholder logic
            batch_size = len(states)
            policy_logits = torch.randn(batch_size, 13)  # Placeholder

            # Apply softmax
            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()

            # Apply legal action masks
            for i, legal_actions in enumerate(legal_actions_list):
                if legal_actions is not None:
                    mask = np.zeros(13, dtype=np.float32)
                    mask[legal_actions] = 1.0
                    policies[i] = policies[i] * mask
                    policies[i] = policies[i] / (policies[i].sum() + 1e-10)

        return [policies[i] for i in range(batch_size)]

    def start_server(self, host: str = "127.0.0.1"):
        """
        Start the HTTP server.

        Args:
            host: Host address to bind to
        """
        logger.info(f"Starting NeuralPolicyServer on {host}:{self.port}")
        self.app.run(host=host, port=self.port, debug=False, threaded=True)

    def shutdown(self):
        """Shutdown the server and cleanup resources."""
        logger.info("Shutting down NeuralPolicyServer")
        self.policy_cache.clear()
        if self.agent is not None:
            del self.agent
        torch.cuda.empty_cache()


def main():
    """Main entry point for running the server standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="Neural Policy Server for MCTS")
    parser.add_argument("--model", default="Minikazam", help="Metamon model to load")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for inference")
    parser.add_argument("--port", type=int, default=5000, help="HTTP server port")
    parser.add_argument("--cache-size", type=int, default=1000, help="Policy cache size")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Create and start server
    server = NeuralPolicyServer(
        model_name=args.model,
        device=args.device,
        cache_size=args.cache_size,
        port=args.port
    )

    try:
        server.start_server(host=args.host)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
