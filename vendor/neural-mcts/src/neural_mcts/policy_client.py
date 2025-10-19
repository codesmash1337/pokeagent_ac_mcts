"""
Policy Client
Client for querying the Neural Policy Server from MCTS (Python side of foul-play)
"""

import logging
import requests
from typing import List, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)


class PolicyClient:
    """
    Client for querying neural policy from the policy server.

    Used by the foul-play integration layer to get policy priors from Metamon.
    """

    def __init__(self, server_url: str = "http://127.0.0.1:5000", timeout: float = 1.0):
        """
        Initialize the policy client.

        Args:
            server_url: URL of the neural policy server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

        # Statistics
        self.total_requests = 0
        self.total_errors = 0
        self.total_cache_hits = 0

        # Check server health
        self._check_health()

        logger.info(f"Initialized PolicyClient pointing to {self.server_url}")

    def _check_health(self) -> bool:
        """
        Check if the policy server is healthy.

        Returns:
            True if server is responding, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=self.timeout)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"Policy server is healthy: {health_data}")
                return True
            else:
                logger.warning(f"Policy server returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to policy server: {e}")
            return False

    def get_policy(
        self,
        state: str,
        legal_actions: Optional[List[int]] = None
    ) -> Optional[np.ndarray]:
        """
        Get policy distribution for a single state.

        Args:
            state: Serialized poke-engine state string
            legal_actions: List of legal action indices

        Returns:
            Policy distribution (13-dim numpy array), or None if request fails
        """
        try:
            payload = {"state": state}
            if legal_actions is not None:
                payload["legal_actions"] = legal_actions

            response = requests.post(
                f"{self.server_url}/policy",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                policy = np.array(data["policy"], dtype=np.float32)

                # Update statistics
                self.total_requests += 1
                if data.get("cache_hit", False):
                    self.total_cache_hits += 1

                logger.debug(f"Got policy: cache_hit={data.get('cache_hit')}, "
                           f"time={data.get('inference_time_ms', 0):.1f}ms")

                return policy
            else:
                logger.error(f"Policy server returned error: {response.status_code} - {response.text}")
                self.total_errors += 1
                return None

        except Exception as e:
            logger.error(f"Failed to get policy: {e}")
            self.total_errors += 1
            return None

    def get_policy_batch(
        self,
        states: List[str],
        legal_actions_list: Optional[List[Optional[List[int]]]] = None
    ) -> Optional[List[np.ndarray]]:
        """
        Get policy distributions for multiple states (batched).

        Args:
            states: List of serialized state strings
            legal_actions_list: List of legal action lists (one per state)

        Returns:
            List of policy distributions, or None if request fails
        """
        try:
            payload = {"states": states}
            if legal_actions_list is not None:
                payload["legal_actions"] = legal_actions_list

            response = requests.post(
                f"{self.server_url}/policy/batch",
                json=payload,
                timeout=self.timeout * 2  # Longer timeout for batch
            )

            if response.status_code == 200:
                data = response.json()
                policies = [np.array(p, dtype=np.float32) for p in data["policies"]]

                # Update statistics
                self.total_requests += len(states)
                self.total_cache_hits += sum(data.get("cache_hits", []))

                logger.debug(f"Got {len(policies)} policies: "
                           f"cache_hits={sum(data.get('cache_hits', []))}, "
                           f"time={data.get('total_inference_time_ms', 0):.1f}ms")

                return policies
            else:
                logger.error(f"Policy server returned error: {response.status_code}")
                self.total_errors += 1
                return None

        except Exception as e:
            logger.error(f"Failed to get batch policies: {e}")
            self.total_errors += 1
            return None

    def clear_cache(self) -> bool:
        """
        Clear the server's policy cache.

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.server_url}/clear_cache",
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def get_stats(self) -> Dict:
        """
        Get client statistics.

        Returns:
            Dictionary with statistics
        """
        cache_hit_rate = 0.0
        if self.total_requests > 0:
            cache_hit_rate = self.total_cache_hits / self.total_requests

        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "total_cache_hits": self.total_cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "error_rate": self.total_errors / max(self.total_requests, 1)
        }
