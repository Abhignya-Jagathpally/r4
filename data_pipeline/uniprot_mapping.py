"""UniProt ID mapping with rate limiting (M12)."""

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

RATE_LIMIT_SECONDS = 0.1  # M12: 100ms delay between API calls


class UniProtMapper:
    """Map protein identifiers via UniProt API with rate limiting."""

    def __init__(self, base_url: str = "https://rest.uniprot.org"):
        self.base_url = base_url
        self._last_request_time = 0.0
        self._cache: Dict[str, str] = {}

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls (M12)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_SECONDS:
            time.sleep(RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

    def map_ids(
        self, ids: List[str], from_db: str = "UniProtKB_AC-ID",
        to_db: str = "Gene_Name",
    ) -> Dict[str, str]:
        """Map protein IDs to gene names with rate limiting."""
        # Check cache first
        uncached = [i for i in ids if i not in self._cache]
        if not uncached:
            return {i: self._cache[i] for i in ids}

        try:
            import requests

            # Batch in chunks of 100
            for start in range(0, len(uncached), 100):
                batch = uncached[start:start + 100]
                self._rate_limit()  # M12

                resp = requests.post(
                    f"{self.base_url}/idmapping/run",
                    data={"from": from_db, "to": to_db, "ids": ",".join(batch)},
                    timeout=30,
                )
                resp.raise_for_status()
                job_id = resp.json().get("jobId")

                if job_id:
                    # Poll for results
                    for _ in range(30):
                        self._rate_limit()
                        status = requests.get(
                            f"{self.base_url}/idmapping/status/{job_id}", timeout=10,
                        )
                        result = status.json()
                        if "results" in result:
                            for r in result["results"]:
                                self._cache[r["from"]] = r.get("to", {}).get("primaryAccession", r["from"])
                            break
                        time.sleep(1)

                logger.debug(f"Mapped batch of {len(batch)} IDs")

        except Exception as e:
            logger.warning(f"UniProt mapping failed: {e}. Using passthrough.")
            for i in uncached:
                self._cache[i] = i

        return {i: self._cache.get(i, i) for i in ids}
