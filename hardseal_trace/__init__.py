# Copyright 2026 Hardseal LLC
# SPDX-License-Identifier: Apache-2.0
"""hardseal-trace v0.2.0 — Cryptographically verifiable Sealed Reasoning Traces.

v0.2 adds PODS (Prediction-Outcome Divergence Score) per DeepSeek panel formalism:
- Single number in [0,1], 1.0 = perfect, 0.0 = maximally wrong
- Equal-weight Field Agreement Score over divergence_record
- Closed-form expected value for random predictor (cd82: 0.125)
- Categorical / numeric / set / status_only field types

Per HARDSEAL_ARC_JEPA v1.1 §2.1 + v1.3 contest-first:
"We seal structured intent + action + outcome — NOT raw latent vectors."
"""
from __future__ import annotations
import hashlib, hmac, json, os, time, uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

__version__ = "0.2.0"


@dataclass
class StructuredIntent:
    claim_text: str
    predicted_outcome: dict
    confidence_class: str  # high|medium|low|exploratory
    prediction_horizon: str


@dataclass
class SealedTrace:
    schema_version: str
    trace_id: str
    session_id: str
    game_id: str
    action_counter: int
    timestamp_utc: float
    input_state_hash: str
    structured_intent: dict
    candidate_action: dict
    pre_action_payload_hash: str
    pre_action_hmac: str
    observed_outcome: Optional[dict] = None
    divergence_record: Optional[dict] = None
    divergence_score: Optional[float] = None  # PODS — added v0.2
    post_action_payload_hash: Optional[str] = None
    post_action_hmac: Optional[str] = None
    prev_post_hash: Optional[str] = None
    agent_name: str = ""
    agent_version: str = ""
    aux_evidence_refs: list = field(default_factory=list)


def _canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hmac_sha256_hex(key: bytes, msg: bytes) -> str:
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def _resolve_key(kp) -> bytes:
    if callable(kp):
        return _resolve_key(kp())
    if isinstance(kp, bytes):
        return kp
    if isinstance(kp, str):
        env = os.environ.get(kp)
        return env.encode("utf-8") if env else kp.encode("utf-8")
    raise TypeError(f"cannot resolve key from {type(kp)!r}")


def _diff(predicted, observed):
    """Structured divergence record — categorical/numeric/set diff per field.
    v0.2 adds explicit `type` field for PODS scoring downstream."""
    keys = set(predicted) | set(observed)
    out = {}
    for k in sorted(keys):
        if k not in observed:
            out[k] = {"type": "status_only", "status": "missing", "predicted": predicted[k]}
        elif k not in predicted:
            out[k] = {"type": "status_only", "status": "extra", "observed": observed[k]}
        elif predicted[k] == observed[k]:
            out[k] = {"type": "categorical", "status": "match"}
        else:
            p, o = predicted[k], observed[k]
            if isinstance(p, (int, float)) and isinstance(o, (int, float)):
                rng = max(abs(p), abs(o), 1.0)
                out[k] = {"type": "numeric", "status": "mismatch",
                          "predicted": p, "observed": o,
                          "delta": o - p, "range": rng}
            elif isinstance(p, (set, list)) and isinstance(o, (set, list)):
                ps, os_ = set(p), set(o)
                out[k] = {"type": "set", "status": "mismatch",
                          "predicted": sorted(ps), "observed": sorted(os_),
                          "only_predicted": sorted(ps - os_),
                          "only_observed": sorted(os_ - ps)}
            else:
                out[k] = {"type": "categorical", "status": "mismatch",
                          "predicted": p, "observed": o}
    return out


def compute_pods(divergence_record: dict) -> float:
    """PODS — Prediction-Outcome Divergence Score in [0, 1] (DeepSeek panel v0.2).

    Equal-weight Field Agreement Score:
    - categorical: 1.0 if status=='match' else 0.0
    - numeric: max(0, 1 - |delta|/range)
    - set: Jaccard similarity
    - status_only (missing/extra): 0.0
    Returns mean across all fields. Empty record → 1.0.

    Random predictor on cd82 (5x5 grid, 8 colors, all categorical): E[PODS] = 1/8 = 0.125.
    """
    scores = []
    for _, entry in divergence_record.items():
        ftype = entry.get("type", "categorical")
        if ftype == "categorical":
            scores.append(1.0 if entry.get("status") == "match" else 0.0)
        elif ftype == "numeric":
            delta = abs(entry.get("delta", 0))
            rng = entry.get("range", 1.0) or 1.0
            scores.append(max(0.0, 1.0 - delta / rng))
        elif ftype == "set":
            pred = set(entry.get("predicted", []))
            obs = set(entry.get("observed", []))
            if not pred and not obs:
                scores.append(1.0)
            else:
                scores.append(len(pred & obs) / len(pred | obs))
        elif ftype == "status_only":
            scores.append(0.0)
        else:
            scores.append(0.0)
    if not scores:
        return 1.0
    return sum(scores) / len(scores)


class SealedReasoningTracer:
    SCHEMA_VERSION = "hardseal-trace/0.2.0"

    def __init__(self, session_id, game_id, key_provider, agent_name="", agent_version=""):
        self.session_id = session_id
        self.game_id = game_id
        self._key = _resolve_key(key_provider)
        self.agent_name = agent_name
        self.agent_version = agent_version
        self._traces = {}
        self._chain_order = []
        self._action_counter = 0

    def seal_pre_action(self, input_state, structured_intent, candidate_action, aux_evidence_refs=None):
        trace_id = uuid.uuid4().hex
        input_state_hash = sha256_hex(_canonical_json(input_state))
        intent_dict = asdict(structured_intent)
        prev_hash = (self._chain_order
                     and self._traces[self._chain_order[-1]].post_action_payload_hash) or None
        pre_payload = {
            "schema_version": self.SCHEMA_VERSION, "trace_id": trace_id,
            "session_id": self.session_id, "game_id": self.game_id,
            "action_counter": self._action_counter,
            "input_state_hash": input_state_hash,
            "structured_intent": intent_dict,
            "candidate_action": candidate_action,
            "prev_post_hash": prev_hash,
        }
        pph = sha256_hex(_canonical_json(pre_payload))
        ph = hmac_sha256_hex(self._key, pph.encode("utf-8"))
        tr = SealedTrace(
            schema_version=self.SCHEMA_VERSION, trace_id=trace_id,
            session_id=self.session_id, game_id=self.game_id,
            action_counter=self._action_counter, timestamp_utc=time.time(),
            input_state_hash=input_state_hash, structured_intent=intent_dict,
            candidate_action=candidate_action, pre_action_payload_hash=pph,
            pre_action_hmac=ph, prev_post_hash=prev_hash,
            agent_name=self.agent_name, agent_version=self.agent_version,
            aux_evidence_refs=aux_evidence_refs or [],
        )
        self._traces[trace_id] = tr
        self._chain_order.append(trace_id)
        self._action_counter += 1
        return tr

    def seal_post_action(self, trace_id, observed_outcome):
        tr = self._traces[trace_id]
        if tr.post_action_payload_hash is not None:
            raise RuntimeError(f"trace {trace_id} already post-sealed")
        predicted = tr.structured_intent.get("predicted_outcome", {})
        div = _diff(predicted, observed_outcome)
        pods = compute_pods(div)  # v0.2 — single number metric
        post_payload = {
            "trace_id": trace_id,
            "pre_action_payload_hash": tr.pre_action_payload_hash,
            "observed_outcome": observed_outcome,
            "divergence_record": div,
            "divergence_score": pods,
        }
        pph = sha256_hex(_canonical_json(post_payload))
        ph = hmac_sha256_hex(self._key, pph.encode("utf-8"))
        tr.observed_outcome = observed_outcome
        tr.divergence_record = div
        tr.divergence_score = pods
        tr.post_action_payload_hash = pph
        tr.post_action_hmac = ph
        return tr

    def export_chain(self):
        return [asdict(self._traces[t]) for t in self._chain_order]

    def average_pods(self) -> float:
        """Aggregate PODS across all post-sealed traces in the chain."""
        scores = [t.divergence_score for t in self._traces.values()
                  if t.divergence_score is not None]
        return sum(scores) / len(scores) if scores else 0.0


def verify_chain(chain, key_provider):
    key = _resolve_key(key_provider)
    prev_post_hash = None
    for i, tr in enumerate(chain):
        pre_payload = {
            "schema_version": tr["schema_version"], "trace_id": tr["trace_id"],
            "session_id": tr["session_id"], "game_id": tr["game_id"],
            "action_counter": tr["action_counter"],
            "input_state_hash": tr["input_state_hash"],
            "structured_intent": tr["structured_intent"],
            "candidate_action": tr["candidate_action"],
            "prev_post_hash": tr["prev_post_hash"],
        }
        rec = sha256_hex(_canonical_json(pre_payload))
        if rec != tr["pre_action_payload_hash"]:
            return False, f"trace {i}: pre_action_payload_hash mismatch"
        if hmac_sha256_hex(key, rec.encode("utf-8")) != tr["pre_action_hmac"]:
            return False, f"trace {i}: pre_action_hmac mismatch"
        if tr["prev_post_hash"] != prev_post_hash:
            return False, f"trace {i}: prev_post_hash chain break"
        if tr["post_action_payload_hash"] is not None:
            post_payload = {
                "trace_id": tr["trace_id"],
                "pre_action_payload_hash": tr["pre_action_payload_hash"],
                "observed_outcome": tr["observed_outcome"],
                "divergence_record": tr["divergence_record"],
                "divergence_score": tr.get("divergence_score"),
            }
            rec2 = sha256_hex(_canonical_json(post_payload))
            if rec2 != tr["post_action_payload_hash"]:
                return False, f"trace {i}: post_action_payload_hash mismatch"
            if hmac_sha256_hex(key, rec2.encode("utf-8")) != tr["post_action_hmac"]:
                return False, f"trace {i}: post_action_hmac mismatch"
            prev_post_hash = rec2
        else:
            prev_post_hash = None
    return True, f"OK: {len(chain)} traces verified"


__all__ = [
    "StructuredIntent", "SealedTrace", "SealedReasoningTracer",
    "verify_chain", "compute_pods", "sha256_hex", "hmac_sha256_hex", "__version__",
]
