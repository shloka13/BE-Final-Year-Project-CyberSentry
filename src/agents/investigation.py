"""CyberSentry Investigation Agent.

Deterministic, evidence-based investigation of network flows.  No external
APIs, no LLM calls, no invented identities or timestamps.

Usage (standalone):
    from src.agents.investigation import InvestigationAgent
    agent = InvestigationAgent()
    report = agent.investigate(detection_result, flow_features, event_id="evt-001")

The agent works with the dict returned by ``Detector.detect()`` in
``src/api/service.py``.  All fields of that dict are optional here so the
agent can also be called without a pre-computed detection result.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Severity levels and their ordering
# ---------------------------------------------------------------------------
SEVERITY_LEVELS = ["info", "low", "medium", "high", "critical"]

# Attack labels that warrant elevated severity
HIGH_SEVERITY_LABELS = {"DoS", "DDoS", "Botnet"}
MEDIUM_SEVERITY_LABELS = {"BruteForce", "WebAttack", "Infiltration", "PortScan"}

# Feature names that carry investigative weight when unusual
NOTABLE_FEATURES: list[str] = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "Max Packet Length",
    "Bwd Packet Length Min",
]

# Known well-known ports for context only (never used to override the model)
WELL_KNOWN_PORTS: dict[int, str] = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    110: "POP3",
    143: "IMAP",
    389: "LDAP",
    443: "HTTPS",
    445: "SMB",
    3306: "MySQL",
    3389: "RDP",
    5432: "PostgreSQL",
    8080: "HTTP-alt",
    8443: "HTTPS-alt",
}


# ---------------------------------------------------------------------------
# Secondary Review Heuristic
# ---------------------------------------------------------------------------
@dataclass
class SecondaryReviewRule:
    """Configurable rule for secondary-review heuristic.

    Identifies suspicious combinations of observed network features that
    may warrant human analyst review even when the primary classifier
    and anomaly detector do not flag the flow.

    These heuristics flag structural anomalies for manual triage without
    treating them as definitive proof of malicious or botnet activity.
    """
    name: str = "zero_fwd_minimal_packet_size"
    enabled: bool = True
    max_fwd_bytes: float = 0.0
    max_bwd_packet_min: float = 6.0
    max_packet_length: float = 6.0
    target_ports: list[int] | tuple[int, ...] | None = None
    description: str = (
        "Zero forward bytes with minimal packet sizes (<= 6 bytes). "
        "Characteristic of asymmetric polling, beaconing, or heartbeat patterns."
    )

    def evaluate(self, flow: dict) -> tuple[bool, str | None]:
        if not self.enabled:
            return False, None

        # Feature extraction with robust numeric conversion
        fwd_raw = flow.get("Subflow Fwd Bytes")
        bwd_min_raw = flow.get("Bwd Packet Length Min")
        max_pkt_raw = flow.get("Max Packet Length")

        if fwd_raw is None or bwd_min_raw is None or max_pkt_raw is None:
            return False, None

        try:
            fwd_val = float(fwd_raw)
            bwd_min_val = float(bwd_min_raw)
            max_pkt_val = float(max_pkt_raw)
        except (TypeError, ValueError):
            return False, None

        import math
        if not (math.isfinite(fwd_val) and math.isfinite(bwd_min_val) and math.isfinite(max_pkt_val)):
            return False, None
        if not (0.0 <= fwd_val <= self.max_fwd_bytes):
            return False, None
        if not (0.0 <= bwd_min_val <= self.max_bwd_packet_min):
            return False, None
        if not (0.0 <= max_pkt_val <= self.max_packet_length):
            return False, None

        # Target ports check if configured
        dest_port = flow.get("Destination Port")
        port_int = None
        if dest_port is not None:
            try:
                port_int = int(float(dest_port))
            except (TypeError, ValueError):
                pass

        if self.target_ports is not None:
            if port_int is None or port_int not in self.target_ports:
                return False, None

        port_str = ""
        if port_int is not None:
            svc = WELL_KNOWN_PORTS.get(port_int, "")
            port_str = f" on port {port_int}" + (f" ({svc})" if svc else "")

        reason = (
            f"Secondary review heuristic triggered ({self.name}): "
            f"observed Subflow Fwd Bytes = {fwd_val:.0f}, "
            f"Bwd Packet Length Min = {bwd_min_val:.0f}, "
            f"Max Packet Length = {max_pkt_val:.0f}{port_str}. "
            "This combination indicates one-directional traffic with minimal packet sizes, "
            "warranting human analyst review for potential beaconing or automated probe activity. "
            "(Note: heuristic observation, not classifier proof of attack)."
        )
        return True, reason


# ---------------------------------------------------------------------------
# Dataclass for the structured report
# ---------------------------------------------------------------------------
@dataclass
class InvestigationReport:
    event_id: str | None
    # --- classifier results ---
    classifier_label: str
    classifier_confidence: float
    alternative_scores: list[dict]  # top3 from detector
    # --- anomaly results ---
    anomaly_score: float | None
    anomaly_threshold: float | None
    is_anomalous: bool | None
    # --- decision ---
    investigate_recommended: bool
    decision_reason: str
    # --- secondary review heuristic ---
    secondary_review_triggered: bool
    secondary_review_reason: str | None
    # --- severity ---
    severity: str
    severity_rationale: str
    # --- evidence ---
    evidence: list[str]
    notable_feature_values: dict[str, Any]
    evidence_gaps: list[str]
    # --- actions ---
    recommended_steps: list[str]
    human_review_recommended: bool
    # --- disclaimer ---
    disclaimer: str
    # --- meta ---
    missing_features: int
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "classifier_label": self.classifier_label,
            "classifier_confidence": self.classifier_confidence,
            "alternative_scores": self.alternative_scores,
            "anomaly_score": self.anomaly_score,
            "anomaly_threshold": self.anomaly_threshold,
            "is_anomalous": self.is_anomalous,
            "investigate_recommended": self.investigate_recommended,
            "decision_reason": self.decision_reason,
            "secondary_review_triggered": self.secondary_review_triggered,
            "secondary_review_reason": self.secondary_review_reason,
            "secondary_review_explanation": self.secondary_review_reason,
            "trigger_explanation": self.secondary_review_reason,
            "severity": self.severity,
            "severity_rationale": self.severity_rationale,
            "evidence": self.evidence,
            "notable_feature_values": self.notable_feature_values,
            "evidence_gaps": self.evidence_gaps,
            "recommended_steps": self.recommended_steps,
            "human_review_recommended": self.human_review_recommended,
            "disclaimer": self.disclaimer,
            "missing_features": self.missing_features,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class InvestigationAgent:
    """Deterministic, evidence-based investigation agent.

    Parameters
    ----------
    high_confidence_threshold:
        Minimum classifier probability to consider the label reliable.
    secondary_review_enabled:
        Whether the secondary-review heuristic is active.
    secondary_review_rules:
        List of SecondaryReviewRule objects to evaluate. Defaults to a rule
        flagging flows with zero forward bytes and packet lengths <= 6.
    """

    DISCLAIMER = (
        "This report is generated by an automated ML system. "
        "Model predictions are statistical estimates, not confirmed proof of an attack. "
        "All findings must be reviewed by a qualified security analyst before taking action."
    )

    def __init__(
        self,
        high_confidence_threshold: float = 0.60,
        secondary_review_enabled: bool = True,
        secondary_review_rules: list[SecondaryReviewRule] | None = None,
    ) -> None:
        self.high_conf_thr = high_confidence_threshold
        self.secondary_review_enabled = secondary_review_enabled
        if secondary_review_rules is not None:
            self.secondary_review_rules = secondary_review_rules
        else:
            self.secondary_review_rules = [SecondaryReviewRule()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def investigate(
        self,
        detection: dict | None = None,
        flow: dict | None = None,
        event_id: str | None = None,
    ) -> dict:
        """Investigate a network event.

        Parameters
        ----------
        detection:
            Dict returned by ``Detector.detect()``. If *None* the agent will
            try to run the detector automatically when *flow* is provided.
        flow:
            Raw feature dict (feature name → numeric value).  Used to extract
            notable feature values and evaluate secondary review heuristics.
        event_id:
            Optional caller-supplied identifier.  Never invented by the agent.

        Returns
        -------
        dict  — serialisable investigation report.
        """
        try:
            # If no detection supplied but flow is provided, run detector
            if detection is None and flow:
                try:
                    from src.api.service import get_detector
                    detection = get_detector().detect(flow)
                except Exception as exc:
                    detection = {}
                    _err = f"Detector unavailable: {exc}"
            else:
                _err = None

            detection = detection or {}
            flow = flow or {}

            report = self._build_report(detection, flow, event_id)
            if _err and report.error is None:
                report.error = _err
            return report.to_dict()

        except Exception:
            return {
                "event_id": event_id,
                "secondary_review_triggered": False,
                "secondary_review_reason": None,
                "error": f"InvestigationAgent internal error: {traceback.format_exc()}",
                "disclaimer": self.DISCLAIMER,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evaluate_secondary_review(self, flow: dict) -> tuple[bool, str | None]:
        if not self.secondary_review_enabled:
            return False, None
        for rule in self.secondary_review_rules:
            triggered, reason = rule.evaluate(flow)
            if triggered:
                return True, reason
        return False, None

    def _build_report(
        self, detection: dict, flow: dict, event_id: str | None
    ) -> InvestigationReport:
        label = str(detection.get("label", "Unknown"))
        confidence = float(detection.get("probability", 0.0))
        top3 = detection.get("top3", [])
        if not isinstance(top3, list):
            top3 = []
        anomaly_score = detection.get("anomaly_score")
        anomaly_threshold = detection.get("anomaly_threshold")
        is_anomalous = detection.get("anomalous")
        investigate = bool(detection.get("investigate", False))
        reason = str(detection.get("reason", "not_provided"))
        missing_features = int(detection.get("missing_features", 0))

        # Normalise types
        anomaly_score = float(anomaly_score) if anomaly_score is not None else None
        anomaly_threshold = (
            float(anomaly_threshold) if anomaly_threshold is not None else None
        )
        is_anomalous = bool(is_anomalous) if is_anomalous is not None else None

        # --- secondary review heuristic --------------------------------------
        sec_triggered, sec_reason = self._evaluate_secondary_review(flow)

        # --- notable features -------------------------------------------------
        notable = self._extract_notable(flow)

        # --- evidence statements ----------------------------------------------
        evidence, gaps = self._build_evidence(
            label, confidence, is_anomalous, anomaly_score, anomaly_threshold,
            missing_features, notable, reason, sec_triggered, sec_reason,
        )

        # --- severity ---------------------------------------------------------
        severity, sev_rationale = self._assess_severity(
            label, confidence, is_anomalous, investigate, sec_triggered
        )

        # --- recommended steps ------------------------------------------------
        steps = self._recommend_steps(
            label, confidence, is_anomalous, notable, reason, sec_triggered
        )

        # --- human review -----------------------------------------------------
        human_review = (
            investigate
            or is_anomalous
            or sec_triggered
            or (label not in ("Benign", "Unknown") and confidence < self.high_conf_thr)
        )

        return InvestigationReport(
            event_id=event_id,
            classifier_label=label,
            classifier_confidence=round(confidence, 4),
            alternative_scores=top3,
            anomaly_score=round(anomaly_score, 4) if anomaly_score is not None else None,
            anomaly_threshold=(
                round(anomaly_threshold, 4) if anomaly_threshold is not None else None
            ),
            is_anomalous=is_anomalous,
            investigate_recommended=investigate,
            decision_reason=reason,
            secondary_review_triggered=sec_triggered,
            secondary_review_reason=sec_reason,
            severity=severity,
            severity_rationale=sev_rationale,
            evidence=evidence,
            notable_feature_values=notable,
            evidence_gaps=gaps,
            recommended_steps=steps,
            human_review_recommended=human_review,
            disclaimer=self.DISCLAIMER,
            missing_features=missing_features,
            error=None,
        )

    # ------------------------------------------------------------------
    def _extract_notable(self, flow: dict) -> dict:
        """Return a subset of flow features relevant to an investigation."""
        result: dict = {}
        for feat in NOTABLE_FEATURES:
            if feat in flow:
                try:
                    result[feat] = float(flow[feat])
                except (TypeError, ValueError):
                    pass
        # Annotate destination port
        port = result.get("Destination Port")
        if port is not None:
            svc = WELL_KNOWN_PORTS.get(int(port))
            if svc:
                result["_dest_port_service"] = svc
        return result

    # ------------------------------------------------------------------
    def _build_evidence(
        self,
        label: str,
        confidence: float,
        is_anomalous: bool | None,
        anomaly_score: float | None,
        anomaly_threshold: float | None,
        missing_features: int,
        notable: dict,
        reason: str,
        sec_triggered: bool = False,
        sec_reason: str | None = None,
    ) -> tuple[list[str], list[str]]:
        evidence: list[str] = []
        gaps: list[str] = []

        # Classifier statement
        if confidence >= self.high_conf_thr:
            evidence.append(
                f"Classifier labels flow as '{label}' with {confidence:.1%} confidence "
                f"(above {self.high_conf_thr:.0%} reliability threshold)."
            )
        else:
            evidence.append(
                f"Classifier labels flow as '{label}' with {confidence:.1%} confidence "
                f"(below {self.high_conf_thr:.0%} threshold — result is uncertain)."
            )

        # Anomaly statement
        if is_anomalous is True and anomaly_score is not None:
            evidence.append(
                f"Isolation Forest flags this flow as anomalous "
                f"(score {anomaly_score:.4f} > threshold {anomaly_threshold:.4f}). "
                "This indicates the flow pattern differs from normal baseline traffic."
            )
        elif is_anomalous is False and anomaly_score is not None:
            evidence.append(
                f"Isolation Forest does not flag this flow as anomalous "
                f"(score {anomaly_score:.4f} ≤ threshold {anomaly_threshold:.4f})."
            )

        # Specific decision reasons
        if reason == "known_attack":
            evidence.append(
                "Decision rule: high-confidence attack label — flow is flagged for investigation."
            )
        elif reason == "low_confidence_and_anomalous":
            evidence.append(
                "Decision rule: low-confidence attack label AND anomalous — "
                "both signals independently suggest this flow warrants review."
            )
        elif reason == "unknown_anomaly":
            evidence.append(
                "Decision rule: flow classified as Benign, but Isolation Forest flags it as anomalous. "
                "The classifier may be missing a novel or rare attack pattern."
            )
        elif reason == "low_confidence_attack_not_anomalous":
            evidence.append(
                "Decision rule: low-confidence attack label but NOT anomalous — "
                "insufficient evidence to flag; classifier is uncertain."
            )
        elif reason == "benign":
            evidence.append(
                "Decision rule: classifier labels as Benign and flow is not anomalous — "
                "no evidence of malicious activity from available signals."
            )

        # Secondary review heuristic statement
        if sec_triggered and sec_reason:
            evidence.append(sec_reason)

        # Notable feature observations (factual, no invented context)
        fwd_bytes = notable.get("Subflow Fwd Bytes")
        bwd_min = notable.get("Bwd Packet Length Min")
        max_pkt = notable.get("Max Packet Length")
        flow_bps = notable.get("Flow Bytes/s")
        syn = notable.get("SYN Flag Count")
        rst = notable.get("RST Flag Count")
        dest_port = notable.get("Destination Port")
        svc = notable.get("_dest_port_service")

        if fwd_bytes is not None and fwd_bytes == 0:
            evidence.append(
                "Observed: Subflow Fwd Bytes = 0 — no data was sent in the forward direction. "
                "This is consistent with one-directional or scan traffic."
            )
        if bwd_min is not None and bwd_min == 6:
            evidence.append(
                "Observed: Bwd Packet Length Min = 6 — minimum backward packet very small, "
                "consistent with TCP header-only or minimal response traffic."
            )
        if max_pkt is not None and max_pkt <= 6:
            evidence.append(
                f"Observed: Max Packet Length = {int(max_pkt)} — extremely small maximum "
                "packet size for this flow."
            )
        if syn is not None and syn > 0:
            evidence.append(f"Observed: SYN flag set ({int(syn)} time(s)) in this flow.")
        if rst is not None and rst > 0:
            evidence.append(f"Observed: RST flag set ({int(rst)} time(s)) — connection reset.")
        if dest_port is not None:
            port_note = f" ({svc})" if svc else ""
            evidence.append(f"Observed: Destination port {int(dest_port)}{port_note}.")
        if flow_bps is not None:
            evidence.append(f"Observed: Flow rate = {flow_bps:.1f} bytes/s.")

        # Missing feature gaps
        if missing_features > 0:
            gaps.append(
                f"{missing_features} feature(s) were missing and imputed using training-set medians. "
                "The classifier result may be less reliable than for complete flows."
            )
        if is_anomalous is None:
            gaps.append(
                "Anomaly score not available — Isolation Forest result could not be retrieved."
            )
        if label == "Botnet" and confidence < self.high_conf_thr:
            gaps.append(
                "Botnet classification has low confidence. The training data for Botnet is limited. "
                "Consider evaluating additional flow context."
            )
        if label == "Unknown":
            gaps.append("Classifier returned an unrecognised label. Model may be stale.")

        return evidence, gaps

    # ------------------------------------------------------------------
    def _assess_severity(
        self,
        label: str,
        confidence: float,
        is_anomalous: bool | None,
        investigate: bool,
        secondary_review_triggered: bool = False,
    ) -> tuple[str, str]:
        if label in HIGH_SEVERITY_LABELS and confidence >= self.high_conf_thr:
            return "high", (
                f"High-severity attack class '{label}' detected with high classifier confidence "
                f"({confidence:.1%})."
            )
        if label in HIGH_SEVERITY_LABELS:
            return "medium", (
                f"High-severity attack class '{label}' flagged but classifier confidence is low "
                f"({confidence:.1%}). Severity downgraded to medium pending review."
            )
        if label in MEDIUM_SEVERITY_LABELS and confidence >= self.high_conf_thr:
            return "medium", (
                f"Medium-severity attack class '{label}' detected with high confidence ({confidence:.1%})."
            )
        if label in MEDIUM_SEVERITY_LABELS:
            return "low", (
                f"Medium-severity attack class '{label}' flagged with low confidence ({confidence:.1%})."
            )
        if label == "Benign" and is_anomalous:
            return "low", (
                "Flow classified as Benign but flagged as anomalous by Isolation Forest. "
                "Low severity assigned pending manual review."
            )
        if label == "Benign" and secondary_review_triggered:
            return "low", (
                "Flow classified as Benign by ML model, but flagged for manual review by "
                "secondary-review heuristic due to suspicious structural flow features. "
                "Low severity assigned pending analyst inspection."
            )
        return "info", "No attack class detected and flow is not flagged as anomalous."

    # ------------------------------------------------------------------
    def _recommend_steps(
        self,
        label: str,
        confidence: float,
        is_anomalous: bool | None,
        notable: dict,
        reason: str,
        secondary_review_triggered: bool = False,
    ) -> list[str]:
        steps: list[str] = []
        dest_port = notable.get("Destination Port")
        svc = notable.get("_dest_port_service")

        if label in HIGH_SEVERITY_LABELS or label in MEDIUM_SEVERITY_LABELS:
            steps.append(
                f"Correlate this flow with other flows from the same source host in the same "
                f"time window to determine if this is an isolated event or part of a campaign."
            )
        if label == "Botnet":
            steps.append(
                "Check whether the affected host has established persistent connections to external "
                "IP addresses (command-and-control indicators). Review process and netstat output."
            )
            steps.append(
                "Verify whether Subflow Fwd Bytes = 0 is consistent with a known Botnet "
                "heartbeat or C2 polling pattern."
            )
        if label == "DoS" or label == "DDoS":
            steps.append(
                "Review traffic volume on the targeted service. Check upstream firewall and rate-limiting rules."
            )
        if label == "BruteForce":
            steps.append(
                "Check authentication logs for the targeted service for repeated failures. "
                "Enforce account lockout if not already configured."
            )
        if label == "WebAttack":
            steps.append(
                "Review web server access logs for SQL injection or XSS payloads. "
                "Check whether any requests reached the application layer."
            )
        if is_anomalous and label == "Benign":
            steps.append(
                "Investigate what process or service generated this flow — it has unusual statistical "
                "properties compared to the baseline but was not classified as a known attack."
            )
        if secondary_review_triggered:
            steps.append(
                "Perform secondary human review: flow exhibits zero-forward-data pattern with "
                "minimal packet sizes. Verify whether this matches authorized application polling "
                "or requires endpoint isolation and forensic inspection."
            )
        if dest_port is not None:
            port_int = int(dest_port)
            if svc:
                steps.append(
                    f"Verify whether the {svc} service on port {port_int} is expected to be "
                    "accessible from the traffic source."
                )
            elif port_int > 1024:
                steps.append(
                    f"Destination port {port_int} is a high/ephemeral port. Verify whether a "
                    "legitimate service is running on this port."
                )
        if reason in ("low_confidence_attack_not_anomalous", "low_confidence_and_anomalous"):
            steps.append(
                "Low-confidence classification — gather additional flow context before escalating."
            )
        if not steps:
            steps.append("No specific defensive steps recommended; continue monitoring baseline traffic.")
        return steps
