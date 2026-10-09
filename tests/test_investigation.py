"""Tests for the Investigation Agent and /investigate API endpoint.

All tests are unit-level — no retraining, no large dataset I/O.
The Detector and API are mocked where the model files are not available.
"""
from __future__ import annotations

import pytest

from src.agents.investigation import InvestigationAgent, SecondaryReviewRule, SEVERITY_LEVELS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(
    label="Benign",
    probability=0.98,
    anomaly_score=0.30,
    anomaly_threshold=0.62,
    anomalous=False,
    investigate=False,
    reason="benign",
    missing_features=0,
    top3=None,
):
    if top3 is None:
        top3 = [{"label": label, "probability": probability}]
    return {
        "label": label,
        "probability": probability,
        "top3": top3,
        "anomaly_score": anomaly_score,
        "anomaly_threshold": anomaly_threshold,
        "anomalous": anomalous,
        "investigate": investigate,
        "reason": reason,
        "missing_features": missing_features,
        "latency_ms": 1.5,
    }


BENIGN_FLOW = {
    "Destination Port": 80.0,
    "Subflow Fwd Bytes": 73.0,
    "Bwd Packet Length Min": 0.0,
    "Max Packet Length": 108.0,
    "SYN Flag Count": 1.0,
}

BOTNET_FLOW = {
    "Destination Port": 8080.0,
    "Subflow Fwd Bytes": 0.0,
    "Bwd Packet Length Min": 6.0,
    "Max Packet Length": 6.0,
    "Flow Bytes/s": 12.5,
}


# ---------------------------------------------------------------------------
# Investigation Agent unit tests
# ---------------------------------------------------------------------------

class TestInvestigationAgentBenign:
    def test_benign_non_anomalous_returns_info_severity(self):
        agent = InvestigationAgent()
        det = _make_detection()
        report = agent.investigate(detection=det, flow=BENIGN_FLOW)
        assert report["severity"] == "info"
        assert report["investigate_recommended"] is False
        assert report["classifier_label"] == "Benign"
        assert "disclaimer" in report
        assert report["error"] is None

    def test_benign_non_anomalous_human_review_not_recommended(self):
        agent = InvestigationAgent()
        det = _make_detection()
        report = agent.investigate(detection=det, flow=BENIGN_FLOW)
        assert report["human_review_recommended"] is False

    def test_report_has_all_required_keys(self):
        agent = InvestigationAgent()
        det = _make_detection()
        report = agent.investigate(detection=det, flow=BENIGN_FLOW)
        required = [
            "event_id", "classifier_label", "classifier_confidence",
            "alternative_scores", "anomaly_score", "anomaly_threshold",
            "is_anomalous", "investigate_recommended", "decision_reason",
            "secondary_review_triggered", "secondary_review_reason",
            "severity", "severity_rationale", "evidence", "notable_feature_values",
            "evidence_gaps", "recommended_steps", "human_review_recommended",
            "disclaimer", "missing_features", "error",
        ]
        for key in required:
            assert key in report, f"Missing key: {key}"


class TestInvestigationAgentHighConfidenceAttack:
    def test_dos_high_confidence_high_severity(self):
        agent = InvestigationAgent()
        det = _make_detection(
            label="DoS", probability=0.97, anomaly_score=0.80,
            anomalous=True, investigate=True, reason="known_attack",
        )
        report = agent.investigate(detection=det, flow={})
        assert report["severity"] == "high"
        assert report["investigate_recommended"] is True
        assert report["human_review_recommended"] is True

    def test_bruteforce_high_confidence_medium_severity(self):
        agent = InvestigationAgent()
        det = _make_detection(
            label="BruteForce", probability=0.85,
            investigate=True, reason="known_attack",
        )
        report = agent.investigate(detection=det, flow={})
        assert report["severity"] == "medium"

    def test_evidence_mentions_known_attack_reason(self):
        agent = InvestigationAgent()
        det = _make_detection(label="DoS", probability=0.95, investigate=True, reason="known_attack")
        report = agent.investigate(detection=det, flow={})
        combined = " ".join(report["evidence"])
        assert "known_attack" in combined or "high-confidence" in combined.lower()


class TestInvestigationAgentLowConfidenceAttack:
    def test_low_conf_attack_anomalous_medium_severity(self):
        agent = InvestigationAgent()
        det = _make_detection(
            label="Botnet", probability=0.35,
            anomaly_score=0.80, anomalous=True,
            investigate=True, reason="low_confidence_and_anomalous",
        )
        report = agent.investigate(detection=det, flow=BOTNET_FLOW)
        # Botnet is HIGH severity label but low confidence → medium
        assert report["severity"] == "medium"
        assert report["investigate_recommended"] is True

    def test_low_conf_attack_not_anomalous_does_not_investigate(self):
        agent = InvestigationAgent()
        det = _make_detection(
            label="DoS", probability=0.30,
            anomalous=False, investigate=False,
            reason="low_confidence_attack_not_anomalous",
        )
        report = agent.investigate(detection=det, flow={})
        assert report["investigate_recommended"] is False
        # High-severity labels downgraded (not upgraded) for low confidence: medium is correct
        assert report["severity"] in ("info", "low", "medium")


class TestInvestigationAgentBenignAnomalous:
    def test_benign_anomalous_gets_unknown_anomaly_reason(self):
        agent = InvestigationAgent()
        det = _make_detection(
            label="Benign", probability=0.99,
            anomaly_score=0.80, anomalous=True,
            investigate=True, reason="unknown_anomaly",
        )
        report = agent.investigate(detection=det, flow=BENIGN_FLOW)
        assert report["investigate_recommended"] is True
        assert report["severity"] == "low"
        combined = " ".join(report["evidence"])
        assert "anomalous" in combined.lower() or "anomaly" in combined.lower()

    def test_benign_anomalous_human_review_recommended(self):
        agent = InvestigationAgent()
        det = _make_detection(
            label="Benign", probability=0.99,
            anomalous=True, investigate=True, reason="unknown_anomaly",
        )
        report = agent.investigate(detection=det, flow={})
        assert report["human_review_recommended"] is True


class TestInvestigationAgentBotnetPrediction:
    def test_botnet_high_confidence_high_severity(self):
        agent = InvestigationAgent()
        det = _make_detection(
            label="Botnet", probability=0.75,
            investigate=True, reason="known_attack",
        )
        report = agent.investigate(detection=det, flow=BOTNET_FLOW)
        assert report["severity"] == "high"
        steps = " ".join(report["recommended_steps"])
        assert "botnet" in steps.lower() or "command" in steps.lower()

    def test_botnet_zero_fwd_bytes_noted_in_evidence(self):
        agent = InvestigationAgent()
        det = _make_detection(label="Botnet", probability=0.75, investigate=True, reason="known_attack")
        report = agent.investigate(detection=det, flow=BOTNET_FLOW)
        combined = " ".join(report["evidence"])
        assert "Subflow Fwd Bytes" in combined or "forward direction" in combined.lower()


class TestInvestigationAgentMissingFeatures:
    def test_missing_features_noted_in_gaps(self):
        agent = InvestigationAgent()
        det = _make_detection(missing_features=5)
        report = agent.investigate(detection=det, flow={})
        gaps = " ".join(report["evidence_gaps"])
        assert "5" in gaps or "missing" in gaps.lower()

    def test_empty_flow_works(self):
        agent = InvestigationAgent()
        det = _make_detection()
        report = agent.investigate(detection=det, flow={})
        assert report["error"] is None

    def test_no_anomaly_info_noted_in_gaps(self):
        agent = InvestigationAgent()
        # Detection result with no anomaly fields
        det = {"label": "Benign", "probability": 0.90, "top3": [], "missing_features": 0,
               "investigate": False, "reason": "benign"}
        report = agent.investigate(detection=det, flow={})
        gaps = " ".join(report["evidence_gaps"])
        assert "anomaly" in gaps.lower() or report["is_anomalous"] is None


class TestInvestigationAgentEdgeCases:
    def test_event_id_echoed_back(self):
        agent = InvestigationAgent()
        report = agent.investigate(detection=_make_detection(), event_id="test-evt-123")
        assert report["event_id"] == "test-evt-123"

    def test_no_event_id_is_none(self):
        agent = InvestigationAgent()
        report = agent.investigate(detection=_make_detection())
        assert report["event_id"] is None

    def test_malformed_detection_does_not_crash(self):
        agent = InvestigationAgent()
        for bad in [{}, {"label": None}, {"label": "Benign", "probability": "not-a-float"}]:
            report = agent.investigate(detection=bad, flow={})
            # should always return a dict with at least a disclaimer
            assert isinstance(report, dict)
            assert "disclaimer" in report

    def test_none_detection_with_flow_tries_auto_detect_and_handles_failure(self):
        """When detector is unavailable, error is reported but report is still returned."""
        agent = InvestigationAgent()
        # Deliberately passing no detection — detector will fail in test environment
        report = agent.investigate(detection=None, flow=BENIGN_FLOW)
        assert isinstance(report, dict)
        # Either error is populated or detection succeeded
        assert "disclaimer" in report

    def test_unexpected_label_does_not_crash(self):
        agent = InvestigationAgent()
        det = _make_detection(label="AlienAttack", probability=0.88)
        report = agent.investigate(detection=det, flow={})
        assert isinstance(report, dict)
        assert report["classifier_label"] == "AlienAttack"

    def test_severity_is_always_valid_level(self):
        agent = InvestigationAgent()
        for label in ["Benign", "DoS", "Botnet", "BruteForce", "WebAttack", "Infiltration"]:
            det = _make_detection(label=label, probability=0.80, investigate=(label != "Benign"))
            report = agent.investigate(detection=det, flow={})
            assert report["severity"] in SEVERITY_LEVELS

    def test_classifier_confidence_preserved_exactly(self):
        agent = InvestigationAgent()
        det = _make_detection(probability=0.7654)
        report = agent.investigate(detection=det, flow={})
        assert report["classifier_confidence"] == round(0.7654, 4)

    def test_original_classifier_prediction_not_modified(self):
        """The agent must never overwrite or upgrade the classifier label."""
        agent = InvestigationAgent()
        det = _make_detection(label="Benign", probability=0.99)
        report = agent.investigate(detection=det, flow=BOTNET_FLOW)
        # Even though flow looks suspicious, classifier said Benign
        assert report["classifier_label"] == "Benign"

    def test_disclaimer_always_present(self):
        agent = InvestigationAgent()
        for label in ("Benign", "Botnet", "DoS"):
            det = _make_detection(label=label)
            report = agent.investigate(detection=det, flow={})
            assert len(report["disclaimer"]) > 10


# ---------------------------------------------------------------------------
# Secondary Review Heuristic Unit Tests
# ---------------------------------------------------------------------------

class TestSecondaryReviewHeuristic:
    """Tests for the configurable secondary-review heuristic."""

    def test_suspicious_botnet_pattern_triggers_secondary_review(self):
        """Synthetic Botnet-like flow triggers secondary review and human review,

        while preserving classifier's Benign prediction and investigate=False.
        """
        agent = InvestigationAgent()
        det = _make_detection(
            label="Benign",
            probability=0.9753,
            anomaly_score=0.3243,
            anomaly_threshold=0.6246,
            anomalous=False,
            investigate=False,
            reason="benign",
        )
        report = agent.investigate(detection=det, flow=BOTNET_FLOW)

        # 1. Secondary review flags the flow
        assert report["secondary_review_triggered"] is True
        assert report["secondary_review_reason"] is not None
        assert "Subflow Fwd Bytes = 0" in report["secondary_review_reason"]
        assert "secondary review" in report["secondary_review_reason"].lower()

        # 2. Human review is recommended
        assert report["human_review_recommended"] is True

        # 3. Classifier investigate decision remains False (strictly separated)
        assert report["investigate_recommended"] is False
        assert report["decision_reason"] == "benign"

        # 4. Original classifier results preserved exactly
        assert report["classifier_label"] == "Benign"
        assert report["classifier_confidence"] == 0.9753
        assert report["anomaly_score"] == 0.3243
        assert report["anomaly_threshold"] == 0.6246
        assert report["is_anomalous"] is False

        # 5. Severity upgraded to low for human triage
        assert report["severity"] == "low"
        assert "secondary-review" in report["severity_rationale"].lower()

        # 6. Evidence and recommended steps reflect heuristic finding
        combined_evidence = " ".join(report["evidence"])
        assert "secondary review heuristic" in combined_evidence.lower()
        combined_steps = " ".join(report["recommended_steps"])
        assert "secondary human review" in combined_steps.lower()

    def test_ordinary_benign_flow_does_not_trigger_secondary_review(self):
        """Ordinary benign flows must not trigger secondary review or human review."""
        agent = InvestigationAgent()
        det = _make_detection(label="Benign", probability=0.99, anomalous=False, investigate=False)
        report = agent.investigate(detection=det, flow=BENIGN_FLOW)

        assert report["secondary_review_triggered"] is False
        assert report["secondary_review_reason"] is None
        assert report["human_review_recommended"] is False
        assert report["investigate_recommended"] is False
        assert report["severity"] == "info"

    def test_missing_features_do_not_trigger_or_crash(self):
        """Flows missing key heuristic features should not trigger or raise errors."""
        agent = InvestigationAgent()
        det = _make_detection()

        # Missing Subflow Fwd Bytes
        flow1 = {"Bwd Packet Length Min": 6.0, "Max Packet Length": 6.0}
        report1 = agent.investigate(detection=det, flow=flow1)
        assert report1["secondary_review_triggered"] is False
        assert report1["error"] is None

        # Missing Bwd Packet Length Min
        flow2 = {"Subflow Fwd Bytes": 0.0, "Max Packet Length": 6.0}
        report2 = agent.investigate(detection=det, flow=flow2)
        assert report2["secondary_review_triggered"] is False
        assert report2["error"] is None

        # Missing Max Packet Length
        flow3 = {"Subflow Fwd Bytes": 0.0, "Bwd Packet Length Min": 6.0}
        report3 = agent.investigate(detection=det, flow=flow3)
        assert report3["secondary_review_triggered"] is False
        assert report3["error"] is None

        # Empty flow dict
        report4 = agent.investigate(detection=det, flow={})
        assert report4["secondary_review_triggered"] is False
        assert report4["error"] is None

    def test_malformed_values_handled_gracefully(self):
        """Malformed, non-numeric, or invalid feature values must not trigger or crash."""
        agent = InvestigationAgent()
        det = _make_detection()

        malformed_flows = [
            {"Subflow Fwd Bytes": "zero", "Bwd Packet Length Min": 6.0, "Max Packet Length": 6.0},
            {"Subflow Fwd Bytes": 0.0, "Bwd Packet Length Min": "six", "Max Packet Length": 6.0},
            {"Subflow Fwd Bytes": None, "Bwd Packet Length Min": 6.0, "Max Packet Length": 6.0},
            {"Subflow Fwd Bytes": -5.0, "Bwd Packet Length Min": 6.0, "Max Packet Length": 6.0},
            {"Subflow Fwd Bytes": 0.0, "Bwd Packet Length Min": -1.0, "Max Packet Length": 6.0},
            {"Subflow Fwd Bytes": float("nan"), "Bwd Packet Length Min": 6.0, "Max Packet Length": 6.0},
            {"Subflow Fwd Bytes": float("inf"), "Bwd Packet Length Min": 6.0, "Max Packet Length": 6.0},
        ]
        for bad_flow in malformed_flows:
            report = agent.investigate(detection=det, flow=bad_flow)
            assert report["secondary_review_triggered"] is False
            assert report["error"] is None

    def test_preservation_of_classifier_prediction_when_triggered(self):
        """The original classifier output is immutable regardless of heuristic escalation."""
        agent = InvestigationAgent()
        det = _make_detection(
            label="Benign",
            probability=0.9876,
            anomaly_score=0.4123,
            anomaly_threshold=0.6246,
            anomalous=False,
            investigate=False,
            reason="benign",
            top3=[{"label": "Benign", "probability": 0.9876}, {"label": "Botnet", "probability": 0.0124}],
        )
        report = agent.investigate(detection=det, flow=BOTNET_FLOW)

        assert report["classifier_label"] == "Benign"
        assert report["classifier_confidence"] == 0.9876
        assert report["anomaly_score"] == 0.4123
        assert report["anomaly_threshold"] == 0.6246
        assert report["is_anomalous"] is False
        assert report["investigate_recommended"] is False
        assert report["alternative_scores"][0]["label"] == "Benign"
        assert report["secondary_review_triggered"] is True
        assert report["human_review_recommended"] is True

    def test_configurable_secondary_review_disabled(self):
        """Disabling secondary review prevents heuristic from triggering."""
        agent = InvestigationAgent(secondary_review_enabled=False)
        det = _make_detection(label="Benign", probability=0.95)
        report = agent.investigate(detection=det, flow=BOTNET_FLOW)

        assert report["secondary_review_triggered"] is False
        assert report["secondary_review_reason"] is None
        assert report["human_review_recommended"] is False

    def test_configurable_rule_thresholds(self):
        """SecondaryReviewRule thresholds can be customized."""
        # Rule requiring max_fwd_bytes == 0 should not trigger on fwd = 10
        strict_rule = SecondaryReviewRule(max_fwd_bytes=0.0)
        agent_strict = InvestigationAgent(secondary_review_rules=[strict_rule])
        flow_with_fwd = {
            "Subflow Fwd Bytes": 10.0,
            "Bwd Packet Length Min": 6.0,
            "Max Packet Length": 6.0,
        }
        report = agent_strict.investigate(detection=_make_detection(), flow=flow_with_fwd)
        assert report["secondary_review_triggered"] is False

        # Custom permissive rule allowing up to 20 fwd bytes should trigger
        permissive_rule = SecondaryReviewRule(max_fwd_bytes=20.0)
        agent_permissive = InvestigationAgent(secondary_review_rules=[permissive_rule])
        report2 = agent_permissive.investigate(detection=_make_detection(), flow=flow_with_fwd)
        assert report2["secondary_review_triggered"] is True

    def test_configurable_target_ports(self):
        """SecondaryReviewRule can be restricted to specific destination ports."""
        port_8080_rule = SecondaryReviewRule(target_ports=[8080])
        agent = InvestigationAgent(secondary_review_rules=[port_8080_rule])

        # Flow on port 8080 triggers
        report_8080 = agent.investigate(detection=_make_detection(), flow=BOTNET_FLOW)
        assert report_8080["secondary_review_triggered"] is True

        # Same packet pattern on port 80 does not trigger
        flow_port_80 = dict(BOTNET_FLOW, **{"Destination Port": 80.0})
        report_80 = agent.investigate(detection=_make_detection(), flow=flow_port_80)
        assert report_80["secondary_review_triggered"] is False


# ---------------------------------------------------------------------------
# API integration tests (require running app + detector)
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client():
    """Return a TestClient if FastAPI and models are available; skip otherwise."""
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app
        # Test detector is loadable
        from src.api.service import get_detector
        get_detector()
        return TestClient(app)
    except Exception as exc:
        pytest.skip(f"API/detector not available in this environment: {exc}")


def _sample_flow():
    """Load the pre-saved API test flow that matches the feature schema."""
    import json
    from src.ml.common import REPORTS
    p = REPORTS / "api_test_flow.json"
    if p.exists():
        return json.loads(p.read_text())
    # Minimal fallback — all features will be imputed from medians
    return {"Destination Port": 80.0, "SYN Flag Count": 1.0}


class TestAPIExistingEndpoints:
    def test_health(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "n_features" in body
        assert "classes" in body

    def test_detect_valid_flow(self, api_client):
        r = api_client.post("/detect", json=_sample_flow())
        assert r.status_code == 200
        body = r.json()
        assert "label" in body
        assert "probability" in body
        assert "anomaly_score" in body
        assert "investigate" in body

    def test_detect_empty_flow_422(self, api_client):
        r = api_client.post("/detect", json={})
        assert r.status_code == 422

    def test_detect_batch_valid(self, api_client):
        r = api_client.post("/detect/batch", json=[_sample_flow()])
        assert r.status_code == 200
        assert isinstance(r.json(), list)
        assert len(r.json()) == 1


class TestAPIInvestigateEndpoint:
    def test_investigate_valid_flow(self, api_client):
        r = api_client.post("/investigate", json={"flow": _sample_flow()})
        assert r.status_code == 200
        body = r.json()
        assert "classifier_label" in body
        assert "severity" in body
        assert "evidence" in body
        assert "disclaimer" in body
        assert "recommended_steps" in body
        assert "investigate_recommended" in body

    def test_investigate_with_event_id(self, api_client):
        r = api_client.post(
            "/investigate",
            json={"flow": _sample_flow(), "event_id": "test-evt-001"},
        )
        assert r.status_code == 200
        assert r.json()["event_id"] == "test-evt-001"

    def test_investigate_empty_flow_422(self, api_client):
        r = api_client.post("/investigate", json={"flow": {}})
        assert r.status_code == 422

    def test_investigate_missing_flow_key_422(self, api_client):
        r = api_client.post("/investigate", json={})
        # flow defaults to {} which triggers the empty-flow check
        assert r.status_code == 422

    def test_investigate_extra_field_rejected(self, api_client):
        r = api_client.post(
            "/investigate",
            json={"flow": _sample_flow(), "unknown_field": "bad"},
        )
        assert r.status_code == 422

    def test_investigate_preserves_detect_label(self, api_client):
        """The /investigate label must equal what /detect returns for the same flow."""
        flow = _sample_flow()
        det_r = api_client.post("/detect", json=flow).json()
        inv_r = api_client.post("/investigate", json={"flow": flow}).json()
        assert inv_r["classifier_label"] == det_r["label"]

    def test_investigate_anomalous_flow_has_evidence(self, api_client):
        """An anomalous flow should mention anomaly in evidence."""
        flow = _sample_flow()
        inv = api_client.post("/investigate", json={"flow": flow}).json()
        if inv.get("is_anomalous"):
            combined = " ".join(inv["evidence"])
            assert "anomal" in combined.lower()

    def test_investigate_botnet_flow_steps_mention_botnet(self, api_client):
        """When classifier labels as Botnet, recommended steps should reference it."""
        # Use the BOTNET_FLOW feature pattern — classifier may or may not return Botnet
        r = api_client.post("/investigate", json={"flow": BOTNET_FLOW})
        assert r.status_code == 200
        body = r.json()
        if body["classifier_label"] == "Botnet":
            steps = " ".join(body["recommended_steps"]).lower()
            assert "botnet" in steps or "command" in steps

    def test_existing_detect_still_works_after_investigate_added(self, api_client):
        """Regression: /detect must still work after /investigate was added."""
        r = api_client.post("/detect", json=_sample_flow())
        assert r.status_code == 200
        assert "label" in r.json()

    def test_investigate_synthetic_botnet_triggers_secondary_review(self, api_client):
        """Live API test: synthetic Botnet-like flow triggers secondary review and human review."""
        r = api_client.post("/investigate", json={"flow": BOTNET_FLOW, "event_id": "test-sec-rev"})
        assert r.status_code == 200
        body = r.json()
        assert body["event_id"] == "test-sec-rev"
        assert body["secondary_review_triggered"] is True
        assert body["human_review_recommended"] is True
        assert body["secondary_review_reason"] is not None
        assert "Subflow Fwd Bytes = 0" in body["secondary_review_reason"]
        # Classifier prediction remains Benign and investigate remains False
        assert body["classifier_label"] == "Benign"
        assert body["investigate_recommended"] is False
        assert body["is_anomalous"] is False

    def test_investigate_known_benign_flow_does_not_trigger_secondary_review(self, api_client):
        """Live API test: known benign flow fixture must not trigger secondary review."""
        r = api_client.post("/investigate", json={"flow": _sample_flow(), "event_id": "test-benign-known"})
        assert r.status_code == 200
        body = r.json()
        assert body["event_id"] == "test-benign-known"
        assert body["secondary_review_triggered"] is False
        assert body["secondary_review_reason"] is None
        assert body["human_review_recommended"] is False
        assert body["investigate_recommended"] is False
        assert body["classifier_label"] == "Benign"
        assert body["error"] is None

