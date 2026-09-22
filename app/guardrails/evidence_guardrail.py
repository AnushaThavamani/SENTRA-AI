"""Active Evidence Guardrail to validate specifications against research evidence."""
from __future__ import annotations

from app.agents.models import GuardrailResult, SpecificationResult


class EvidenceGuardrail:
    """Inspects the generated ML specification against retrieved evidence."""

    def __init__(self, max_attempts: int = 3) -> None:
        self.max_attempts = max_attempts

    def validate(self, spec: SpecificationResult, attempt_number: int) -> GuardrailResult:
        unsupported = []
        missing = []
        actions = {}

        for name, item in spec.items.items():
            if item.status == "UNSUPPORTED":
                unsupported.append(name)
                # Formulate a targeted query to look for this specific missing detail
                actions[name] = f"Find specific evidence for the ML {name} used in the model."
            elif item.status == "MISSING":
                missing.append(name)
                actions[name] = f"Find specific evidence for the ML {name} used in the model."

        passed = len(unsupported) == 0 and len(missing) == 0

        # If it passed or we've exhausted our max attempts, we don't request a revision
        revision_required = not passed and attempt_number < self.max_attempts

        return GuardrailResult(
            passed=passed,
            revision_required=revision_required,
            attempt_number=attempt_number,
            max_attempts=self.max_attempts,
            unsupported_requirements=unsupported,
            missing_requirements=missing,
            corrective_actions=actions if revision_required else {}
        )
