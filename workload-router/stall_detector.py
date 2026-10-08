"""Bounded progress and escalation signals for agent sessions."""

from dataclasses import dataclass


@dataclass
class StallState:
    phase: str = ""
    last_action: str = ""
    repeated_actions: int = 0
    failed_verifications: int = 0
    escalation_level: int = 0

    def observe(self, action: str, verification_failed: bool = False) -> None:
        """Track repeated actions and failed verification without raw history."""
        if action and action == self.last_action:
            self.repeated_actions += 1
        else:
            self.last_action = action
            self.repeated_actions = 1 if action else 0
        self.failed_verifications = self.failed_verifications + 1 if verification_failed else 0

    def should_escalate(self, repeat_limit: int = 3, verification_limit: int = 2) -> bool:
        return self.repeated_actions >= repeat_limit or self.failed_verifications >= verification_limit

    def escalate(self, ceiling: int = 2) -> bool:
        """Raise effort once up to a bounded ceiling; return whether it changed."""
        if self.escalation_level >= ceiling:
            return False
        self.escalation_level += 1
        return True

    def mark_progress(self, phase: str | None = None) -> None:
        self.repeated_actions = 0
        self.failed_verifications = 0
        if phase is not None:
            self.phase = phase
