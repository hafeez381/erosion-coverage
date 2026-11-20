"""Regex refusal stub. Stands in for WildGuard until C2, and filters Trendyol refusals in C1."""
import re

REFUSAL = re.compile(
    r"\b(I can(?:no|')t|I'm sorry|I am sorry|I apologi[sz]e|As an AI|I won't|I will not|"
    r"I'm (?:unable|not able)|I am (?:unable|not able)|I must decline|"
    r"(?:cannot|can't|won't) (?:help|assist|provide|comply|fulfill))",
    re.IGNORECASE,
)


def is_refusal(response: str, prefix: int = 300) -> bool:
    # Refusals open the answer; looking further flags safety caveats inside helpful answers.
    return bool(REFUSAL.search(response[:prefix]))
