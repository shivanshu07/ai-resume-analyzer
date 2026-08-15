from src.preprocessing.requirement_normalizer import RequirementNormalizer


normalizer = RequirementNormalizer()


text = (
    "Experience with statistical software "
    "(e.g., Python, R or MATLAB) and database "
    "languages (e.g., SQL) with a good understanding "
    "of the AI/ML and statistical methods typically "
    "used in marketing analytics."
)


result = normalizer.normalize_requirement(
    text,
    "preferred",
    "medium"
)


print("\nNORMALIZED REQUIREMENT")
print("=" * 60)

print(
    "Original:",
    result["original_text"]
)

print(
    "Category:",
    result["category"]
)

print(
    "Importance:",
    result["importance"]
)

print(
    "Skills:",
    result["skills"]
)

print(
    "Experience:",
    result["experience"]
)