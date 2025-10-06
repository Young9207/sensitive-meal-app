from typing import Dict, Any, Tuple, List, Optional, Iterable, Union
from dataclasses import dataclass

CORE_NUTRIENTS = ['Protein', 'LightProtein', 'ComplexCarb', 'HealthyFat', 'Fiber', 'VitaminA', 'VitaminB', 'VitaminC', 'VitaminD', 'VitaminE', 'VitaminK', 'Iron', 'Magnesium', 'Omega3', 'Potassium', 'Iodine', 'Calcium', 'Hydration', 'Circulation']
ESSENTIALS = ['Protein', 'ComplexCarb', 'Fiber', 'VitaminB', 'VitaminC', 'VitaminA', 'VitaminK', 'Magnesium', 'Omega3', 'Potassium', 'HealthyFat', 'VitaminD']
ALIAS_MAP = {'a': 'VitaminA', 'b': 'VitaminB', 'c': 'VitaminC', 'd': 'VitaminD', 'e': 'VitaminE', 'k': 'VitaminK', 'vitamina': 'VitaminA', 'vitaminb': 'VitaminB', 'vitaminc': 'VitaminC', 'vitamind': 'VitaminD', 'vitamine': 'VitaminE', 'vitamink': 'VitaminK', 'fe': 'Iron', 'iron': 'Iron', 'mg': 'Magnesium', 'magnesium': 'Magnesium', 'k_potassium': 'Potassium', 'kpotassium': 'Potassium', 'potassium': 'Potassium', 'ca': 'Calcium', 'calcium': 'Calcium', 'iodine': 'Iodine', 'protein': 'Protein', 'lightprotein': 'LightProtein', 'complexcarb': 'ComplexCarb', 'healthyfat': 'HealthyFat', 'fiber': 'Fiber', 'omega3': 'Omega3', 'hydration': 'Hydration', 'circulation': 'Circulation'}
CANONICAL_SET = set(CORE_NUTRIENTS)

def normalize_key(key: str) -> str:
    k = key.strip().replace(" ", "").replace("-", "").replace("_", "").lower()
    return ALIAS_MAP.get(k, key)

@dataclass
class AssessmentResult:
    missing: List[str]
    insufficient: List[Tuple[str, Any, Any]]
    ok: List[str]
    unknown: List[str]
    not_recognized: List[str]

def assess_intake(intake: Dict[str, Any], targets: Optional[Dict[str, Union[int, float]]] = None, check_list: Iterable[str] = ESSENTIALS) -> AssessmentResult:
    normalized_intake = {}
    not_recognized = []
    for k, v in intake.items():
        canon = normalize_key(k)
        if canon in CANONICAL_SET:
            normalized_intake[canon] = v
        else:
            not_recognized.append(k)
    missing = []
    insufficient = []
    ok = []
    unknown = []
    for key in check_list:
        if key not in normalized_intake:
            missing.append(key)
            continue
        val = normalized_intake[key]
        if targets and key in targets and isinstance(targets[key], (int, float)):
            tgt = targets[key]
            if isinstance(val, (int, float)):
                if val < tgt:
                    insufficient.append((key, val, tgt))
                else:
                    ok.append(key)
            else:
                unknown.append(key)
        else:
            if bool(val):
                ok.append(key)
            else:
                insufficient.append((key, val, None))
    return AssessmentResult(missing, insufficient, ok, unknown, not_recognized)

DEFAULT_TARGETS = {
    "Protein": 60,
    "Fiber": 25,
    "Omega3": 1.1,
    "Magnesium": 310,
    "Potassium": 3500,
    "Calcium": 1000,
    "Iron": 18,
    "VitaminC": 75,
    "VitaminD": 15,
    "VitaminA": 700,
    "VitaminE": 15,
    "VitaminK": 90,
}

def find_deficiencies(intake: Dict[str, Any], targets: Optional[Dict[str, Union[int, float]]] = None, essentials_only: bool = True):
    result = assess_intake(intake=intake, targets=targets if targets is not None else DEFAULT_TARGETS, check_list=ESSENTIALS if essentials_only else CORE_NUTRIENTS)
    return {
        "missing": result.missing,
        "insufficient": [
            {"nutrient": k, "value": v, "target": t} for (k, v, t) in result.insufficient
        ],
        "ok": result.ok,
        "unknown": result.unknown,
        "not_recognized": result.not_recognized,
    }

def pretty_report(result: AssessmentResult) -> str:
    lines = []
    if result.missing:
        lines.append("❌ Missing (not reported): " + ", ".join(result.missing))
    if result.insufficient:
        def fmt(item):
            k, v, t = item
            if t is None:
                return f"{k} (value={v})"
            return f"{k} (value={v}, target={t})"
        lines.append("⚠️ Insufficient: " + ", ".join(fmt(i) for i in result.insufficient))
    if result.ok:
        lines.append("✅ OK: " + ", ".join(result.ok))
    if result.unknown:
        lines.append("❓ Unknown/Uncomparable: " + ", ".join(result.unknown))
    if result.not_recognized:
        lines.append("ℹ️ Not recognized keys ignored: " + ", ".join(result.not_recognized))
    return "\n".join(lines) if lines else "No data to assess."
