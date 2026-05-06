import re
import sympy
from sympy.parsing.latex import parse_latex


def extract_boxed_answer(text):
    """Extract content from the last \\boxed{} in the text."""
    # Find all \boxed{...} patterns, handling nested braces
    results = []
    i = 0
    while i < len(text):
        idx = text.find('\\boxed{', i)
        if idx == -1:
            break
        # Find matching closing brace
        depth = 0
        start = idx + 7  # len('\\boxed{')
        for j in range(start, len(text)):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                if depth == 0:
                    results.append(text[start:j])
                    i = j + 1
                    break
                depth -= 1
        else:
            i = idx + 1
            continue

    if results:
        return results[-1]  # Return last boxed answer
    return None


def normalize_answer(answer):
    """Normalize a math answer string for comparison."""
    if answer is None:
        return None

    answer = str(answer).strip()

    # Remove common LaTeX wrappers
    answer = answer.replace('\\text{', '').replace('\\mathrm{', '').replace('\\textbf{', '')
    answer = answer.replace('\\left', '').replace('\\right', '')
    answer = answer.replace('\\,', ' ').replace('\\ ', ' ')
    answer = answer.replace('\\!', '')
    answer = answer.replace('$', '').replace('\\$', '')
    answer = answer.replace('\\dfrac', '\\frac')
    answer = answer.replace('\\tfrac', '\\frac')

    # Remove trailing periods, commas
    answer = answer.rstrip('.,;')

    # Handle percentage
    if answer.endswith('\\%') or answer.endswith('%'):
        answer = answer.replace('\\%', '').replace('%', '').strip()

    # Collapse whitespace: remove spaces around commas, parens, and collapse runs
    answer = re.sub(r'\s+', ' ', answer).strip()
    answer = re.sub(r'\s*,\s*', ',', answer)
    answer = re.sub(r'\(\s+', '(', answer)
    answer = re.sub(r'\s+\)', ')', answer)

    # Remove unmatched closing braces left from stripping \text{...} etc
    while answer.endswith('}') and answer.count('}') > answer.count('{'):
        answer = answer[:-1]

    answer = answer.strip()

    return answer


def is_equiv(ground_truth, prediction):
    """Check if two math answers are equivalent."""
    if ground_truth is None or prediction is None:
        return False

    gt = normalize_answer(ground_truth)
    pred = normalize_answer(prediction)

    if gt is None or pred is None:
        return False

    # Direct string match
    if gt == pred:
        return True

    # Try numeric comparison
    try:
        gt_float = float(gt.replace(',', ''))
        pred_float = float(pred.replace(',', ''))
        if abs(gt_float - pred_float) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    # Try fraction comparison (cross-format: one may be decimal, other fraction)
    try:
        gt_num = parse_fraction(gt)
        pred_num = parse_fraction(pred)
        if gt_num is None:
            try:
                gt_num = float(gt.replace(',', ''))
            except ValueError:
                pass
        if pred_num is None:
            try:
                pred_num = float(pred.replace(',', ''))
            except ValueError:
                pass
        if gt_num is not None and pred_num is not None:
            if abs(float(gt_num) - float(pred_num)) < 1e-6:
                return True
    except:
        pass

    # Try sympy symbolic comparison
    try:
        gt_sym = parse_math_expr(gt)
        pred_sym = parse_math_expr(pred)
        if gt_sym is not None and pred_sym is not None:
            diff = sympy.simplify(gt_sym - pred_sym)
            if diff == 0:
                return True
    except:
        pass

    return False


def parse_fraction(s):
    """Parse a fraction string like '3/4' or '\\frac{3}{4}'."""
    # Handle \frac{a}{b}
    frac_match = re.match(r'\\frac\{([^}]+)\}\{([^}]+)\}', s)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            return num / den
        except ValueError:
            pass

    # Handle a/b
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            try:
                return float(parts[0].strip()) / float(parts[1].strip())
            except ValueError:
                pass

    return None


def parse_math_expr(s):
    """Try to parse a math expression using sympy."""
    try:
        return parse_latex(s)
    except:
        pass

    try:
        return sympy.sympify(s)
    except:
        pass

    return None


def check_correctness(model_response, ground_truth):
    """Check if a model response contains the correct answer.
    Returns (is_correct: bool, extracted_answer: str or None)
    """
    extracted = extract_boxed_answer(model_response)
    if extracted is None:
        # Try to find answer without boxed
        # Look for "answer is X" or "= X" at end
        patterns = [
            r'(?:the answer is|answer:|final answer:?)\s*(.+?)(?:\.|$)',
            r'=\s*(.+?)(?:\.|$)',
        ]
        for pat in patterns:
            match = re.search(pat, model_response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                break

    if extracted is None:
        return False, None

    correct = is_equiv(ground_truth, extracted)
    return correct, extracted
