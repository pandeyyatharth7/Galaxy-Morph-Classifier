# ============================================================
# ENVIRONMENT & PERCEPTS
# ============================================================

# The environment provides uncertain observations (percepts)
galaxy_percept = {
    "P_EL": 0.72,     # Probability of Elliptical galaxy
    "P_CW": 0.15,     # Probability of Clockwise Spiral
    "P_ACW": 0.10,    # Probability of Anti-clockwise Spiral
    "EDGE_ON": 0.05,  # Edge-on likelihood
    "MERGER": 0.03    # Merger likelihood
}

# ============================================================
# KNOWLEDGE REPRESENTATION
# (Explicit symbolic + probabilistic rules)
# ============================================================

rules = [
    {
        "conditions": {"P_EL": ">0.6", "P_CW": "<0.2", "P_ACW": "<0.2"},
        "conclusion": "Elliptical",
        "confidence": 0.7
    },
    {
        "conditions": {"P_CW": ">0.4"},
        "conclusion": "Spiral",
        "confidence": 0.7
    },
    {
        "conditions": {"MERGER": ">0.3"},
        "conclusion": "Uncertain",
        "confidence": 0.9
    }
]

# ============================================================
# UTILITY: CONDITION CHECKER
# ============================================================

def check_condition(value, condition):
    """
    Evaluates symbolic conditions like >0.6 or <0.2
    """
    operator = condition[0]
    threshold = float(condition[1:])

    if operator == ">":
        return value > threshold
    elif operator == "<":
        return value < threshold
    return False

# ============================================================
# REASONING ENGINE (Inference under uncertainty)
# ============================================================

def reasoning_engine(percepts, rule_base):
    """
    Performs symbolic reasoning.
    Computes belief values for each hypothesis.
    """
    beliefs = {
        "Spiral": 0.0,
        "Elliptical": 0.0,
        "Uncertain": 0.0
    }

    fired_rules = []

    for rule in rule_base:
        satisfied = True

        for feature, condition in rule["conditions"].items():
            if not check_condition(percepts[feature], condition):
                satisfied = False
                break

        if satisfied:
            beliefs[rule["conclusion"]] += rule["confidence"]
            fired_rules.append(rule)

    return beliefs, fired_rules

# ============================================================
# DECISION MAKING (Rational Agent)
# ============================================================

def decide_action(beliefs, threshold=0.6):
    """
    Selects the best action based on maximum belief.
    """
    best_class = max(beliefs, key=beliefs.get)

    if beliefs[best_class] < threshold:
        return "Uncertain"

    return best_class

# ============================================================
# LEARNING MODULE (Feedback-Based Learning)
# ============================================================

def learn_from_feedback(rules, fired_rules, true_label, learning_rate=0.1):
    """
    Updates rule confidence values based on feedback.
    Implements learning in an AI agent.
    """
    for rule in rules:
        if rule in fired_rules:
            if rule["conclusion"] == true_label:
                # Reward correct reasoning
                rule["confidence"] = min(1.0, rule["confidence"] + learning_rate)
            else:
                # Penalize incorrect reasoning
                rule["confidence"] = max(0.0, rule["confidence"] - learning_rate)

# ============================================================
# INTELLIGENT AGENT LOOP
# ============================================================
 
def intelligent_agent(percepts, true_label=None):
    """
    Agent perceives, reasons, decides, and learns.
    """
    beliefs, fired_rules = reasoning_engine(percepts, rules)
    decision = decide_action(beliefs)

    # Learning occurs only if feedback is available
    if true_label is not None:
        learn_from_feedback(rules, fired_rules, true_label)

    return beliefs, decision

# ============================================================
# DEMONSTRATION
# ============================================================

# Simulated expert feedback (ground truth)
true_label = "Elliptical"

beliefs, decision = intelligent_agent(galaxy_percept, true_label=true_label)

#print("Percepts:", galaxy_percept)
#print("\nBelief State:", beliefs)
#print("Final Decision:", decision)
#print("\nUpdated Rule Confidences:")
#for rule in rules:
    #print(f"{rule['conclusion']} → {rule['confidence']:.2f}")
