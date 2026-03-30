import requests

BASE_URL = "http://localhost:7860"

def run_inference(task_level: str = "easy"):
    obs = requests.post(f"{BASE_URL}/reset?task={task_level}").json()
    
    body = obs["body"].lower()
    subject = obs["subject"].lower()
    combined = body + " " + subject
    tier = obs["customer_tier"]
    sentiment = obs["sentiment_hint"]

    if task_level == "hard" or tier == "enterprise":
        priority = "critical"
    elif task_level == "medium" or sentiment == "angry":
        priority = "high"
    else:
        priority = "low"

    if any(w in combined for w in ["sla", "violation", "legal", "outage"]):
        category = "general"
    elif any(w in combined for w in ["crash", "bug", "error", "breach"]):
        category = "technical"
    elif any(w in combined for w in ["charge", "invoice", "billing", "payment"]):
        category = "billing"
    else:
        category = "account"

    if tier == "enterprise":
        sentiment_response = "formal"
    elif sentiment == "angry":
        sentiment_response = "empathetic"
    else:
        sentiment_response = "neutral"

    action = {
        "priority": priority,
        "category": category,
        "sentiment_response": sentiment_response,
        "response_draft": f"Thank you for contacting support. I understand your concern about '{obs['subject']}' and will resolve this as a {priority} priority issue."
    }

    result = requests.post(
        f"{BASE_URL}/grader?task={task_level}",
        json=action
    ).json()

    return {"task": task_level, "score": result["score"], "breakdown": result["breakdown"]}

if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        print(run_inference(level))