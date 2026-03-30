import random
from pydantic import BaseModel

# --- Typed Models ---

class Observation(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: str
    previous_contacts: int
    sentiment_hint: str
    task_level: str

class Action(BaseModel):
    priority: str
    category: str
    sentiment_response: str
    response_draft: str

class Reward(BaseModel):
    score: float
    breakdown: dict

# --- Ticket Bank ---

TICKETS = {
    "easy": [
        {
            "subject": "How do I reset my password?",
            "body": "Hi, I forgot my password and need help resetting it. Thanks!",
            "customer_tier": "free",
            "previous_contacts": 0,
            "sentiment_hint": "polite",
            "answer": {"priority": "low", "category": "account", "sentiment_response": "neutral"}
        },
        {
            "subject": "Where can I find my invoice?",
            "body": "Hello, I need to download my invoice for last month.",
            "customer_tier": "pro",
            "previous_contacts": 1,
            "sentiment_hint": "neutral",
            "answer": {"priority": "low", "category": "billing", "sentiment_response": "neutral"}
        },
    ],
    "medium": [
        {
            "subject": "App keeps crashing on login",
            "body": "Your app crashes every time I try to log in on iPhone 14. I have tried reinstalling but the problem persists. This is very frustrating.",
            "customer_tier": "pro",
            "previous_contacts": 2,
            "sentiment_hint": "angry",
            "answer": {"priority": "high", "category": "technical", "sentiment_response": "empathetic"}
        },
        {
            "subject": "Charged twice this month",
            "body": "I see two charges on my card this month but I only have one subscription. Please fix this immediately.",
            "customer_tier": "pro",
            "previous_contacts": 0,
            "sentiment_hint": "angry",
            "answer": {"priority": "high", "category": "billing", "sentiment_response": "empathetic"}
        },
    ],
    "hard": [
        {
            "subject": "Data breach concern - urgent",
            "body": "We are an enterprise client and our security team flagged suspicious activity on our account. We need immediate escalation and a full audit log. This may be a compliance issue.",
            "customer_tier": "enterprise",
            "previous_contacts": 5,
            "sentiment_hint": "angry",
            "answer": {"priority": "critical", "category": "technical", "sentiment_response": "formal"}
        },
        {
            "subject": "SLA violation - 3rd time this week",
            "body": "This is the third outage this week violating our SLA agreement. We are considering legal action if this is not resolved and compensated within 24 hours.",
            "customer_tier": "enterprise",
            "previous_contacts": 8,
            "sentiment_hint": "angry",
            "answer": {"priority": "critical", "category": "general", "sentiment_response": "formal"}
        },
    ]
}

# --- Environment ---

class SupportTriageEnv:

    def __init__(self, task_level: str = "easy"):
        self.task_level = task_level
        self.current_ticket = None
        self.steps_taken = 0
        self.max_steps = 10
        self.done = False

    def reset(self) -> Observation:
        self.steps_taken = 0
        self.done = False
        ticket = random.choice(TICKETS[self.task_level])
        self.current_ticket = ticket
        return Observation(
            ticket_id=f"TKT-{random.randint(1000,9999)}",
            subject=ticket["subject"],
            body=ticket["body"],
            customer_tier=ticket["customer_tier"],
            previous_contacts=ticket["previous_contacts"],
            sentiment_hint=ticket["sentiment_hint"],
            task_level=self.task_level
        )

    def step(self, action: Action):
        self.steps_taken += 1
        reward, breakdown = self._compute_reward(action)
        self.done = self.steps_taken >= self.max_steps
        obs = self.reset() if not self.done else None
        return obs, reward, self.done, {"breakdown": breakdown, "steps": self.steps_taken}

    def state(self) -> dict:
        return {
            "task_level": self.task_level,
            "steps_taken": self.steps_taken,
            "done": self.done,
            "current_ticket": self.current_ticket
        }

    def _compute_reward(self, action: Action) -> tuple:
        answer = self.current_ticket["answer"]
        breakdown = {}

        priority_score = 1.0 if action.priority == answer["priority"] else 0.0
        breakdown["priority"] = priority_score

        category_score = 1.0 if action.category == answer["category"] else 0.0
        breakdown["category"] = category_score

        sentiment_score = 1.0 if action.sentiment_response == answer["sentiment_response"] else 0.0
        breakdown["sentiment"] = sentiment_score

        response_len = len(action.response_draft.split())
        if response_len >= 30:
            response_score = 1.0
        elif response_len >= 15:
            response_score = 0.5
        else:
            response_score = 0.1
        breakdown["response_quality"] = response_score

        total = (
            0.4 * priority_score +
            0.3 * category_score +
            0.2 * sentiment_score +
            0.1 * response_score
        )
        breakdown["total"] = round(total, 3)
        return round(total, 3), breakdown