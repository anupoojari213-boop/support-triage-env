from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from environment import SupportTriageEnv, Action
import uvicorn

app = FastAPI(title="Support Triage Environment", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env instance per task level
envs = {
    "easy":   SupportTriageEnv("easy"),
    "medium": SupportTriageEnv("medium"),
    "hard":   SupportTriageEnv("hard"),
}

# ── Required endpoints ──────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "env": "support-triage-env"}

@app.post("/reset")
def reset(task: str = "easy"):
    obs = envs[task].reset()
    return obs.model_dump()

@app.post("/step")
def step(action: Action, task: str = "easy"):
    obs, reward, done, info = envs[task].step(action)
    return {
        "observation": obs.model_dump() if obs else None,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state(task: str = "easy"):
    return envs[task].state()

@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "description": "Simple password/billing queries from free-tier customers",
                "difficulty": "easy",
                "action_schema": Action.model_json_schema()
            },
            {
                "id": "medium",
                "description": "Frustrated pro-tier customers with app or billing issues",
                "difficulty": "medium",
                "action_schema": Action.model_json_schema()
            },
            {
                "id": "hard",
                "description": "Enterprise clients with security, SLA, or compliance issues",
                "difficulty": "hard",
                "action_schema": Action.model_json_schema()
            }
        ]
    }

@app.post("/grader")
def grader(action: Action, task: str = "easy"):
    env = envs[task]
    if env.current_ticket is None:
        env.reset()
    score, breakdown = env._compute_reward(action)
    return {"score": score, "breakdown": breakdown}

@app.post("/baseline")
def baseline(task: str = "easy"):
    env = envs[task]
    obs = env.reset()
    scores = []
    for _ in range(3):
        dummy_action = Action(
            priority="high",
            category="technical",
            sentiment_response="empathetic",
            response_draft="Thank you for reaching out. I understand your concern and will help you resolve this issue as quickly as possible. Our team is looking into this right now."
        )
        _, reward, done, _ = env.step(dummy_action)
        scores.append(reward)
        if done:
            break
    return {
        "task": task,
        "scores": scores,
        "average": round(sum(scores)/len(scores), 3)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)