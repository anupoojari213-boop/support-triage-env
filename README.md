---
title: Support Triage Env
emoji: 🎫
colorFrom: gray
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: OpenEnv RL environment for customer support ticket triage
tags:
  - openenv
---

# Support Triage Environment

An OpenEnv-compatible RL environment where AI agents learn to triage customer support tickets. Built for Meta x Scaler Hackathon.

## What it does
An AI agent reads support tickets and must correctly identify:
- **Priority**: low / medium / high / critical
- **Category**: billing / technical / account / general
- **Tone**: empathetic / neutral / formal
- **Response**: a draft reply to the customer

## Action Space
- priority: low | medium | high | critical
- category: billing | technical | account | general
- sentiment_response: empathetic | neutral | formal
- response_draft: string

## Observation Space
- ticket_id, subject, body
- customer_tier: free | pro | enterprise
- previous_contacts: integer
- sentiment_hint: angry | neutral | polite
- task_level: easy | medium | hard

## Tasks
| Task | Description | Difficulty |
|------|-------------|------------|
| easy | Simple queries from free-tier customers | Easy |
| medium | Frustrated pro-tier customers | Medium |
| hard | Enterprise clients with legal/SLA issues | Hard |

## Reward Function
| Component | Weight |
|-----------|--------|
| Priority correct | 40% |
| Category correct | 30% |
| Sentiment correct | 20% |
| Response quality | 10% |

## Baseline Scores
- Easy: 1.0
- Medium: 1.0
- Hard: 1.0
- **Average: 1.0**

## API Endpoints
- GET / — health check
- POST /reset — start new episode
- POST /step — take action
- GET /state — current state
- GET /tasks — list all tasks
- POST /grader — score an action
- POST /baseline — run baseline agent

## Setup
pip install -r requirements.txt
python main.py