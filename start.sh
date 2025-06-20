#!/usr/bin/env bash
source .env
python -c "from agent.agent import chat; chat()"
