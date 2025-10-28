#!/bin/bash
# GitHub Repository Configuration Script for Tantra-LLM
# Run this to set up repository settings via GitHub CLI

# Description
echo "Setting repository description..."
gh api repos/atulyaai/Tantra-LLM -X PATCH -F description="🧠 Proprietary Multimodal Cognitive Architecture - SpikingBrain + Long-VITA + Whisper with dynamic context, fusion layers, hybrid memory, and adaptive personality. Not a chatbot. A cognitive mind."

# Topics/Tags
echo "Setting repository topics..."
gh api repos/atulyaai/Tantra-LLM -X PUT \
  -H 'Accept: application/vnd.github.mercy-preview+json' \
  -F names='["tantra-llm","multimodal-ai","spikingbrain","cognitive-architecture","fusion-layers","adaptive-personality","hybrid-memory","proprietary-ai","reasoning-system","neural-architecture"]'

# Homepage URL
echo "Setting homepage URL..."
gh api repos/atulyaai/Tantra-LLM -X PATCH -F homepage="https://github.com/atulyaai/Tantra-LLM"

# Default branch to main
echo "Setting default branch to main..."
gh api repos/atulyaai/Tantra-LLM -X PATCH -F default_branch="main"

echo "Repository configuration complete!"

