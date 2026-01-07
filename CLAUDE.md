# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a technical documentation website ("Spark Note") built with VitePress. It's a personal knowledge base containing computer science learning notes organized by topic categories.

## Commands

### Development
```bash
# Install dependencies
pnpm install

# Start development server (starts at localhost:5173)
pnpm dev

# Build static site to docs/ directory
pnpm build

# Preview built site locally
pnpm preview
```

### Deployment
The site automatically deploys to GitHub Pages via GitHub Actions when pushing to the master branch.

## Architecture

### Content Organization
All documentation content is in `src/` directory, organized by topic:
- `ai/` - AI/ML topics (LLM, neural networks, PyTorch)
- `basic/` - CS fundamentals (algorithms, data structures, OS, networking)
- `client/` - Frontend technologies (JavaScript, React, Vue, TypeScript)
- `server/` - Backend technologies (Java, Go, databases, distributed systems)
- `design/` - Software design patterns and architecture
- `devops/` - DevOps tools (Docker, Kubernetes, Git)
- `kernel/` - Low-level systems (Linux kernel, C, assembly)
- `web3/` - Blockchain technologies
- `other/` - Miscellaneous topics

### Configuration System
The sidebar navigation is auto-generated from the file structure using custom logic in `.vitepress/config.ts`. Each directory's `index.md` file defines its section title and order via YAML frontmatter:
```yaml
---
title: Section Title
order: 1
---
```

### Build Configuration
- Uses VitePress with custom Vite plugin that adds a server restart button
- MathJax3 enabled for mathematical expressions
- Output directory: `docs/` (used by GitHub Pages)
- Clean URLs enabled for SEO

### Key Files
- `.vitepress/config.ts` - Main configuration, auto-sidebar generation logic
- `.github/workflows/deploy.yml` - CI/CD pipeline for GitHub Pages deployment
- `package.json` - Minimal dependencies (VitePress, TypeScript, MathJax3)