# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a technical documentation website ("Spark Notes") built with VitePress. It's a personal knowledge base containing computer science learning notes organized by topic categories.

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

### Configuration System
The sidebar navigation is auto-generated from the file structure using custom logic in `.vitepress/config.ts`. Each directory's `index.md` file defines its section title and order via YAML frontmatter:
```yaml
---
title: Section Title
order: 1
---

# Article Title ≠ Section Title
```

The yaml frontmatter must be write at the first line of the file before the first-level title.

### Build Configuration
- Uses VitePress with custom Vite plugin that adds a server restart button
- MathJax3 enabled for mathematical expressions
- Output directory: `docs/` (used by GitHub Pages)
- Clean URLs enabled for SEO

### Key Files
- `.vitepress/config.ts` - Main configuration, auto-sidebar generation logic
- `.github/workflows/deploy.yml` - CI/CD pipeline for GitHub Pages deployment
- `package.json` - Minimal dependencies (VitePress, TypeScript, MathJax3)

## Writing Style Guidelines

To maintain a consistent style across the site, follow these language rules when writing documentation:

1. **Reduce "AI-generated" tone** - Write in the voice of a technical professional recording personal learning insights, emphasizing unique perspectives and deep understanding of the subject matter.

2. **Minimize bold formatting** - Only use `****` for bold text on a few core concept keywords per article.

3. **Limit heading depth** - Use at most three levels of headings (`###`).

4. **Reduce short content sections**:
   - Reduce headings with less than 40 characters of content
   - Reduce unordered list items with less than 20 characters, ordered list items can ignore this rule
   - When such content appears, consider combining it into longer sentence paragraphs

   Example:
   ```
   Not recommended:
   学习的好处：
   + 增加见识；
   + 锻炼解决问题的能力；
   + 打发时间；

   Recommended:
   学习可以帮助我们增加见识，同时让我们在打发时间之余锻炼解决问题的能力。
   ```

5. **Focus on practical application** - Emphasize the relationship between knowledge and actual engineering practice, including guidance for engineering and summaries of engineering experience.

6. **No blank line after headings but one blank line before headings** - In markdown, do not leave blank lines between `#` headings and their content. Leave a blank line before headings.