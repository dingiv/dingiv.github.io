# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a VitePress-based technical documentation site written in Chinese (漫话随笔 - "Casual Notes"), covering computer science topics including algorithms, operating systems, networking, databases, frontend/backend development, and more.

## Development Commands

```bash
# Install dependencies (using pnpm)
pnpm install

# Start development server with hot reload
pnpm dev

# Build static site (outputs to docs/)
pnpm build

# Preview production build
pnpm preview
```

## Architecture

### Content Organization

All documentation content lives in `src/` and is organized by topic:

- `basic/` - Computer science fundamentals (algorithms, OS, networking, databases, graphics, hardware, compilers)
- `design/` - Software design (OOP, FP, reactive programming, concurrency patterns)
- `client/` - Client-side development (Rust, Dart/Flutter, rendering, shader languages)
- `server/` - Server-side development (Go, Java, SQL/NoSQL, architecture)
- `devops/` - DevOps topics (Git, Docker, Kubernetes, CI/CD, shell scripting)
- `kernel/` - Low-level programming (C, Assembly, Linux kernel, embedded systems)
- `ai/` - Artificial intelligence and machine learning
- `web3/` - Blockchain and Web3 technologies
- `other/` - Miscellaneous topics
- `public/` - Static assets (images, files) that will be copied to the site root

### Content Structure

Each topic directory follows this structure:
- `index.md` - Contains frontmatter with `title` and `order` fields, acts as the section homepage
- Topic-specific markdown files with optional frontmatter for title and ordering

### VitePress Configuration

**Location**: `.vitepress/config.ts`

Key architectural features:

1. **Dynamic Sidebar Generation**: The `createSidebarItem()` function recursively scans `src/` to build the sidebar navigation tree
   - Reads frontmatter from each `index.md` to get section titles and ordering
   - Supports `order` field for custom sorting (defaults to 9999)
   - Automatically collapses nested sections

2. **Frontmatter Parsing**: The `readFrontMatter()` function extracts metadata from markdown files
   - Supports YAML frontmatter format (`---\ntitle: ...\norder: ...\n---`)
   - Falls back to parsing the first heading if no frontmatter exists
   - Used for both sidebar text and ordering

3. **Custom Vite Plugin**: `vitePluginRestart()` adds a restart button to the dev server
   - Accessible at fixed position (top-left corner)
   - Triggers server restart via `/__vite_plugin_restart` endpoint
   - Useful for configuration changes that require full restart

4. **Build Output**: Configured to output to `docs/` directory (likely for GitHub Pages deployment)

### Theme

**Location**: `.vitepress/theme.ts`

Currently uses the default VitePress theme with no customizations.

## Key Technical Details

- **Language**: Chinese (zh-CN)
- **Framework**: VitePress 1.6.4+
- **Math Support**: LaTeX math rendering via `markdown-it-mathjax3`
- **Clean URLs**: Enabled (removes .html extensions)
- **Module System**: ES Modules (type: "module" in package.json)
- **TypeScript**: ESNext target, bundler module resolution

## Working with Content

When adding or modifying documentation:

1. Place markdown files in the appropriate `src/` subdirectory
2. Add frontmatter to control sidebar ordering:
   ```yaml
   ---
   title: 文章标题
   order: 10
   ---
   ```
3. Lower `order` values appear first in the sidebar
4. Each directory should have an `index.md` that serves as the section landing page
5. Math expressions are supported using LaTeX syntax

## Deployment

The build output goes to `docs/`, suggesting this site is deployed via GitHub Pages (common pattern for `username.github.io` repos).
