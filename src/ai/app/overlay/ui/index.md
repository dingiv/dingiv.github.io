---
title: AI UI
---

# 前端 AI 开发
前端是 LLM 应用与用户交互的第一界面，用户体验的成败很大程度上取决于前端的实现质量。传统前端是"用户点击 → 后端处理 → 渲染结果"的模式，AI 时代则演变为"用户提问 → 流式生成 → 动态渲染"的新范式。这种转变带来了技术栈的升级，也打开了交互创新的空间。

## 流式渲染
LLM 的输出不是一次性完成的，而是逐个 token 生成的。如果等待完整响应再渲染，用户会感到卡顿和延迟。流式渲染让前端能够实时接收并展示生成过程，类似打字机效果，提升交互体验。

Server-Sent Events (SSE) 是实现流式传输的标准协议。服务端保持连接开放，持续发送数据块，前端通过 `EventSource` 接收。与 WebSocket 不同，SSE 是单向的（服务端到前端），更适合 LLM 响应这种不需要客户端频繁发送的场景。

```javascript
const eventSource = new EventSource('/api/chat?message=' + encodeURIComponent(input));
eventSource.onmessage = (event) => {
    const chunk = JSON.parse(event.data);
    appendToChat(chunk.content); // 逐字追加到对话区域
};
eventSource.onerror = () => eventSource.close();
```

Vercel AI SDK 的 `useChat` hook 封装了流式处理的复杂性。它自动管理 SSE 连接、处理重连、累积消息历史、更新 UI 状态。开发者只需要提供 `api` 路径和回调函数，就能得到一个完整的聊天界面。

```javascript
import { useChat } from 'ai/react';

function Chat() {
    const { messages, input, handleInputChange, handleSubmit } = useChat({
        api: '/api/chat'
    });

    return (
        <div>
            {messages.map(m => <div key={m.id}>{m.content}</div>)}
            <form onSubmit={handleSubmit}>
                <input value={input} onChange={handleInputChange} />
            </form>
        </div>
    );
}
```

流式渲染的细节处理很重要。代码块需要语法高亮（Prism.js、Shiki），公式需要 LaTeX 渲染（KaTeX、MathJax），这些库通常不支持增量更新，需要在代码块闭合后一次性渲染。Markdown 解析可以增量处理，但需要处理不完整的语法（如代码块未闭合），用临时的占位符避免布局错乱。

## 生成式 UI
生成式 UI（Generative UI）是前端 AI 的前沿方向。传统 AI 应用返回的是文本，用户阅读后自行操作；生成式 UI 让 AI 根据内容类型，动态渲染对应的交互组件。AI 说"今天北京晴天，25°C"，前端不只是显示文字，而是渲染一个天气卡片，包含温度图标、详细数据、未来几天预报。

Vercel AI SDK 的 `streamUI` 函数支持这种模式。服务端可以返回 React 组件而非纯文本，前端接收后动态挂载。比如 AI 决定展示股票信息，服务端返回 `<StockCard symbol="AAPL" />`，前端渲染这个组件，用户可以直接在卡片内操作（买入、卖出、查看详情）。

```javascript
import { streamUI } from 'ai/rsc';

async function submitMessage(message) {
    const ui = await streamUI({
        model: openai('gpt-4o'),
        messages: [{ role: 'user', content: message }],
        text: ({ content }) => <div>{content}</div>,
        tools: {
            showWeather: {
                description: '显示天气信息',
                parameters: z.object({ city: z.string() }),
                generate: async ({ city }) => {
                    const data = await getWeather(city);
                    return <WeatherCard data={data} />;
                }
            }
        }
    });
    return ui.value;
}
```

生成式 UI 的挑战是组件设计和状态管理。每个可被 AI 调用的组件都需要清晰的用途和参数定义，类似工具调用的 JSON Schema。组件应该自包含，不依赖外部状态，因为 AI 无法理解复杂的前端状态管理。组件交互的结果（用户点击股票卡片的"买入"按钮）需要传递给 AI，让 AI 知道发生了什么，继续对话流程。

## 结构化数据校验
AI 返回的是字符串，但前端组件需要结构化数据。Zod 是运行时类型校验库，可以定义 Schema 并解析数据，确保 AI 返回的 JSON 符合前端预期。这在复杂场景（表格、图表、表单）尤为重要，格式错误的输入会导致渲染崩溃或安全漏洞。

```typescript
import { z } from 'zod';

const ChartDataSchema = z.object({
    type: z.enum(['line', 'bar', 'pie']),
    title: z.string(),
    data: z.array(z.object({
        label: z.string(),
        value: z.number()
    }))
});

function renderChart(aiResponse: string) {
    const parsed = ChartDataSchema.parse(JSON.parse(aiResponse));
    // 如果 AI 返回的数据不符合 Schema，这里会抛出错误
    return <Chart type={parsed.type} data={parsed.data} title={parsed.title} />;
}
```

Zod 的优势是类型推导，Schema 定义后可以自动生成 TypeScript 类型，开发时获得完整的类型检查。TypeChat 是微软推出的方案，它先让 AI 返回符合 TypeScript 类型的 JSON，再用类型检查验证，失败时让 AI 重新生成，迭代直到通过。这种"类型驱动"的方式提升了 AI 返回的可靠性。

## 浏览器端 AI
随着 WebGPU 和 WebAssembly 的成熟，越来越多的 AI 任务可以直接在浏览器中运行，无需后端支持。Transformers.js 是 Hugging Face 推出的 JavaScript 版本 Transformers 库，支持在浏览器中运行 BERT、Whisper、Llama 等模型。

浏览器端 AI 的优势是隐私保护和延迟降低。用户的语音、图片、文本无需上传到服务器，在本地完成处理，适合医疗、金融等敏感场景。实时交互（语音识别、手势识别）的延迟降到毫秒级，无需网络往返。劣势是性能受限于设备，高端笔记本可以运行 7B 模型，但手机端只能处理更小的模型。

```javascript
import { pipeline } from '@xenova/transformers';

const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny');
const audio = await fetch('audio.mp3');
const output = await transcriber(await audio.blob());
console.log(output.text); // "你好，世界"
```

WebGPU 是浏览器访问 GPU 的标准接口，相比 WebGL 更适合通用计算。2026 年起，Chrome、Firefox、Safari 都已支持 WebGPU，让浏览器能够利用 GPU 加速 AI 推理。WebNN 是另一个正在制定的标准，专门针对神经网络推理，提供更高效的算子实现。这些技术让浏览器端的 AI 能力持续提升，未来很多轻量级任务（图像分类、文本摘要、语音助手）都能在本地完成。

## 状态管理新挑战
AI 对话的状态管理比传统应用更复杂。用户可能"重新生成"让 AI 换一种说法，这需要保留原始问题，创建新的分支。用户可能"编辑消息"修改之前的输入，这需要重新执行后续对话，更新所有相关的消息。多轮对话的历史越来越长，如何在保持上下文的同时控制内存占用？

对话树是管理分支对话的数据结构。每个消息节点可以有多个子节点，代表不同的生成结果。用户切换分支时，沿着树路径回溯，重新加载对应的消息序列。持久化时需要保存完整的树结构，而不仅仅是线性历史。

乐观更新提升交互体验。用户点击发送时，立即在前端展示用户消息，不等待后端响应。AI 的回答也先显示占位符（骨架屏、点点动画），流式接收后逐步替换。如果请求失败，展示错误提示并提供重试按钮。这种"先假设成功，失败后回滚"的策略让应用感觉更快速。

多模态输入处理是 AI 应用的常见需求。用户上传图片（拍照、文件选择），前端需要预览、压缩（降低传输成本）、转码（Base64 或 Blob）。语音输入需要录音、降噪、转文字（可以用浏览器端的 Whisper）。文件上传需要切片，大文件分块上传并显示进度条。这些处理都应该在客户端完成，减少后端负担。

## 开发工具链
AI 前端开发有专门的框架和工具。Vercel AI SDK 是当前最成熟的方案，支持 React、Next.js、Vue、Svelte，提供 `useChat`、`useCompletion`、`streamUI` 等核心 hooks。LangChain.js 是 LangChain 的 JavaScript 版本，支持在浏览器中构建 LLM 应用，适合需要更灵活控制的场景。

AI SDK 的 `useCompletion` 适合文本补全场景，用户输入前半句，AI 补全后半句。`useChat` 适合对话场景，管理完整的消息历史和流式响应。两者都支持自定义的中间件，可以在请求发送前修改 Prompt，在响应接收后处理数据，实现日志记录、缓存、重试等横切关注点。

调试工具也在演进。React DevTools 可以查看 AI 相关的组件状态，但流式更新的特性让调试更困难。LangSmith、LangFuse 提供的前端 SDK 可以记录用户交互，将前端事件与后端 LLM 调用关联，完整的链路追踪帮助定位问题。

前端 AI 开发正在重塑用户交互的范式。从 GUI（图形用户界面）到 LUI（语言用户界面），从固定布局到动态生成，从客户端-服务器到边缘计算。前端工程师需要掌握新的工具和思维模式，但同时也获得了更大的创作空间——UI 不再是静态的，而是可以根据用户意图实时生成的。
