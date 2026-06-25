import { resolve, join } from "node:path"
import { readdirSync } from "node:fs"
import { readFile, writeFile } from "node:fs/promises"

/**
 * 递归迭代某个目录中的所有文件，每次 yield 一个文件的绝对路径
 */
function* DirIterator(dir: string): Generator<string> {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const fullPath = join(dir, entry.name)
        if (entry.isDirectory()) {
            yield* DirIterator(fullPath)
        } else if (entry.isFile()) {
            yield fullPath
        }
    }
}

/**
 * 规范化 markdown 标题前后的空行：
 * - 标题行下方紧挨着的空行全部移除，让标题直接接住后面的内容
 * - 标题行上方若不是空行（且不在文件开头），补一个空行与上方内容隔开
 * 围栏代码块内部不做任何处理，避免误伤代码里的 # 注释
 */
function normalizeMdFileHeader(text: string) {
    const lines = text.split('\n')
    const out: string[] = []
    let inCode = false
    let lastWasHeader = false
    for (const line of lines) {
        // 围栏代码块用 ``` 或 ~~~ 开/闭，状态切换后原样保留
        if (/^\s*(`{3,}|~{3,})/.test(line)) {
            inCode = !inCode
            out.push(line)
            lastWasHeader = false
            continue
        }
        if (inCode) {
            out.push(line)
            lastWasHeader = false
            continue
        }
        // 上一行刚是标题：吃掉其后紧挨着的空行
        if (lastWasHeader && line.trim() === '') {
            continue
        }
        const isHeader = /^#{1,6}(\s|$)/.test(line)
        if (isHeader) {
            // 标题上方保证至少一个空行（文件开头除外）
            if (out.length > 0 && out[out.length - 1] !== '') {
                out.push('')
            }
            out.push(line)
            lastWasHeader = true
        } else {
            out.push(line)
            lastWasHeader = false
        }
    }
    return out.join('\n')
}

async function normalizeMdHeader(dir: string) {
    const queue: Promise<void>[] = []
    for (const filePath of DirIterator(dir)) {
        if (!filePath.endsWith('.md')) continue
        const task = (async () => {
            const text = await readFile(filePath, 'utf-8')
            const normalized = normalizeMdFileHeader(text)
            // 只在内容确实变化时才写回，避免无谓的文件改动
            if (normalized !== text) {
                await writeFile(filePath, normalized)
                console.log('normalized:', filePath)
            }
        })().catch((e) => {
            console.log('fail to normalize file:', filePath)
            console.log(e)
        })
        queue.push(task)
    }
    await Promise.all(queue)
}

async function main() {
    await normalizeMdHeader(resolve(import.meta.dirname, '..', 'src'))
}

main().catch((e) => {
    console.log('fail to run lint')
    console.log(e)
})
