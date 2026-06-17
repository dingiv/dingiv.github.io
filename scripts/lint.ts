import { resolve } from "node:path"
import { readFile } from "node:fs/promises"

/**
 * 递归迭代某个目录中的所有文件，每次返回的对象是一个文件的绝对路径
 */
function* DirIterator(dir: string) {



    return null as any
}

function normalizeMdFileHeader(text: string) {
    /* 让每个二级、三级、四级小标题前保持有且只有一个空行 */
    const s1 = text.replaceAll(/\n(?=\#\#+)/g, '')
    /* 让每级标题的下面紧接内容，删除存在的空行 */
    const s2 = s1.replaceAll()
    return s2
}

async function normalizeMdHeader(dir: string) {
    const queue = []
    for (const filePath of DirIterator(dir)) {
        const tmp = readFile(filePath, 'utf-8').then(normalizeMdFileHeader).catch((e) => {
            console.log("fail to normalize file")
            console.log(e)
        })
        queue.push(tmp)
    }
    return Promise.all(queue)
}

function main() {
    normalizeMdHeader(resolve(import.meta.dirname, '..', 'src'))
}

main()