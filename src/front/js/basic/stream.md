# Stream API

## ReadableStream
### 获得生产者
1. 构造函数
   ```js
   new ReadableStream({
       start?(ctrl) {
         ctrl.enqueue(data)
       },
       pull?(ctrl){ }
   })
   ```
2. 从其他API获得

### 使用生产者
+ 对接至 WriteableStream，rs.pipeTo(ws)
+ 调用 reader，手动消费，
  reader=rs.getReader()
  reader.read().then((res)=>process(res))


## WriteableStream：消费者
### 获得消费者
+ 构造函数
```js
new WriteableStream({
    write(data){ }
})
```

### 使用消费者
+ 让Readable对接
+ 手动调用writer，
writer=ws.getWriter()
writer.ready.write()

## node:stream.Readable
### 获得生产者
+ 构造函数
```js
new Readable({
  read(){
    this.push(chunk)
  }
})
```
+ 从其他API获得，文件流，标准流，网络流，压缩流
+ 从可迭代对象获取，Readable.from(iterable)


### 使用生产者
+ 对接至 Writeable，rs.pipe(ws)
+ 监听 data 事件
+ 监听 readable事件，并且手动调用 rs.read() 方法消费值，这种情况下，流式静止的，必须循环调用 read()
+ 使用 node:stream/promise 的 pipeline()，流编排函数

生产者有静止状态、流动状态和关闭状态，初始为静止状态，在调用了pipe和被监听了data事件后变为流动状态

## node:stream.Writeable
### 获得消费者
+ 构造函数
```js
new Writeable({
  write(chunk, encoding, cb){
    process(chunk)
    cb()
  }
})
```
+ 从其他API获得，文件流，标准流，网络流，压缩流

### 使用消费者
+ 让Readable对接
+ 监听drain事件
+ 监听writeable事件，手动调用ws.write()生产值
+ pipeline

消费者有静止状态、流动状态和关闭状态，初始为静止状态，被对接pipe和被监听了drain事件后变为流动状态
![alt text](stream.png)

## 生成器
生成器函数 `function* generator()`，调用后返回一个生成器，生成器可以间断执行生成器函数中的代码。在生成器函数中，可以使用 `yield` 关键字进行分割代码，将一个大型的任务切割成一个个小的任务，特别是非常多次的循环任务，在循环中使用 yield，可以将循环拆分成多次子任务。生成器是一个迭代器，同时又是一个可迭代对象，因为生成器的[Symbol.iterator]键值指向了自身。生成器函数语法是可读流和可写流的语法糖，甚至是上位替代。
