# Async/Await
Async/Await 是异步编程中中断当前函数的语法糖，使得程序员可以像书写同步代码那样来书写异步代码，让异步代码的编写和阅读更加直观和简单。它本质上是一种更优雅的异步编程方式。

```javascript
async function fetchData() {
    // 函数体
}
```

```javascript
async function fetchData() {
    const response = await fetch('https://api.example.com/data');
    const data = await response.json();
    return data;
}
```

## 实现原理
async/await 语法是状态机的语法糖，一个 async 函数往往被编译成一个状态机函数，状态机函数接受一个状态对象，包含函数中所需全部变量，函数内部根据当前的状态进入相应的处理分支，然后进行状态转移，如此往复，直至到达结束状态，然后退出执行。

```c
int async_fn(int arg) {
   int a = await arg + 1;   // line 1

   int b = await a + 2;   // line 3

   return a + b;
}

// 编译产物
typedef struct State {
   int arg;
   int a;
   int b;

   int _line;
   void *_data;
   int _fd;
} State;

void sync_fn(void **_state) {
   State *state = *_state;
   switch (state->_line) {
      case 0:
         *_state = state = malloc(sizeof(State));
         state->_line = 1;  // 设置下一个分支为 1
         state->_data = state->arg + 1;
         state->_fd = somefd();
         return;
      case 1:
         state->_line = 3;
         state->a = state->_data;
         state->_data = state->a + 2;
         return;
      case 3:
         state->_line = -1;
         state->b = state->_data;
         state->_data = state->a + state->b;
         return;
      default:
         state->_line = -1;
   }
   return;
}

// 异步函数运行时
void *run_async(void*(*fn)(void** state)) {
   void *state = NULL;
   while (state->_line >= 0) {
      fn(&state);
      // 阻塞等待事件到来, 伪代码
      select(state->_fd);
   }
   return state->_data;
}


int ret = async_fn(1);
// ret 4

// ==> 编译调用
int ret = run_async(sync_fn);
// ret 4;
```

## 错误处理

### Try-Catch 方式
```javascript
async function fetchData() {
    try {
        const response = await fetch('https://api.example.com/data');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}
```

### Promise.catch 方式
```javascript
async function fetchData() {
    const response = await fetch('https://api.example.com/data')
        .catch(error => {
            console.error('Fetch error:', error);
            throw error;
        });
    const data = await response.json();
    return data;
}
```

## 并发执行

### 顺序执行
```javascript
async function sequential() {
    const result1 = await task1();
    const result2 = await task2();
    return [result1, result2];
}
```

### 并行执行
```javascript
async function parallel() {
    const [result1, result2] = await Promise.all([
        task1(),
        task2()
    ]);
    return [result1, result2];
}
```

