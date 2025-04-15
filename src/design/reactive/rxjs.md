# RxJS 响应式编程

RxJS 是一个用于处理异步数据流的 JavaScript 库，它基于响应式编程范式，提供了一套丰富的操作符和工具来处理各种异步场景。RxJS 的核心概念是 Observable（可观察对象），它代表了一个随时间变化的数据流。

## Observable 基础

Observable 是一个表示数据流的对象，它可以发出零个或多个值，然后可能完成或出错。创建 Observable 的基本方式：

```javascript
import { Observable } from 'rxjs';

// 创建自定义 Observable
const observable = new Observable(subscriber => {
    subscriber.next(1);
    subscriber.next(2);
    subscriber.next(3);
    subscriber.complete();
});

// 订阅 Observable
observable.subscribe({
    next: value => console.log(value),
    error: err => console.error(err),
    complete: () => console.log('Completed')
});
```

## 创建操作符

RxJS 提供了多种创建 Observable 的操作符：

```javascript
import { of, from, interval, fromEvent } from 'rxjs';

// 从固定值创建
const source1 = of(1, 2, 3);

// 从数组或 Promise 创建
const source2 = from([1, 2, 3]);
const source3 = from(fetch('https://api.example.com'));

// 创建定时器
const source4 = interval(1000);

// 从 DOM 事件创建
const source5 = fromEvent(document, 'click');
```

## 转换操作符

转换操作符用于修改 Observable 发出的值：

```javascript
import { map, pluck, scan } from 'rxjs/operators';

// map 操作符
source1.pipe(
    map(x => x * 2)
).subscribe(console.log);

// pluck 操作符
fromEvent(document, 'keyup').pipe(
    pluck('target', 'value')
).subscribe(console.log);

// scan 操作符（类似 reduce）
source1.pipe(
    scan((acc, curr) => acc + curr, 0)
).subscribe(console.log);
```

## 过滤操作符

过滤操作符用于选择性地处理 Observable 发出的值：

```javascript
import { filter, take, skip, distinctUntilChanged } from 'rxjs/operators';

// filter 操作符
source1.pipe(
    filter(x => x % 2 === 0)
).subscribe(console.log);

// take 操作符
source1.pipe(
    take(2)
).subscribe(console.log);

// distinctUntilChanged 操作符
source1.pipe(
    distinctUntilChanged()
).subscribe(console.log);
```

## 组合操作符

组合操作符用于合并多个 Observable：

```javascript
import { merge, concat, combineLatest, zip } from 'rxjs';
import { mergeMap, switchMap, concatMap } from 'rxjs/operators';

// merge 操作符
const sourceA = interval(1000);
const sourceB = interval(2000);
merge(sourceA, sourceB).subscribe(console.log);

// combineLatest 操作符
combineLatest([sourceA, sourceB]).subscribe(console.log);

// switchMap 操作符
fromEvent(input, 'input').pipe(
    switchMap(event => fetch(`/api/search?q=${event.target.value}`))
).subscribe(console.log);
```

## 错误处理

RxJS 提供了多种处理错误的方式：

```javascript
import { catchError, retry, retryWhen } from 'rxjs/operators';

// catchError 操作符
source1.pipe(
    map(x => {
        if (x === 2) throw new Error('Error!');
        return x;
    }),
    catchError(err => of('Handled error'))
).subscribe(console.log);

// retry 操作符
source1.pipe(
    retry(3)
).subscribe(console.log);
```

## 实用工具

RxJS 还提供了一些实用工具：

```javascript
import { tap, delay, timeout } from 'rxjs/operators';

// tap 操作符（用于调试）
source1.pipe(
    tap(x => console.log('Before:', x)),
    map(x => x * 2),
    tap(x => console.log('After:', x))
).subscribe();

// delay 操作符
source1.pipe(
    delay(1000)
).subscribe(console.log);

// timeout 操作符
source1.pipe(
    timeout(500)
).subscribe(console.log);
```

## 实际应用示例

### 搜索建议

```javascript
import { fromEvent } from 'rxjs';
import { debounceTime, distinctUntilChanged, switchMap } from 'rxjs/operators';

const searchBox = document.getElementById('search');
const results = document.getElementById('results');

fromEvent(searchBox, 'input').pipe(
    map(event => event.target.value),
    debounceTime(300),
    distinctUntilChanged(),
    switchMap(term => fetch(`/api/search?q=${term}`).then(res => res.json()))
).subscribe(data => {
    results.innerHTML = data.map(item => `<div>${item}</div>`).join('');
});
```

### 拖拽实现

```javascript
import { fromEvent } from 'rxjs';
import { map, switchMap, takeUntil } from 'rxjs/operators';

const draggable = document.getElementById('draggable');

const mousedown$ = fromEvent(draggable, 'mousedown');
const mousemove$ = fromEvent(document, 'mousemove');
const mouseup$ = fromEvent(document, 'mouseup');

mousedown$.pipe(
    switchMap(() => mousemove$.pipe(
        map(event => ({
            x: event.clientX,
            y: event.clientY
        })),
        takeUntil(mouseup$)
    ))
).subscribe(pos => {
    draggable.style.transform = `translate(${pos.x}px, ${pos.y}px)`;
});
```

## 最佳实践

1. **及时取消订阅**：使用 takeUntil 或 unsubscribe 避免内存泄漏
2. **合理使用操作符**：选择最适合的操作符组合
3. **错误处理**：始终处理可能的错误情况
4. **性能优化**：使用 debounceTime、throttleTime 等操作符优化性能
5. **测试**：使用 marble testing 测试 Observable

## 总结

RxJS 是一个强大的响应式编程库，它通过 Observable 和丰富的操作符提供了处理异步数据流的优雅方式。掌握 RxJS 可以帮助开发者更好地处理复杂的异步场景，提高代码的可维护性和可读性。 