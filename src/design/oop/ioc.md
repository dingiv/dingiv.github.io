# IOC (JS篇)
在js中使用 IOC，一般使用的是 inversify 这个库，inversify 是 js 中比较流行的 ioc 库，它借鉴了 java 中的 spring，但是更轻量级，并且使用装饰器语法，使得代码更简洁。

inversify 的 ioc 容器是 container，它是一个单例对象，可以通过 container.get(id) 来获取实例对象，id 是 bind 的时候指定的 id。

## bind语法
声明哪些id值可以从容器中获得实例对象，id可以是string或者是Symbol类型。（相较于spring，spring的id是唯一的，而inversify的id可以重复，并且给绑定添加了额外的限定，一个是name，一个是tag，并且可以增加条件绑定，只有满足条件的时候才会绑定成功）
+ container.bind(id).to(构造函数/类名)
+ container.bind(id).toDynamicValue(工厂函数)
+ container.bind(id).toFactory(工厂函数)  // 与二相似，不过该工厂函数可以接受一个参数为 context
+ container.bind(id).toConstantValue(实例)   // 绑定实例
+ container.bind(id).toProvider(工厂函数)  // 与二相似，不过该工厂函数可以接受一个参数为 context
+ container.bind(id).toSelf()  // 绑定自身，id 是类名
+ container.bind(id).toConstructor(构造函数)  // 绑定构造函数
+ container.bind(id).toService(类名)  // 绑定服务

bind条件，inversify 的 bind 可以添加条件，言下之意是 bind 可以绑定多个值，但是只有满足条件的时候才会绑定成功，条件有：
+ when
+ whenTargetNamed
+ whenTargetTagged

bind装饰器

## get语法
注入装饰器

生命周期 hooks

## Angular中的IOC
在 angular 中，ioc 容器是 Injector，它是一个单例对象，可以通过 Injector.get(id) 来获取实例对象，id 是 provide 的时候指定的 id。

provide语法
+ provide(id, {useClass: 类名})
+ provide(id, {useFactory: 工厂方法})
+ provide(id, {useValue: 实例})
+ provide(id, {useExisting: id})
+ provide(id, {useClass: 类名, deps: [id1, id2]})

inject语法
+ @Inject(id)
### Angular中的依赖注入
```js
// 实例
@Injectable({
  providedIn: 'root'
})
export class UserService {
  constructor(private http: HttpClient) {}
  getUser() {
    return this.http.get('/user');
  }
}

// 使用
@Injectable({
  providedIn: 'root'
})
export class AppComponent {
  constructor(private userService: UserService) {}
  ngOnInit() {
    this.userService.getUser().subscribe(user => {
      console.log(user);
    });
  }
}
```

## Nestjs
