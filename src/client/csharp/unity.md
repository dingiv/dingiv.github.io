# Unity

Unity 是流行的游戏引擎，使用 C# 作为脚本语言。Unity 的 C# 开发与传统 .NET 开发有一些区别，Unity 封装了游戏开发的常见模式，如游戏循环、组件系统、事件系统等。理解 Unity 的 C# 脚本系统是游戏开发的基础。

## MonoBehaviour

MonoBehaviour 是 Unity 脚本的基类，所有游戏脚本都继承自它。MonoBehaviour 提供了生命周期方法，这些方法由 Unity 引擎在特定时机自动调用。

```csharp
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    // Awake 在脚本实例化时调用，只调用一次
    void Awake()
    {
        Debug.Log("Player created");
    }

    // Start 在第一帧 Update 之前调用，只调用一次
    void Start()
    {
        Debug.Log("Player started");
    }

    // Update 每帧调用一次
    void Update()
    {
        // 处理输入、移动、动画等
        float horizontal = Input.GetAxis("Horizontal");
        transform.Translate(Vector3.right * horizontal * Time.deltaTime);
    }

    // FixedUpdate 以固定时间间隔调用，适合物理计算
    void FixedUpdate()
    {
        // 物理移动
        GetComponent<Rigidbody>().AddForce(Vector3.up);
    }

    // LateUpdate 在所有 Update 完成后调用
    void LateUpdate()
    {
        // 相机跟随等需要等物体移动完成的操作
    }

    // OnEnable 当对象启用时调用
    void OnEnable()
    {
        Debug.Log("Player enabled");
    }

    // OnDisable 当对象禁用时调用
    void OnDisable()
    {
        Debug.Log("Player disabled");
    }

    // OnDestroy 当对象销毁时调用
    void OnDestroy()
    {
        Debug.Log("Player destroyed");
    }
}
```

生命周期方法的调用顺序是：Awake → Start → Update/LateUpdate → OnDisable → OnDestroy。Awake 适合初始化内部状态，Start 适合初始化依赖于其他组件的引用。

## 协程

协程 (Coroutine) 是 Unity 的异步执行机制，允许代码在多个帧中分散执行。协程通过 IEnumerator 返回类型和 yield return 语句实现。

```csharp
using UnityEngine;
using System.Collections;

public class CoroutineExample : MonoBehaviour
{
    void Start()
    {
        // 启动协程
        StartCoroutine(DoSomething());
    }

    IEnumerator DoSomething()
    {
        Debug.Log("Start");
        yield return new WaitForSeconds(2f);  // 等待 2 秒
        Debug.Log("After 2 seconds");

        yield return new WaitUntil(() => Time.time > 5f);  // 等待条件
        Debug.Log("Time > 5");

        yield return null;  // 等待下一帧
        Debug.Log("Next frame");

        yield break;  // 停止协程
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            // 停止所有协程
            StopAllCoroutines();
        }
    }
}
```

常用的 yield 语句包括 WaitForSeconds（等待秒数）、WaitForEndOfFrame（等待帧结束）、WaitUntil（等待条件）、WaitWhile（等待条件不成立）。

## 组件系统

Unity 采用组件化架构，游戏对象由多个组件组成。Transform 是每个游戏对象必备的组件，管理位置、旋转、缩放。其他组件包括 Renderer（渲染）、Collider（碰撞）、Rigidbody（物理）、Animator（动画）等。

```csharp
using UnityEngine;

public class ComponentExample : MonoBehaviour
{
    private Rigidbody rb;
    private Collider col;

    void Start()
    {
        // 获取组件
        rb = GetComponent<Rigidbody>();
        col = GetComponent<Collider>();

        // 添加组件
        var light = gameObject.AddComponent<Light>();
        light.color = Color.red;

        // 查找子对象的组件
        var childRenderer = transform.Find("Child").GetComponent<Renderer>();

        // 查找所有指定类型的组件
        Collider[] allColliders = FindObjectsOfType<Collider>();
    }

    void Update()
    {
        // 访问 Transform 组件
        transform.position = new Vector3(0, 0, 0);
        transform.rotation = Quaternion.identity;
        transform.localScale = Vector3.one;

        // 相对移动
        transform.Translate(Vector3.forward * Time.deltaTime);
        transform.Rotate(Vector3.up * Time.deltaTime * 90f);
    }
}
```

组件可以通过 Inspector 面板配置，也可以在运行时动态添加和移除。GetComponent 是开销较大的操作，应该在 Start 或 Awake 中缓存组件引用。

## 输入处理

Unity 的输入系统通过 Input 类获取键盘、鼠标、手柄等输入。

```csharp
using UnityEngine;

public class InputExample : MonoBehaviour
{
    void Update()
    {
        // 键盘按键
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Debug.Log("Space pressed");
        }

        // 鼠标按键
        if (Input.GetMouseButton(0))
        {
            Debug.Log("Left mouse button");
        }

        // 轴输入（WASD、方向键、手柄摇杆）
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");

        Vector3 movement = new Vector3(horizontal, 0, vertical);
        transform.Translate(movement * Time.deltaTime);
    }
}
```

Unity 2020+ 推荐使用新的 Input System 包，提供了更灵活和可扩展的输入处理方式。

```bash
dotnet add package Unity.InputSystem
```

## 物理系统

Unity 的物理引擎处理碰撞检测和刚体模拟。Rigidbody 组件赋予物理属性，Collider 组件定义碰撞形状。

```csharp
using UnityEngine;

public class PhysicsExample : MonoBehaviour
{
    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();

        // 添加力（瞬时力）
        rb.AddForce(Vector3.up * 10f, ForceMode.Impulse);

        // 添加扭矩
        rb.AddTorque(Vector3.up * 5f);

        // 设置速度
        rb.velocity = Vector3.forward * 10f;
    }

    void OnCollisionEnter(Collision collision)
    {
        // 碰撞开始
        Debug.Log("Collided with " + collision.gameObject.name);
    }

    void OnCollisionStay(Collision collision)
    {
        // 碰撞持续
    }

    void OnCollisionExit(Collision collision)
    {
        // 碰撞结束
    }

    void OnTriggerEnter(Collider other)
    {
        // 触发器进入（需要 Collider 设置为 Trigger）
        if (other.CompareTag("Pickup"))
        {
            Destroy(other.gameObject);
        }
    }
}
```

ForceMode.Impulse 应用瞬时力（如爆炸），ForceMode.Force 应用持续力（如发动机）。触发器 (Trigger) 只检测碰撞不产生物理反应，适合拾取物品、检测区域等。

## 预制体

Prefab (预制体) 是 Unity 的可复用游戏对象模板。Prefab 可以包含任意组件和子对象，修改 Prefab 会影响所有实例。

```csharp
using UnityEngine;

public class PrefabExample : MonoBehaviour
{
    public GameObject enemyPrefab;

    void Start()
    {
        // 实例化预制体
        GameObject enemy = Instantiate(enemyPrefab, transform.position, Quaternion.identity);

        // 实例化到指定位置
        Vector3 spawnPos = new Vector3(0, 0, 10);
        Instantiate(enemyPrefab, spawnPos, Quaternion.identity);
    }
}
```

## 单例模式

单例模式在 Unity 中很常见，用于管理全局状态和游戏管理器。

```csharp
using UnityEngine;

public class GameManager : MonoBehaviour
{
    private static GameManager instance;

    public static GameManager Instance
    {
        get
        {
            if (instance == null)
            {
                // 查找现有实例
                instance = FindObjectOfType<GameManager>();

                if (instance == null)
                {
                    // 创建新实例
                    GameObject go = new GameObject("GameManager");
                    instance = go.AddComponent<GameManager>();
                }
            }
            return instance;
        }
    }

    void Awake()
    {
        if (instance == null)
        {
            instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }
}
```

DontDestroyOnLoad 使对象在场景切换时不被销毁，适合用于全局管理器。

## 对象池

对象池是优化性能的常用技术，避免频繁创建和销毁对象（如子弹、敌人）。

```csharp
using UnityEngine;
using System.Collections.Generic;

public class BulletPool : MonoBehaviour
{
    public GameObject bulletPrefab;
    private Queue<GameObject> pool = new Queue<GameObject>();

    GameObject Get()
    {
        if (pool.Count > 0)
        {
            GameObject obj = pool.Dequeue();
            obj.SetActive(true);
            return obj;
        }
        return Instantiate(bulletPrefab);
    }

    void Return(GameObject obj)
    {
        obj.SetActive(false);
        pool.Enqueue(obj);
    }
}
```

## 脚本执行顺序

Unity 允许控制脚本的执行顺序，通过 Project Settings → Script Execution Order 配置。

```csharp
[DefaultExecutionOrder(-100)]
public class EarlyUpdate : MonoBehaviour
{
    void Update()
    {
        // 此 Update 会在大多数脚本之前执行
    }
}

[DefaultExecutionOrder(100)]
public class LateUpdate : MonoBehaviour
{
    void Update()
    {
        // 此 Update 会在大多数脚本之后执行
    }
}
```

正数的 Order 值表示较晚执行，负数表示较早执行。
