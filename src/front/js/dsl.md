# dsl
DSL（Domain-Specific Language，领域专用语言）是一种专门为特定应用领域设计的计算机语言。与通用编程语言（如 Python、Java、C++等）不同，DSL 主要用于解决某一特定领域中的问题，因此在该领域中更高效、更易于使用。DSL 可以是编程语言、标记语言或配置语言，常见的 DSL 包括：

- 前端 UI 描述语言。HTML可以集成其他语言，包括CSS、JavaScript，还有SVG、MathML，通过JS间接集成的有MD、JSON、Web Component、XML、WebGL、WebAssembly。
  - HTML（HyperText Markup Language）：用于网页结构的标记语言。
  - CSS（Cascading Style Sheets）：用于网页样式的描述。
  - MD（Markdown）：轻量级标记型文本文件，HTML的同构异形体。
  - SVG（Scalable Vector Graphic）：使用类似于 XML 的语法。可以看做是 HTML 的语法扩展。
  - MathML（Mathematical Markup Language）：使用XML语法表示数学公式。可以看做是HTML的语法扩展。
  - Sass（Syntactically Awesome Style Sheets）：CSS 的语法扩展，为 CSS 提供更多高级语言的能力。
  - JSX（JavaScript Extension/XML）：由 React 框架所引入的一种 JavaScript 语法糖，属于 JavaScript 的语法扩展。
  - Vue SFC：Vue单文件组件语法。HTML、CSS、JS的组合体。
  - WASM（WebAssembly）：由编译语言编译而成。

- 后端数据处理语言。
  - SQL（Structured Query Language）：用于数据库查询和操作。
  - nginx conf：用于配置

- 通用格式化语言或者数据语言。
  - RE（Regular Expresstion）正则表达式
  - XML（Extensible Markup Language）：是一种标记语言，提供定义任何数据的规则。与其他编程语言不同，XML 本身无法执行计算操作。相反，任何编程语言或软件都可以实现结构化数据管理。
  - YAML（Yet Another Markup Language）：使用空格和缩进来代表层级的标记型语言。目前常用的场景在k8s的配置文件和Java SpringBoot项目的配置文件中。
