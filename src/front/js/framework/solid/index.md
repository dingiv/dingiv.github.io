# Solid

## list.map()，Index和For的区别
list.map()全量更新：适合做静态列表渲染；Index在依赖的键值发生变化的时候更新整个列表中的每个项目的相应的键值，不涉及列表dom的重新增删；For在某个依赖的列表项的上数据发生变化的时候，将会销毁这个列表项及其dom，重新创一个新的，而不管其他项；一般地，Index用的多，渲染长列表用For。