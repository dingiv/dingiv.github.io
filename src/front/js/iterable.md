# 可迭代


可迭代对象，是指拥有键值[Symbol.iterator](){}的对象，该方法返回这个对象的迭代器，一个迭代器应当具有一个方法，next(){}，该方法在每次迭代时调用，返回一个迭代结果对象，{done和value}，done代表迭代是否还有下次，value代表本次迭代的值是多少。


生成器函数 function* generator()，调用后返回一个生成器，生成器可以间断执行生成器函数中的代码。在生成器函数中，可以使用yield关键字进行分割代码，将一个大型的任务切割成一个个小的任务，特别是非常多次的循环任务，在循环中使用yield，可以将循环拆分成多次子任务。生成器是一个迭代器，同时又是一个可迭代对象，因为生成器的[Symbol.iterator]键值指向了自身。生成器函数语法是可读流和可写流的语法糖，甚至是上位替代。
