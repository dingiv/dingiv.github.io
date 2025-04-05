---
title: 加密和解密
order: 6
---

# 加密和解密

加密和解密是信息安全的核心技术，用于保护数据的机密性、完整性和可用性。本文将介绍常见的加密算法及其实现。

## 朴素加密

朴素加密算法通常基于简单的替换或移位操作，虽然安全性较低，但易于理解和实现。

### 凯撒密码

凯撒密码是一种替换加密技术，通过将字母表中的每个字母移动固定的位数来实现加密。

```js
class CaesarCipher {
  constructor(shift) {
    this.shift = shift;
  }

  encrypt(text) {
    return text.split('').map(char => {
      if (char.match(/[a-z]/i)) {
        const code = char.charCodeAt(0);
        const base = code >= 97 ? 97 : 65; // 小写字母从97开始，大写字母从65开始
        return String.fromCharCode(((code - base + this.shift) % 26) + base);
      }
      return char;
    }).join('');
  }

  decrypt(text) {
    return text.split('').map(char => {
      if (char.match(/[a-z]/i)) {
        const code = char.charCodeAt(0);
        const base = code >= 97 ? 97 : 65;
        return String.fromCharCode(((code - base - this.shift + 26) % 26) + base);
      }
      return char;
    }).join('');
  }
}

// 使用示例
const cipher = new CaesarCipher(3);
const encrypted = cipher.encrypt('HELLO'); // 'KHOOR'
const decrypted = cipher.decrypt('KHOOR'); // 'HELLO'
```

### Base64编码

Base64是一种将二进制数据编码为ASCII字符的方法，常用于在文本协议中传输二进制数据。

```js
// 使用浏览器内置的Base64编码/解码
const text = 'Hello World';
const encoded = btoa(text); // 编码
const decoded = atob(encoded); // 解码

// 自定义实现
class Base64 {
  static encode(str) {
    return btoa(unescape(encodeURIComponent(str)));
  }

  static decode(str) {
    return decodeURIComponent(escape(atob(str)));
  }
}
```

## 对称加密

对称加密使用相同的密钥进行加密和解密，速度快，适合大量数据的加密。

### AES加密

AES（高级加密标准）是目前最常用的对称加密算法。

```js
// 使用Web Crypto API实现AES加密
async function encryptAES(text, key) {
  const encoder = new TextEncoder();
  const data = encoder.encode(text);
  
  // 生成随机IV
  const iv = crypto.getRandomValues(new Uint8Array(16));
  
  // 导入密钥
  const cryptoKey = await crypto.subtle.importKey(
    'raw',
    encoder.encode(key),
    { name: 'AES-CBC', length: 256 },
    false,
    ['encrypt']
  );
  
  // 加密
  const encrypted = await crypto.subtle.encrypt(
    { name: 'AES-CBC', iv },
    cryptoKey,
    data
  );
  
  // 返回IV和加密数据的组合
  return {
    iv: Array.from(iv).map(b => b.toString(16).padStart(2, '0')).join(''),
    data: Array.from(new Uint8Array(encrypted)).map(b => b.toString(16).padStart(2, '0')).join('')
  };
}

async function decryptAES(encrypted, key) {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  
  // 解析IV和加密数据
  const iv = new Uint8Array(encrypted.iv.match(/.{2}/g).map(byte => parseInt(byte, 16)));
  const data = new Uint8Array(encrypted.data.match(/.{2}/g).map(byte => parseInt(byte, 16)));
  
  // 导入密钥
  const cryptoKey = await crypto.subtle.importKey(
    'raw',
    encoder.encode(key),
    { name: 'AES-CBC', length: 256 },
    false,
    ['decrypt']
  );
  
  // 解密
  const decrypted = await crypto.subtle.decrypt(
    { name: 'AES-CBC', iv },
    cryptoKey,
    data
  );
  
  return decoder.decode(decrypted);
}
```

## 非对称加密

非对称加密使用不同的密钥进行加密和解密，安全性更高，但速度较慢。

### RSA加密

RSA是最常用的非对称加密算法，基于大数分解的困难性。

```js
// 使用Web Crypto API实现RSA加密
async function generateRSAKeys() {
  const keyPair = await crypto.subtle.generateKey(
    {
      name: 'RSA-OAEP',
      modulusLength: 2048,
      publicExponent: new Uint8Array([1, 0, 1]),
      hash: 'SHA-256'
    },
    true,
    ['encrypt', 'decrypt']
  );
  
  return {
    publicKey: await crypto.subtle.exportKey('jwk', keyPair.publicKey),
    privateKey: await crypto.subtle.exportKey('jwk', keyPair.privateKey)
  };
}

async function encryptRSA(text, publicKey) {
  const encoder = new TextEncoder();
  const data = encoder.encode(text);
  
  const importedKey = await crypto.subtle.importKey(
    'jwk',
    publicKey,
    { name: 'RSA-OAEP', hash: 'SHA-256' },
    false,
    ['encrypt']
  );
  
  const encrypted = await crypto.subtle.encrypt(
    { name: 'RSA-OAEP' },
    importedKey,
    data
  );
  
  return Array.from(new Uint8Array(encrypted)).map(b => b.toString(16).padStart(2, '0')).join('');
}

async function decryptRSA(encrypted, privateKey) {
  const decoder = new TextDecoder();
  const data = new Uint8Array(encrypted.match(/.{2}/g).map(byte => parseInt(byte, 16)));
  
  const importedKey = await crypto.subtle.importKey(
    'jwk',
    privateKey,
    { name: 'RSA-OAEP', hash: 'SHA-256' },
    false,
    ['decrypt']
  );
  
  const decrypted = await crypto.subtle.decrypt(
    { name: 'RSA-OAEP' },
    importedKey,
    data
  );
  
  return decoder.decode(decrypted);
}
```

## 哈希加密

哈希加密是单向加密，不可逆，常用于密码存储和数据完整性验证。

### SHA-256

SHA-256是SHA-2系列中的一种哈希算法，输出256位的哈希值。

```js
async function hashSHA256(text) {
  const encoder = new TextEncoder();
  const data = encoder.encode(text);
  
  const hash = await crypto.subtle.digest('SHA-256', data);
  
  return Array.from(new Uint8Array(hash))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

// 使用示例
const hash = await hashSHA256('Hello World');
console.log(hash); // 输出64个字符的十六进制字符串
```

### HMAC

HMAC（基于哈希的消息认证码）用于验证消息的完整性和真实性。

```js
async function generateHMAC(message, key) {
  const encoder = new TextEncoder();
  const messageData = encoder.encode(message);
  const keyData = encoder.encode(key);
  
  const cryptoKey = await crypto.subtle.importKey(
    'raw',
    keyData,
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );
  
  const signature = await crypto.subtle.sign(
    'HMAC',
    cryptoKey,
    messageData
  );
  
  return Array.from(new Uint8Array(signature))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

async function verifyHMAC(message, key, signature) {
  const encoder = new TextEncoder();
  const messageData = encoder.encode(message);
  const keyData = encoder.encode(key);
  
  const cryptoKey = await crypto.subtle.importKey(
    'raw',
    keyData,
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['verify']
  );
  
  const signatureBytes = new Uint8Array(signature.match(/.{2}/g).map(byte => parseInt(byte, 16)));
  
  return await crypto.subtle.verify(
    'HMAC',
    cryptoKey,
    signatureBytes,
    messageData
  );
}
```

## 加密算法的选择

选择加密算法时，需要考虑以下因素：

1. **安全性需求**
   - 数据敏感度
   - 攻击风险
   - 合规要求

2. **性能要求**
   - 加密/解密速度
   - 资源消耗
   - 并发处理能力

3. **应用场景**
   - 数据传输
   - 数据存储
   - 身份认证
   - 数字签名

4. **密钥管理**
   - 密钥生成
   - 密钥存储
   - 密钥分发
   - 密钥轮换

## 最佳实践

1. **密码存储**
   - 使用强哈希算法（如SHA-256）
   - 添加盐值（salt）
   - 使用密钥派生函数（如PBKDF2）

2. **数据传输**
   - 使用HTTPS
   - 实现端到端加密
   - 定期更新密钥

3. **密钥管理**
   - 使用安全的密钥存储
   - 实现密钥轮换机制
   - 保护密钥不被泄露

4. **错误处理**
   - 优雅处理加密/解密失败
   - 记录安全事件
   - 实现故障转移机制

## 总结

加密和解密是保护数据安全的重要手段。选择合适的加密算法和实现方式，遵循安全最佳实践，可以有效保护数据的机密性、完整性和可用性。在实际应用中，需要根据具体场景和需求，选择合适的加密方案，并注意密钥管理和安全实现。
