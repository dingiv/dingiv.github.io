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
+ 在选择加密算法时，首先要评估系统对安全性的需求。对于高度敏感的数据，必须采用经过严格验证的加密算法，并结合数据的敏感度、潜在的攻击风险以及相关的合规要求来综合考量。只有在充分理解数据所面临的威胁和法律法规的前提下，才能做出合理的安全性决策。
+ 性能。不同算法在加密和解密速度、资源消耗以及对并发处理能力的支持上存在差异。对于需要处理大量数据或高并发场景的应用，应优先考虑那些在保证安全性的同时，能够高效运行且资源占用较低的加密算法，以确保系统的整体性能不受影响。
+ 应用场景的不同也会影响加密算法的选择。例如，数据传输、数据存储、身份认证和数字签名等场景对加密算法的要求各不相同。应根据具体的业务需求，选择最适合该场景的加密方式，从而在安全性和实用性之间取得平衡。
+ 密钥管理能力直接关系到加密系统的安全性。无论选择何种加密算法，都需要有完善的密钥生成、存储、分发和轮换机制。只有确保密钥在整个生命周期内都受到严格保护，才能真正发挥加密算法的安全作用，防止因密钥泄露而导致的安全事件。

## 最佳实践
在密码存储方面，安全性至关重要。应当采用强大的哈希算法（如SHA-256）对密码进行加密存储，并为每个密码添加独特的盐值（salt），以防止彩虹表攻击。此外，推荐使用密钥派生函数（如PBKDF2），通过多次迭代进一步增强密码的安全性。这样可以有效提升密码存储的抗攻击能力，降低数据泄露带来的风险。

数据在传输过程中同样需要严密保护。建议始终通过HTTPS等安全协议进行数据传输，确保信息在网络中不会被窃听或篡改。对于敏感数据，可以实现端到端加密，确保只有通信双方能够解密内容。同时，定期更新加密密钥，有助于防止长期密钥泄露带来的安全隐患。

密钥管理是加密体系的核心环节。应将密钥存储在安全的环境中，避免明文存放或暴露在不安全的地方。建立完善的密钥轮换机制，定期更换密钥，能够降低密钥泄露后的潜在损失。此外，密钥的生成、分发和销毁都应遵循严格的安全流程，确保密钥在整个生命周期内都受到保护。

在加密和解密过程中，错误处理同样不可忽视。系统应优雅地处理加密或解密失败的情况，避免泄露敏感信息。对于所有安全相关的异常和事件，建议进行详细记录，以便后续审计和追踪。同时，设计合理的故障转移机制，确保在部分系统出现故障时，整体服务依然能够保持可用性和安全性。