import base64

# 编码
with open(r"E:\14924\Pictures\Camera Roll\兼职背景图.jpg", "rb") as f:
    binary_data = f.read()                         # bytes: 原始二进制
    base64_bytes = base64.b64encode(binary_data)   # bytes: Base64 编码后仍是 bytes 类型
    base64_string = base64_bytes.decode("utf-8")   # str:   转成字符串（Python 类型要求）

# .decode("utf-8") 的作用：
# Python 的 b64encode 返回 bytes 类型，但 JSON/f-string 需要 str 类型
# 因为 Base64 输出全是 ASCII 字符，所以这里只是做类型转换，不涉及字符编码解读

# 解码
binary_data = base64.b64decode(base64_string)
print(base64_string)