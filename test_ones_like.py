#coding=utf8
import tensorflow as tf
import basic.util.prints as p

sess = tf.InteractiveSession()

# create a tensor object
original = [1,2,3,4,5]
# original = tf.zeros([5], dtype=tf.float16)
p.printRowValue("Original value", original)

# 调用ones_like，默认类型为int
data = tf.ones_like(original)
p.printValue("tf.ones_like(original)", data)

# 调用ones_like，默认类型为double
data = tf.ones_like(original, dtype=tf.double)
p.printValue("tf.ones_like(original, dtype=tf.double)", data)

# dtype类型为: float, int, double, uint以及complex(复数), 如下类型支持

data = tf.ones_like(original, dtype=tf.float32)
p.printValue("tf.ones_like(original, dtype=tf.float32)", data)
data = tf.ones_like(original, dtype=tf.float32_ref)
p.printValue("tf.ones_like(original, dtype=tf.float32_ref)", data)
data = tf.ones_like(original, dtype=tf.float64)
p.printValue("tf.ones_like(original, dtype=tf.float64)", data)
data = tf.ones_like(original, dtype=tf.float64_ref)
p.printValue("tf.ones_like(original, dtype=tf.float64_ref)", data)
data = tf.ones_like(original, dtype=tf.int8)
p.printValue("tf.ones_like(original, dtype=tf.int8)", data)
data = tf.ones_like(original, dtype=tf.int8_ref)
p.printValue("tf.ones_like(original, dtype=tf.int8_ref)", data)
data = tf.ones_like(original, dtype=tf.int16)
p.printValue("tf.ones_like(original, dtype=tf.int16)", data)
data = tf.ones_like(original, dtype=tf.int16_ref)
p.printValue("tf.ones_like(original, dtype=tf.int16_ref)", data)
data = tf.ones_like(original, dtype=tf.int32)
p.printValue("tf.ones_like(original, dtype=tf.int32)", data)
data = tf.ones_like(original, dtype=tf.int32_ref)
p.printValue("tf.ones_like(original, dtype=tf.int32_ref)", data)
data = tf.ones_like(original, dtype=tf.int64)
p.printValue("tf.ones_like(original, dtype=tf.int64)", data)
data = tf.ones_like(original, dtype=tf.int64_ref)
p.printValue("tf.ones_like(original, dtype=tf.int64_ref)", data)
data = tf.ones_like(original, dtype=tf.uint8)
p.printValue("tf.ones_like(original, dtype=tf.uint8)", data)
data = tf.ones_like(original, dtype=tf.uint8_ref)
p.printValue("tf.ones_like(original, dtype=tf.uint8_ref)", data)

data = tf.ones_like(original, dtype=tf.double)
p.printValue("tf.ones_like(original, dtype=tf.double)", data)
data = tf.ones_like(original, dtype=tf.double_ref)
p.printValue("tf.ones_like(original, dtype=tf.double_ref)", data)
data = tf.ones_like(original, dtype=tf.complex64)
p.printValue("tf.ones_like(original, dtype=tf.complex64)", data)
data = tf.ones_like(original, dtype=tf.complex64_ref)
p.printValue("tf.ones_like(original, dtype=tf.complex64_ref)", data)
data = tf.ones_like(original, dtype=tf.complex128)
p.printValue("tf.ones_like(original, dtype=tf.complex128)", data)
data = tf.ones_like(original, dtype=tf.complex128_ref)
p.printValue("tf.ones_like(original, dtype=tf.complex128_ref)", data)

# 特殊情况
data = tf.ones_like(original, dtype=tf.float16)
# p.printValue("tf.ones_like(original, dtype=tf.float16)", data)
data = tf.ones_like(original, dtype=tf.float16_ref)
# p.printValue("tf.ones_like(original, dtype=tf.float16_ref)", data)
data = tf.ones_like(original, dtype=tf.uint16)
# p.printValue("tf.ones_like(original, dtype=tf.uint16)", data)
data = tf.ones_like(original, dtype=tf.uint16_ref)
# p.printValue("tf.ones_like(original, dtype=tf.uint16_ref)", data)

# [ERROR] 不支持类型包括: bfloat, qint, quint
# data = tf.ones_like(original, dtype=tf.bfloat16)
# data = tf.ones_like(original, dtype=tf.quint8)
# data = tf.ones_like(original, dtype=tf.qint16)
# data = tf.ones_like(original, dtype=tf.qint32)


sess.close()