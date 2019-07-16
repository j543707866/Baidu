import paddle.fluid as fluid
# 定义数组维度及数据类型，可以修改shape参数定义任意大小的数组
data = fluid.layers.ones(shape=[5], dtype='int64')
# 在CPU上执行运算
place = fluid.CPUPlace()
# 创建执行器
exe = fluid.Executor(place)
# 执行计算
ones_result = exe.run(fluid.default_main_program(),
                        # 获取数据data
                        fetch_list=[data],
                        return_numpy=True)
# 输出结果
print(ones_result[0])