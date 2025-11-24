def bubble_sort(arr):
    """
    冒泡排序（升序）
    :param arr: 待排序的列表（支持数字、字符串等可比较类型）
    :return: 排序后的列表（原列表会被修改，返回是为了方便链式调用）
    """
    # 获取列表长度
    n = len(arr)

    # 外层循环：控制排序轮数（最多需要 n-1 轮，因为每轮确定一个最大元素的位置）
    for i in range(n - 1):
        # 内层循环：每轮比较相邻元素，将未排序部分的最大元素冒泡到末尾
        # 优化点：每轮后末尾 i 个元素已排序，无需再比较，所以范围是 n-1 - i
        for j in range(n - 1 - i):
            # 若当前元素大于下一个元素，交换两者（升序逻辑）
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return arr


# 测试示例
if __name__ == "__main__":
    # 测试数字列表
    num_list = [64, 34, 25, 12, 22, 11, 90]
    print("原始数字列表：", num_list)
    bubble_sort(num_list)
    print("升序排序后：", num_list)

    # 测试字符串列表（按 ASCII 码排序）
    str_list = ["banana", "apple", "cherry", "date"]
    print("\n原始字符串列表：", str_list)
    bubble_sort(str_list)
    print("升序排序后：", str_list)