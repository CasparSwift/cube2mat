import importlib
import numpy as np
import re
import sys
import os

def get_cube2mat_functions_from_cpp(cpp_path):
    """
    识别machine_alpha.cpp中所有cube2mat开头的bind函数
    """
    if not os.path.exists(cpp_path):
        print(f"未找到 {cpp_path}，请确认路径正确。")
        return []
    with open(cpp_path, "r", encoding="utf-8") as f:
        content = f.read()
    # 匹配 void bind_cube2mat_xxx(py::module& m);
    pattern = re.compile(r"void\s+(bind_cube2mat_[a-zA-Z0-9_]+)\s*\(")
    matches = pattern.findall(content)
    # 这些bind函数名，转换为cube2mat_xxx
    func_names = [name.replace("bind_", "") for name in matches]
    return func_names

def test_machine_alpha_cpp():
    # 假设machine_alpha已经通过pybind11暴露为Python模块
    try:
        machine_alpha = importlib.import_module("machine_alpha")
    except ImportError:
        print("machine_alpha 模块未找到，请确认已正确编译并安装。")
        return

    # 识别所有cube2mat开头的函数
    cpp_path = os.path.join(os.path.dirname(__file__), "machine_alpha.cpp")
    func_names = get_cube2mat_functions_from_cpp(cpp_path)
    if not func_names:
        print("未能识别到任何cube2mat开头的函数。")
        return

    # 构造模拟数据
    d0, d1, d2 = 3, 4, 5
    close = np.random.rand(d0, d1, d2).astype(np.float32)
    volume = np.abs(np.random.rand(d0, d1, d2).astype(np.float32)) + 1e-3  # 避免出现0或负数
    vwap = np.random.rand(d0, d1, d2).astype(np.float32)
    high = close + np.abs(np.random.rand(d0, d1, d2).astype(np.float32))  # 保证high>=close
    low = close - np.abs(np.random.rand(d0, d1, d2).astype(np.float32))   # 保证low<=close

    cubes_map = {
        "last_price": close,
        "interval_volume": volume,
        "interval_vwap": vwap,
        "interval_high": high,
        "interval_low": low,
    }

    for func in func_names:
        result = np.zeros((d0, d1), dtype=np.float32)
        func_obj = getattr(machine_alpha, func, None)
        if func_obj is None:
            print(f"模块中未找到函数 {func}")
            continue
        try:
            func_obj(result, cubes_map)
            # print(f"{func} 运行成功，输出如下：")
            # print(result)
        except Exception as e:
            print(f"{func} 运行失败：{e}")

if __name__ == "__main__":
    test_machine_alpha_cpp()
