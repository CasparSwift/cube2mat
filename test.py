import numpy as np
import machine_alpha # 这里是你编译生成的 pybind11 模块名

def test_machine_alpha_functions():
    """测试 machine_alpha 模块中的多个函数"""
    print("=== 测试 machine_alpha 模块函数 ===\n")
    
    d0, d1, d2 = 3, 5, 20
    
    # 构造测试数据
    np.random.seed(42)  # 设置随机种子以便结果可重现
    close = np.random.rand(d0, d1, d2).astype(np.float32) * 100  # 随机价格
    volume = (np.random.rand(d0, d1, d2) * 1000).astype(np.float32)  # 随机成交量
    volume[2, 3, 3:6] = np.nan  # 添加一些 NaN 值
    
    # 构造 high, low, open 数据用于其他函数
    high = close + np.random.rand(d0, d1, d2).astype(np.float32) * 5
    low = close - np.random.rand(d0, d1, d2).astype(np.float32) * 5
    open_price = close + np.random.rand(d0, d1, d2).astype(np.float32) * 2 - 1
    
    # 输出数组 result (d0, d1)
    result = np.empty((d0, d1), dtype=np.float32)
    
    # 测试函数列表
    test_functions = [
        ("cube2mat_vwap", {"close": close, "volume": volume}),
        ("cube2mat_volume_median_to_mean_ratio", {"interval_volume": volume}),
        ("cube2mat_volume_gini", {"interval_volume": volume}),
        ("cube2mat_volume_skew", {"interval_volume": volume}),
        ("cube2mat_ret_skew", {"last_price": close}),
        ("cube2mat_parkinson_vol", {"high": high, "low": low}),
        ("cube2mat_path_efficiency", {"last_price": close}),
        ("cube2mat_total_variation_close", {"last_price": close}),
    ]
    
    for func_name, cubes_map in test_functions:
        try:
            print(f"测试函数: {func_name}")
            print(f"输入数据形状: {[v.shape for v in cubes_map.values()]}")
            
            # 调用函数
            getattr(machine_alpha, func_name)(result, cubes_map)
            
            print(f"结果形状: {result.shape}")
            print(f"结果统计:")
            print(f"  非NaN值数量: {np.sum(~np.isnan(result))}")
            print(f"  最小值: {np.nanmin(result):.6f}")
            print(f"  最大值: {np.nanmax(result):.6f}")
            print(f"  平均值: {np.nanmean(result):.6f}")
            print(f"  标准差: {np.nanstd(result):.6f}")
            print(f"结果矩阵:\n{result}")
            print("-" * 50)
            
        except Exception as e:
            print(f"函数 {func_name} 测试失败: {e}")
            print("-" * 50)
    
    # 测试需要特殊数据格式的函数
    print("\n=== 测试需要特殊数据格式的函数 ===")
    
    # 测试 trade_size_gini (需要 trade_size 数据)
    try:
        print("测试函数: cube2mat_trade_size_gini")
        trade_size = (np.random.rand(d0, d1, d2) * 100).astype(np.float32)
        trade_size[1, 2, 4:8] = np.nan
        cubes_map = {"interval_volume": trade_size}
        machine_alpha.cube2mat_trade_size_gini(result, cubes_map)
        print(f"trade_size_gini 结果:\n{result}")
        print("-" * 50)
    except Exception as e:
        print(f"trade_size_gini 测试失败: {e}")
        print("-" * 50)
    
    # 测试需要多个价格数据的函数
    try:
        print("测试函数: cube2mat_oc_efficiency_over_range")
        cubes_map = {"open": open_price, "last_price": close, "interval_high": high, "interval_low": low}
        machine_alpha.cube2mat_oc_efficiency_over_range(result, cubes_map)
        print(f"oc_efficiency_over_range 结果:\n{result}")
        print("-" * 50)
    except Exception as e:
        print(f"oc_efficiency_over_range 测试失败: {e}")
        print("-" * 50)

def test_alpha_example():
    """测试 alpha_example 函数"""
    print("=== 测试 alpha_example 函数 ===\n")
    
    d0, d1, d2 = 3, 5, 20
    
    # 构造测试数据
    np.random.seed(42)  # 设置随机种子以便结果可重现
    close = np.random.rand(d0, d1, d2).astype(np.float32) * 100  # 随机价格
    volume = (np.random.rand(d0, d1, d2) * 1000).astype(np.float32)  # 随机成交量
    volume[2, 3, 3:6] = np.nan  # 添加一些 NaN 值
    
    # 输出数组 result (d0, d1)
    result = np.empty((d0, d1), dtype=np.float32)
    
    try:
        print("测试函数: alpha_example")
        print(f"输入数据形状: close={close.shape}, volume={volume.shape}")
        
        # 调用 alpha_example 函数
        cubes_map = {"close": close, "volume": volume}
        machine_alpha.alpha_example(result, cubes_map)
        
        print(f"结果形状: {result.shape}")
        print(f"结果统计:")
        print(f"  非NaN值数量: {np.sum(~np.isnan(result))}")
        print(f"  最小值: {np.nanmin(result):.6f}")
        print(f"  最大值: {np.nanmax(result):.6f}")
        print(f"  平均值: {np.nanmean(result):.6f}")
        print(f"  标准差: {np.nanstd(result):.6f}")
        print(f"结果矩阵:\n{result}")
        
        # 验证结果：alpha_example 计算的是 VWAP (Volume Weighted Average Price)
        print("\n验证结果 (手动计算 VWAP 进行对比):")
        for i in range(d0):
            for j in range(d1):
                # 手动计算 VWAP
                valid_mask = ~(np.isnan(close[i, j, :]) | np.isnan(volume[i, j, :]) | (volume[i, j, :] <= 0))
                if np.any(valid_mask):
                    manual_vwap = np.sum(close[i, j, :][valid_mask] * volume[i, j, :][valid_mask]) / np.sum(volume[i, j, :][valid_mask])
                    print(f"  [{i},{j}] 手动VWAP: {manual_vwap:.6f}, 函数结果: {result[i, j]:.6f}, 差异: {abs(manual_vwap - result[i, j]):.8f}")
                else:
                    print(f"  [{i},{j}] 无有效数据，函数结果: {result[i, j]}")
        
        print("-" * 50)
        
    except Exception as e:
        print(f"alpha_example 测试失败: {e}")
        print("-" * 50)

if __name__ == "__main__":
    # test_machine_alpha_functions()
    print("\n" + "="*60 + "\n")
    test_alpha_example()