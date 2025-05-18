import torch
import argparse
from pathlib import Path
import numpy


def preview_pt_file(pt_path, max_items=10):
    """预览PT文件中保存的列表数据
    
    Args:
        pt_path: PT文件路径
        max_items: 最多显示的列表项数量，默认10项
    """
    # 检查文件是否存在
    if not Path(pt_path).exists():
        print(f"错误: 文件 {pt_path} 不存在")
        return

    try:
        # 加载PT文件
        data = torch.load(pt_path, weights_only=False)
        
        # 确保加载的数据是列表或字典类型
        if not isinstance(data, (list, dict)):
            print(f"警告: 加载的数据不是列表或字典类型，而是 {type(data)}")
            return
            
        # 获取列表或字典总长度
        total_length = len(data)
        print(f"\n文件: {pt_path}")
        print(f"列表/字典总长度: {total_length}")
        
        # 显示前几项内容
        print(f"\n前 {min(max_items, total_length)} 项内容:")
        if isinstance(data, list):
            for i, item in enumerate(data[:max_items]):
                print(f"\n[{i}] 类型: {type(item)}")
                if isinstance(item, (torch.Tensor, list, tuple, dict)):
                    print(f"形状/长度: {len(item) if isinstance(item, (list, tuple, dict)) else item.shape}")
                print(f"内容: {item}")
        elif isinstance(data, dict):
            for i, (key, value) in enumerate(data.items()):
                if i >= max_items:
                    break
                print(f"\n键: {key} 类型: {type(value)}")
                if isinstance(value, (torch.Tensor, list, tuple, dict)):
                    print(f"形状/长度: {len(value) if isinstance(value, (list, tuple, dict)) else value.shape}")
                # if key == 'encoded_action':              
                #     if isinstance(value, list):
                #         print(f"encoded_action[0] 形状: {value[0].shape if isinstance(value[0], (torch.Tensor, numpy.ndarray)) else 'N/A'}")
                #     else:
                #         print(f"encoded_action 形状: {value.shape if isinstance(value, (torch.Tensor, numpy.ndarray)) else 'N/A'}")
                if isinstance(value, list) and len(value) > 0:
                    print(f"第一个项: {value[0]}")
                    print(f"第一个项形状: {value[0].shape if isinstance(value[0], (torch.Tensor, numpy.ndarray)) else 'N/A'}")
                # print(f"内容: {value}")
            
    except Exception as e:
        print(f"错误: 预览文件时发生异常: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='预览PT文件中保存的列表数据')
    parser.add_argument('pt_path', type=str, help='PT文件路径')
    parser.add_argument('--max-items', type=int, default=2, help='最多显示的列表项数量')
    
    args = parser.parse_args()
    preview_pt_file(args.pt_path, args.max_items)

if __name__ == '__main__':
    main() 