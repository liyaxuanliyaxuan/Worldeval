import h5py
import argparse
import numpy as np

def preview_h5_file(file_path):
    """
    预览 HDF5 文件的内容，显示数据集结构和基本统计信息
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n{'='*50}")
            print(f"文件路径: {file_path}")
            print(f"{'='*50}\n")
            # if 'reasoning' in f:
            #     print("reasoning: ", f"{f['reasoning'][:]}")
            # if 'raw_language' in f:
            #     print("raw_language: ", f"{f['raw_language'][:]}")
            # if 'language_raw' in f:
            #     print("language_raw: ", f"{f['language_raw'][:]}")
            # print(f"{f['reasoning'][:].shape}")

            def print_dataset_info(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"\n数据集名称: {name}")
                    print(f"形状: {obj.shape}")
                    print(f"数据类型: {obj.dtype}")
                    
                    # 如果数据集不是太大，显示一些基本统计信息
                    if np.prod(obj.shape) < 1000000:  # 限制大小以避免内存问题
                        data = obj[()]
                        if np.issubdtype(obj.dtype, np.number):
                            print(f"最小值: {np.min(data)}")
                            print(f"最大值: {np.max(data)}")
                            print(f"平均值: {np.mean(data)}")
                            print(f"标准差: {np.std(data)}")
                    
                    print("-" * 30)
            
            # 遍历文件中的所有数据集
            f.visititems(print_dataset_info)

    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}")
        print(f"错误信息: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='预览 HDF5 文件内容')
    parser.add_argument('file_path', type=str, help='HDF5 文件路径')
    args = parser.parse_args()
    
    preview_h5_file(args.file_path)

if __name__ == "__main__":
    main()
