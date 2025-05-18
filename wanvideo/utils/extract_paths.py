import json
import argparse

def extract_paths(json_file, output_file):
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 提取所有file_path并写入到文本文件
    with open(output_file, 'w') as f:
        for item in data:
            if 'file_path' in item:
                f.write(item['file_path'] + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从JSON文件中提取file_path到文本文件')
    parser.add_argument('json_file', help='输入的JSON文件路径')
    parser.add_argument('output_file', help='输出的文本文件路径')
    
    args = parser.parse_args()
    extract_paths(args.json_file, args.output_file) 