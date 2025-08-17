#!/usr/bin/env python3
"""
数据预处理脚本：将TSV格式的情感分析数据转换为MLX-LM训练所需的JSONL格式

输入格式 (TSV):
qid	label	text_a
0	1	这间酒店环境和服务态度亦算不错...

输出格式 (JSONL):
{"prompt": "请判断以下文本的情感倾向，正面回复'1'，负面回复'0'。\n文本：这间酒店环境和服务态度亦算不错...\n情感：", "completion": "1"}
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple


def create_prompt_completion_pair(text: str, label: int) -> Dict[str, str]:
    """
    创建prompt-completion对
    
    Args:
        text: 原始文本
        label: 情感标签 (0=负面, 1=正面)
    
    Returns:
        包含prompt和completion的字典
    """
    prompt = f"请判断以下文本的情感倾向，正面回复'1'，负面回复'0'。\n文本：{text}\n情感："
    completion = str(label)
    
    return {
        "prompt": prompt,
        "completion": completion
    }


def load_tsv_data(file_path: str) -> pd.DataFrame:
    """
    加载TSV格式的数据
    
    Args:
        file_path: TSV文件路径
    
    Returns:
        pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"成功加载数据: {len(df)} 条记录")
        print(f"数据列: {list(df.columns)}")
        
        # 检查必需的列是否存在
        required_columns = ['text_a', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必需的列: {missing_columns}")
        
        # 检查标签分布
        label_counts = df['label'].value_counts().sort_index()
        print(f"标签分布:")
        for label, count in label_counts.items():
            print(f"  标签 {label}: {count} 条 ({count/len(df)*100:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise


def convert_to_jsonl_format(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    将DataFrame转换为JSONL格式的数据
    
    Args:
        df: 包含text_a和label列的DataFrame
    
    Returns:
        JSONL格式的数据列表
    """
    jsonl_data = []
    
    for _, row in df.iterrows():
        text = str(row['text_a']).strip()
        label = int(row['label'])
        
        # 跳过空文本
        if not text or text == 'nan':
            continue
        
        # 创建prompt-completion对
        pair = create_prompt_completion_pair(text, label)
        jsonl_data.append(pair)
    
    print(f"转换完成: {len(jsonl_data)} 条有效记录")
    return jsonl_data


def split_data(data: List[Dict[str, str]], train_ratio: float = 0.8, 
               valid_ratio: float = 0.1, random_state: int = 42) -> Tuple[List, List, List]:
    """
    分割数据为训练集、验证集和测试集
    
    Args:
        data: JSONL格式的数据
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        random_state: 随机种子
    
    Returns:
        (训练集, 验证集, 测试集)
    """
    # 提取标签用于分层抽样
    labels = [int(item['completion']) for item in data]
    
    # 首先分割出训练集和临时集
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, 
        train_size=train_ratio, 
        stratify=labels, 
        random_state=random_state
    )
    
    # 计算验证集在临时集中的比例
    temp_ratio = 1 - train_ratio
    valid_ratio_in_temp = valid_ratio / temp_ratio
    
    # 分割验证集和测试集
    valid_data, test_data = train_test_split(
        temp_data, 
        train_size=valid_ratio_in_temp, 
        stratify=temp_labels, 
        random_state=random_state
    )
    
    print(f"数据分割完成:")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(valid_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    
    return train_data, valid_data, test_data


def save_jsonl(data: List[Dict[str, str]], file_path: str) -> None:
    """
    保存数据为JSONL格式
    
    Args:
        data: 要保存的数据
        file_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"已保存 {len(data)} 条记录到: {file_path}")


def create_sample_data_info(output_dir: str, train_data: List, valid_data: List, test_data: List) -> None:
    """
    创建数据统计信息文件
    """
    info = {
        "dataset_info": {
            "total_samples": len(train_data) + len(valid_data) + len(test_data),
            "train_samples": len(train_data),
            "valid_samples": len(valid_data),
            "test_samples": len(test_data)
        },
        "sample_examples": {
            "train_example": train_data[0] if train_data else None,
            "valid_example": valid_data[0] if valid_data else None,
            "test_example": test_data[0] if test_data else None
        }
    }
    
    info_path = os.path.join(output_dir, "data_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    print(f"数据信息已保存到: {info_path}")


def main():
    parser = argparse.ArgumentParser(description="将TSV情感分析数据转换为MLX-LM训练格式")
    parser.add_argument("--input_file", type=str, required=True, help="输入TSV文件路径")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Qwen3 情感分析数据预处理")
    print("=" * 50)
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}")
        return
    
    # 加载数据
    print(f"\n1. 加载数据: {args.input_file}")
    df = load_tsv_data(args.input_file)
    
    # 转换格式
    print(f"\n2. 转换数据格式")
    jsonl_data = convert_to_jsonl_format(df)
    
    if not jsonl_data:
        print("错误: 没有有效的数据可以转换")
        return
    
    # 分割数据
    print(f"\n3. 分割数据集")
    train_data, valid_data, test_data = split_data(
        jsonl_data, 
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        random_state=args.random_state
    )
    
    # 保存数据
    print(f"\n4. 保存处理后的数据到: {args.output_dir}")
    save_jsonl(train_data, os.path.join(args.output_dir, "train.jsonl"))
    save_jsonl(valid_data, os.path.join(args.output_dir, "valid.jsonl"))
    save_jsonl(test_data, os.path.join(args.output_dir, "test.jsonl"))
    
    # 创建数据信息文件
    create_sample_data_info(args.output_dir, train_data, valid_data, test_data)
    
    print(f"\n✅ 数据预处理完成!")
    print(f"训练数据: {args.output_dir}/train.jsonl")
    print(f"验证数据: {args.output_dir}/valid.jsonl")
    print(f"测试数据: {args.output_dir}/test.jsonl")
    print(f"数据信息: {args.output_dir}/data_info.json")


if __name__ == "__main__":
    main()
