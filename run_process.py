"""运行数据处理脚本的包装器"""
import sys
import traceback

if __name__ == '__main__':
    try:
        from src.process import process
        process()
    except Exception as e:
        print(f"错误: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
