import argparse

def parse_commandline():
    parser = argparse.ArgumentParser(description='---------------------')
    parser.add_argument("--k_path", type=str, default="/one/two", help="keyword path")
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_commandline()
    with open(args.k_path, 'r') as fr:

# 命令行中使用python run.py --k_path /three/four  (换行用\)
