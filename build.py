import os
import sys
import shutil
from pathlib import Path

def check_requirements():
    """检查必要的依赖是否已安装"""
    try:
        import PyInstaller
        print("PyInstaller已安装，版本:", PyInstaller.__version__)
    except ImportError:
        print("正在安装PyInstaller...")
        os.system("pip install pyinstaller")
        try:
            import PyInstaller
            print("PyInstaller安装成功，版本:", PyInstaller.__version__)
        except ImportError:
            print("PyInstaller安装失败，请手动安装后重试")
            sys.exit(1)

def build_exe():
    """构建可执行文件"""
    print("开始构建可执行文件...")
    
    # 确保weight目录存在
    weight_dir = Path("weight")
    if not weight_dir.exists() or not any(weight_dir.iterdir()):
        print("警告: weight目录不存在或为空，请确保模型文件已放置在weight目录中")
        return
    
    # 构建命令
    cmd = [
        "pyinstaller",
        "--name=SU7DetectSystem",
        "--windowed",  # 不显示控制台窗口
        "--onedir",   # 创建一个目录，包含exe和依赖
        "--add-data=weight;weight",  # 添加模型文件
        "--icon=NONE",  # 可以替换为自定义图标
        "main.py"
    ]
    
    # 执行构建命令
    os.system(" ".join(cmd))
    
    # 检查构建结果
    dist_dir = Path("dist") / "小米SU7检测系统"
    if dist_dir.exists():
        print(f"\n构建成功! 可执行文件位于: {dist_dir.absolute()}\\小米SU7检测系统.exe")
        print("\n使用说明:")
        print("1. 首次运行可能需要等待较长时间加载")
        print("2. 如果使用GPU加速，请确保已安装CUDA环境")
        print("3. 如需分发给其他用户，请将整个'小米SU7检测系统'文件夹一起分发")
    else:
        print("构建失败，请检查错误信息")

def clean_build_files():
    """清理构建过程中生成的临时文件"""
    print("\n是否清理构建过程中生成的临时文件? (y/n)")
    choice = input().lower()
    if choice == 'y':
        # 删除build目录和spec文件
        if os.path.exists("build"):
            shutil.rmtree("build")
        if os.path.exists("小米SU7检测系统.spec"):
            os.remove("小米SU7检测系统.spec")
        print("临时文件已清理")

def main():
    print("===== 小米SU7检测系统打包工具 =====\n")
    
    # 检查依赖
    check_requirements()
    
    # 构建可执行文件
    build_exe()
    
    # 清理临时文件
    clean_build_files()
    
    print("\n打包过程完成!")

if __name__ == "__main__":
    main()