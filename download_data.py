from pathlib import Path
from icrawler.builtin import BingImageCrawler

# =========================
# 1. 项目根目录
# =========================
PROJECT_DIR = Path(r"D:\AI_engineer\multimodal_rl_agent")

# =========================
# 2. 自动创建文件夹
# =========================
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw_images"
SFT_DIR = DATA_DIR / "sft"
PREFERENCE_DIR = DATA_DIR / "preference"
EVAL_DIR = DATA_DIR / "eval"

for folder in [DATA_DIR, RAW_DIR, SFT_DIR, PREFERENCE_DIR, EVAL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("文件夹创建完成：")
print(DATA_DIR)
print(RAW_DIR)
print(SFT_DIR)
print(PREFERENCE_DIR)
print(EVAL_DIR)

# =========================
# 3. 下载关键词
# =========================
queries = [
    "science lab setup",
    "chemistry lab apparatus",
    "physics experiment setup",
    "laboratory equipment table",
    "electrical experiment setup",
    "control panel laboratory"
]

max_num_per_query = 5

# =========================
# 4. 下载图片
# =========================
for query in queries:
    print(f"\n开始下载关键词: {query}")

    crawler = BingImageCrawler(
        storage={"root_dir": str(RAW_DIR)}
    )

    crawler.crawl(
        keyword=query,
        max_num=max_num_per_query,
        min_size=(400, 300)
    )

print("\n图片下载完成，请去 data/raw_images 查看。")