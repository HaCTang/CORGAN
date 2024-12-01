# import subprocess
# import re

# output_file = "summary_results.txt"

# # 正则表达式匹配 "Summary Results" 部分
# summary_pattern = re.compile(r"~~~ Summary Results ~~~\n(.*?:.*\n)+", re.MULTILINE)

# with open(output_file, "w") as f:
#     for i in range(10):
#         # 运行 example.py 并捕获输出
#         process = subprocess.run(["python", "example.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
#         # 搜索输出中 "Summary Results" 部分
#         match = summary_pattern.search(process.stdout)
#         if match:
#             f.write(f"Run {i+1}:\n")
#             f.write(match.group(0))  # 写入匹配的 "Summary Results" 部分
#             f.write("\n")
#         else:
#             f.write(f"Run {i+1}: No summary results found\n\n")

import subprocess
import re

output_file = "summary_results.txt"
# 正则表达式匹配 "Summary Results" 部分
summary_pattern = re.compile(r"~~~ Summary Results ~~~\n(.*?:.*\n)+", re.MULTILINE)

with open(output_file, "w") as f:
    for i in range(30):
        print(f"Run {i+1}")
        # 运行 example.py 并捕获输出
        process = subprocess.run(
            ["python", "example.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True  # 替代 text=True
        )
        
        # 搜索输出中 "Summary Results" 部分
        match = summary_pattern.search(process.stdout)
        if match:
            f.write(f"Run {i+1}:\n")
            f.write(match.group(0))  # 写入匹配的 "Summary Results" 部分
            f.write("\n")
        else:
            f.write(f"Run {i+1}: No summary results found\n\n")
