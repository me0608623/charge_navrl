import os
import re

def cleanup_latex(text):
    def fix_math_block(match):
        content = match.group(1).strip()
        # If the block contains more than 2 Chinese characters, it's probably not a pure math block
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        if chinese_chars > 2:
            # Revert to text block but keep the fixes for symbols if they look okay
            # Actually, better to just revert to a cleaner text representation
            content = content.replace("\\approx", "≈")
            content = content.replace("\\times", "×")
            content = content.replace("\\cdot", "·")
            content = content.replace("\\frac{mean}{std}", "mean/std")
            content = content.replace("\\_", "_")
            # Remove \text{...} and \operatorname{...}
            content = re.sub(r"\\text\{(.*?)\}", r"\1", content)
            content = re.sub(r"\\operatorname\{(.*?)\}", r"\1", content)
            return f"```text\n{content}\n```"
        return f"$$\n{content}\n$$"

    text = re.sub(r"\$\$\n(.*?)\n\$\$", fix_math_block, text, flags=re.DOTALL)
    
    # Fix double escaping or other artifacts
    text = text.replace("\\\\_", "\\_")
    
    return text

dir_path = "/home/aa/Documents/Obsidian Vault/消融前"
for filename in os.listdir(dir_path):
    if filename.endswith(".md"):
        file_path = os.path.join(dir_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        new_content = cleanup_latex(content)
        
        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Cleaned: {filename}")
        else:
            print(f"No changes: {filename}")
